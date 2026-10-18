#!/usr/bin/env python3
"""Check project documentation against the claudev documentation standard.

What it checks (it cannot judge whether a claim is TRUE; audits do that):
  * metadata block on every doc: title, kind, summary, owns, verified
  * status tags allowed for each document kind (no plans hiding in state pages)
  * FORBIDDEN has "until: ..." or "permanent"; TARGET cites D-###; SUPERSEDED cites "by X-###";
    PLANNED / IN-PROGRESS / BLOCKED cite T-###
  * every ID entry in decisions / constraints / ideas / research pages carries a status tag
  * IDs defined once; references to undefined IDs (warning)
  * each "owns" topic has exactly one home
  * relative links and #anchors resolve
  * size budgets, future tense in state pages, banned phrases, stale "verified" dates (warnings)

Usage:
  python tools/check_docs.py                     # check AGENTS.md + docs/ under the current directory
  python tools/check_docs.py --strict            # warnings fail too
  python tools/check_docs.py --root DIR --paths AGENTS.md docs
  python tools/check_docs.py --allow-placeholders  # for templates containing <...> and ### placeholders

Standard library only. Exit code 1 when there are errors (or warnings with --strict).
"""

import argparse
import datetime as dt
import os
import re
import sys

TAGS = [
    "CURRENT", "PARTIAL", "DEFECT", "TARGET", "PLANNED", "IN-PROGRESS", "BLOCKED",
    "PROPOSED", "IDEA", "DEFERRED", "REJECTED", "ABANDONED", "SUPERSEDED",
    "REQUIRED", "FORBIDDEN", "PREFERRED", "ACCEPTED",
    "ESTABLISHED", "SUPPORTED", "TENTATIVE", "REFUTED", "OPEN",
]

ANY = set(TAGS)
ALLOWED = {
    "state": {"CURRENT", "PARTIAL", "DEFECT"},
    "architecture": {"CURRENT", "PARTIAL", "DEFECT", "TARGET"},
    "decisions": {"ACCEPTED", "PROPOSED", "SUPERSEDED", "REJECTED"},
    "constraints": {"REQUIRED", "FORBIDDEN", "PREFERRED", "SUPERSEDED"},
    "roadmap": {"PLANNED", "IN-PROGRESS", "BLOCKED", "DEFERRED"},
    "ideas": {"IDEA", "PROPOSED", "DEFERRED", "REJECTED", "ABANDONED", "SUPERSEDED"},
    "research": {"ESTABLISHED", "SUPPORTED", "TENTATIVE", "REFUTED", "OPEN", "SUPERSEDED"},
    "reference": {"CURRENT", "PARTIAL", "DEFECT", "TARGET"},
    "gotchas": set(),
    "evidence": set(),
    "navigation": set(),
    "guide": ANY,
    "history": ANY,
    "scratch": ANY,
}
EVERY_ENTRY_TAGGED = {"decisions", "constraints", "ideas", "research"}
NO_CLAIM_CHECKS = {"guide", "history", "scratch"}  # pages that explain or archive; never authority
REQUIRED_FIELDS = ["title", "kind", "summary", "owns", "verified"]

SIZE_BUDGET = {"AGENTS.md": 120}
DEFAULT_BUDGET = 300
FUTURE_WORDS = re.compile(r"\b(will|going to|eventually|planned|TODO|soon)\b", re.IGNORECASE)
BANNED = re.compile(
    r"effectively complete|basically done|essentially (?:done|working|complete)|should work|"
    r"production[- ]ready|fully implemented",
    re.IGNORECASE,
)

TAG_RE = re.compile(r"\[(" + "|".join(re.escape(t) for t in TAGS) + r")(?=[\]\s:])([^\]]*)\]")
ID_NUM = r"[DCTIGRFX]-\d{3,}"

ID_DEF_HEADING = re.compile(r"^#{2,4}\s+(" + ID_NUM + r")\b")
ID_DEF_TABLE = re.compile(r"^\|\s*(" + ID_NUM + r")\s*\|")
ID_REF = re.compile(r"(?<![\w-])(" + ID_NUM + r")(?![\w-])")
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:\s*@\s*\S+)?$")


class Report:
    def __init__(self):
        self.errors = 0
        self.warnings = 0

    def error(self, path, line, msg):
        self.errors += 1
        print(f"ERROR {path}:{line}: {msg}")

    def warn(self, path, line, msg):
        self.warnings += 1
        print(f"WARN  {path}:{line}: {msg}")


def is_placeholder(value):
    return "<" in value or "###" in value


def parse_front_matter(lines):
    """Return (fields, body_start_index). Supports `key: value` and `key: [a, b]`."""
    if not lines or lines[0].strip() != "---":
        return None, 0
    fields = {}
    for i in range(1, len(lines)):
        line = lines[i].rstrip("\n")
        if line.strip() == "---":
            return fields, i + 1
        m = re.match(r"^([A-Za-z_-]+):\s*(.*)$", line)
        if m:
            key, value = m.group(1), m.group(2).strip()
            value = re.sub(r"\s+#.*$", "", value)  # trailing comment
            if value.startswith("[") and value.endswith("]"):
                value = [v.strip() for v in value[1:-1].split(",") if v.strip()]
            fields[key] = value
    return None, 0  # unterminated


def strip_code(lines):
    """Yield (line_number, text) with fenced blocks removed and inline code blanked."""
    fence = None
    for n, line in enumerate(lines, 1):
        s = line.lstrip()
        m = re.match(r"(`{3,}|~{3,})", s)
        if m:
            if fence is None:
                fence = m.group(1)[0] * len(m.group(1))
                continue
            if s.startswith(fence):
                fence = None
                continue
        if fence is not None:
            continue
        yield n, re.sub(r"`[^`]*`", "``", line)


def slugify(heading):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = text.replace("`", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors_of(path, cache):
    if path in cache:
        return cache[path]
    slugs, counts = set(), {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        cache[path] = slugs
        return slugs
    for _, line in strip_code(lines):
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            base = slugify(m.group(1))
            n = counts.get(base, 0)
            slugs.add(base if n == 0 else f"{base}-{n}")
            counts[base] = n + 1
    cache[path] = slugs
    return slugs


def collect_files(root, paths):
    files = []
    for p in paths:
        full = os.path.join(root, p)
        if os.path.isfile(full) and full.endswith(".md"):
            files.append(full)
        elif os.path.isdir(full):
            for dirpath, dirnames, filenames in os.walk(full):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "node_modules"]
                files.extend(os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(".md"))
    return sorted(set(files))


def check(root, paths, strict, allow_placeholders, stale_days):
    rep = Report()
    files = collect_files(root, paths)
    if not files:
        rep.error(root, 0, f"no markdown files found under {paths}")
    owners, id_defs, id_refs, docs = {}, {}, [], []
    anchor_cache = {}
    today = dt.date.today()

    for path in files:
        rel = os.path.relpath(path, root)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        fields, start = parse_front_matter(lines)
        if fields is None:
            rep.error(rel, 1, "missing metadata block (--- title/kind/summary/owns/verified ---)")
            continue
        for key in REQUIRED_FIELDS:
            if key not in fields or fields[key] in ("", []) and key != "owns":
                rep.error(rel, 1, f"metadata field '{key}' missing or empty")
        kind = fields.get("kind", "")
        if kind not in ALLOWED:
            rep.error(rel, 1, f"unknown kind '{kind}' (allowed: {', '.join(sorted(ALLOWED))})")
            continue
        for key, value in fields.items():
            text = " ".join(value) if isinstance(value, list) else value
            if is_placeholder(text) and not allow_placeholders and key in REQUIRED_FIELDS:
                rep.error(rel, 1, f"metadata field '{key}' still has a placeholder")
        verified = fields.get("verified", "")
        if isinstance(verified, str) and not is_placeholder(verified):
            m = DATE_RE.match(verified)
            if not m:
                rep.error(rel, 1, "verified must be 'YYYY-MM-DD' or 'YYYY-MM-DD @ <commit>'")
            elif kind not in ("evidence", "history"):
                age = (today - dt.date.fromisoformat(m.group(1))).days
                if age > stale_days:
                    rep.warn(rel, 1, f"verified {age} days ago; re-check against the code")
        owns = fields.get("owns", [])
        for topic in owns if isinstance(owns, list) else [owns]:
            owners.setdefault(topic, []).append(rel)
        docs.append((path, rel, kind, lines, start))

        # size budgets
        budget = SIZE_BUDGET.get(os.path.basename(path), DEFAULT_BUDGET)
        if len(lines) > budget:
            rep.warn(rel, len(lines), f"{len(lines)} lines exceeds budget of {budget}; split behind an index")

        body = list(strip_code(lines))
        allowed = ALLOWED[kind]
        claims = kind not in NO_CLAIM_CHECKS
        section_id, section_line, section_tagged = None, 0, False

        def close_section():
            if section_id and kind in EVERY_ENTRY_TAGGED and not section_tagged:
                rep.error(rel, section_line, f"{section_id} has no status tag")

        for n, line in body:
            if n <= start:
                continue
            # ID definitions and sections
            mh = ID_DEF_HEADING.match(line)
            if mh or re.match(r"^#{1,4}\s", line):
                close_section()
                section_id, section_line, section_tagged = (mh.group(1), n, False) if mh else (None, 0, False)
            if claims:
                md = mh or ID_DEF_TABLE.match(line)
                if md:
                    id_defs.setdefault(md.group(1), []).append(f"{rel}:{n}")
                for ref in ID_REF.findall(line):
                    id_refs.append((ref, rel, n))
            # tags
            for tag, rest in TAG_RE.findall(line):
                section_tagged = True
                if not claims:
                    continue
                if tag not in allowed:
                    where = "evidence/gotchas/navigation pages carry no status tags" if not allowed else \
                        f"allowed here: {', '.join(sorted(allowed))}"
                    rep.error(rel, n, f"[{tag}] not allowed in a '{kind}' page ({where}); move the item to the page for its status")
                rest = rest.strip()
                placeholder_ok = allow_placeholders and is_placeholder(rest)
                if tag == "FORBIDDEN" and not re.search(r"until:\s*\S|permanent", rest):
                    rep.error(rel, n, "[FORBIDDEN] needs 'until: <condition>' or 'permanent'")
                if tag == "TARGET" and not (re.search(r"\bD-\d{3,}", rest) or placeholder_ok):
                    rep.error(rel, n, "[TARGET] must cite the decision: [TARGET D-###]")
                if tag == "SUPERSEDED" and not (re.search(r"by\s+" + ID_NUM, rest) or placeholder_ok):
                    rep.error(rel, n, "[SUPERSEDED] must say what replaced it: [SUPERSEDED by X-###]")
                if tag in ("PLANNED", "IN-PROGRESS", "BLOCKED") and not (re.search(r"\bT-\d{3,}", rest) or placeholder_ok):
                    rep.error(rel, n, f"[{tag}] must cite its task: [{tag} T-###]")
            # wording
            if kind == "state" and FUTURE_WORDS.search(line) and not line.lstrip().startswith(("<!--", ">")):
                rep.warn(rel, n, "future/plan wording in a state page; plans belong in ROADMAP.md, ideas in IDEAS.md")
            if claims and BANNED.search(line):
                rep.warn(rel, n, f"banned phrase '{BANNED.search(line).group(0)}'; name the closure rung instead")
            if not allow_placeholders and claims and re.search(r"\b[DCTIGRFX]-###", line):
                rep.warn(rel, n, "unfilled ID placeholder")
            # links
            for target in LINK_RE.findall(line):
                if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("<"):
                    continue  # external (http:, mailto:) or placeholder
                target_path, _, anchor = target.partition("#")
                resolved = os.path.normpath(os.path.join(os.path.dirname(path), target_path)) if target_path else path
                if not os.path.exists(resolved):
                    rep.error(rel, n, f"broken link: {target}")
                    continue
                if anchor and resolved.endswith(".md") and anchor not in anchors_of(resolved, anchor_cache):
                    rep.error(rel, n, f"broken anchor: {target}")
        close_section()

    for ident, places in id_defs.items():
        if len(places) > 1:
            rep.error(places[1].split(":")[0], places[1].split(":")[1], f"{ident} defined more than once: {', '.join(places)}")
    for ref, rel, n in id_refs:
        if ref not in id_defs:
            rep.warn(rel, n, f"{ref} is referenced but not defined in these docs")
    for topic, homes in owners.items():
        if len(homes) > 1:
            rep.error(homes[1], 1, f"topic '{topic}' is owned by more than one page: {', '.join(homes)}")

    print(f"\nchecked {len(files)} file(s): {rep.errors} error(s), {rep.warnings} warning(s)")
    if rep.errors or (strict and rep.warnings):
        return 1
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root (default: current directory)")
    ap.add_argument("--paths", nargs="+", default=["AGENTS.md", "docs"], help="files/dirs to check, relative to root")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--allow-placeholders", action="store_true", help="accept <...> and ### placeholders (templates)")
    ap.add_argument("--stale-days", type=int, default=120, help="warn when 'verified' is older than this")
    args = ap.parse_args(argv)
    return check(os.path.abspath(args.root), args.paths, args.strict, args.allow_placeholders, args.stale_days)


if __name__ == "__main__":
    sys.exit(main())
