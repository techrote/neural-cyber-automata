"""Execute the frozen morphology admission gate; a blocked run starts no trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

from nca.controllers.morphology.admission import prepare_comparison
from nca.contracts import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/morphology/comparison-v1.json"))
    parser.add_argument("--declaration", type=Path, default=Path("configs/morphology/entry-gate-v1.json"))
    parser.add_argument("--result", type=Path, default=Path("docs/evidence/t106-prerequisite-result.json"))
    parser.add_argument("--profiles", type=Path, default=Path("configs/morphology/profiles-v1.json"))
    parser.add_argument("--output", type=Path, required=True, help="new immutable result directory")
    parser.add_argument("--expect-blocked", action="store_true", help="zero exit only when the expected closed prerequisite is recorded")
    args = parser.parse_args()
    record = prepare_comparison(root=args.source, config_path=args.config, declaration_path=args.declaration,
                                result_path=args.result, profiles_path=args.profiles, output=args.output)
    print(canonical_json({"status": record["status"], "admitted": record["admitted"],
                          "work": record["work"], "sha256": record["sha256"], "errors": record["errors"]}))
    return 0 if args.expect_blocked and record["status"] == "training_blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
