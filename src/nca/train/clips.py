"""Matched recorded-pose GIFs; no game renderer or invented post-terminal motion.

This optional visualizer needs Pillow only, not PyTorch. It can run separately
from the pinned learner environment after an immutable screen completes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Mapping

from nca.contracts import canonical_json
from nca.data import EpisodeRecord, read_episode


def render_matched(episodes: Mapping[str, EpisodeRecord]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    records = list(episodes.values())
    if len(records) != 3 or len({episode.scenario_family for episode in records}) != 1:
        raise ValueError("a matched clip requires baseline/BC/PPO for one scenario family")
    if len({episode.profile_sha256 for episode in records}) != 1 or len({episode.config_sha256 for episode in records}) != 1:
        raise ValueError("comparison actors have different initial scenarios or physical capability")
    if len({canonical_json(episode.host_identity) for episode in records}) != 1:
        raise ValueError("comparison host/runtime identities differ")
    timelines = [[episode.initial_observation] + [tick.observation for event in episode.events for tick in event.ticks]
                 for episode in records]
    if any(len(rows) < 2 for rows in timelines):
        raise ValueError("a matched clip needs actual completed body observations")
    contexts = [next(event.context for event in episode.events if event.context is not None) for episode in records]
    if len({context.target for context in contexts}) != 1:
        raise ValueError("matched clip goals differ")
    target = contexts[0].target
    points = [part["position"] for rows in timelines for row in rows for part in row["parts"]] + [target]
    low_x, high_x = min(p[0] for p in points) - 12, max(p[0] for p in points) + 12
    low_y, high_y = min(p[1] for p in points) - 12, max(p[1] for p in points) + 12
    width, height = 420, 336
    scale = min(4., 384 / max(1., high_x - low_x), 218 / max(1., high_y - low_y))
    try:
        font, small = ImageFont.truetype("DejaVuSans.ttf", 13), ImageFont.truetype("DejaVuSans.ttf", 10)
    except OSError:
        font = small = ImageFont.load_default()
    by_tick = [{row["tick"]: row for row in timeline} for timeline in timelines]
    last_ticks = [timeline[-1]["tick"] for timeline in timelines]
    terrain = [{} for _ in records]
    frames = []
    for tick in range(max(last_ticks) + 1):
        for panel, timeline in enumerate(by_tick):
            if tick not in timeline:
                continue
            patch = timeline[tick].get("cell_patch", {})
            patch_width = int(patch.get("size", (0, 0))[0])
            origin, bodies = patch.get("origin", (0, 0)), patch.get("body_ids", ())
            if patch_width:
                for cell, material in enumerate(patch.get("materials", ())):
                    if cell < len(bodies) and bodies[cell] != 0:
                        continue
                    terrain[panel][(origin[0] + cell % patch_width, origin[1] + cell // patch_width)] = material
        if tick % 6:
            continue
        frame = Image.new("RGB", (width * 3, height), "#101722")
        draw = ImageDraw.Draw(frame)
        for panel, (label, episode) in enumerate(episodes.items()):
            actual_tick = min(tick, last_ticks[panel])
            observation = by_tick[panel][actual_tick]
            offset = panel * width

            def pixel(point):
                return (offset + 18 + (point[0] - low_x) * scale, 65 + (point[1] - low_y) * scale)

            draw.text((offset + 14, 9), label, font=font, fill="#e9f2ff")
            ended = tick > last_ticks[panel]
            detail = (f"END {actual_tick / 60:.2f}s: {episode.reason}; final pose held" if ended else
                      f"Recorded ACT004 body poses | {tick / 60:.2f}s | tick {tick}")
            draw.text((offset + 14, 29), detail, font=small, fill="#b6c2d6")
            for (x, y), material in terrain[panel].items():
                if material:
                    draw.rectangle((pixel((x, y)), pixel((x + 1, y + 1))), fill="#465164" if material == 1 else "#596c7f")
            goal = pixel(target)
            draw.line((goal[0] - 5, goal[1], goal[0] + 5, goal[1]), fill="#e4b365", width=2)
            draw.line((goal[0], goal[1] - 5, goal[0], goal[1] + 5), fill="#e4b365", width=2)
            for part in observation["parts"]:
                c, s = math.cos(part["rotation"]), math.sin(part["rotation"])
                x, y = part["size"][0] / 2, part["size"][1] / 2
                corners = [pixel((part["position"][0] + a * c - b * s, part["position"][1] + a * s + b * c))
                           for a, b in ((-x, -y), (x, -y), (x, y), (-x, y))]
                draw.polygon(corners, fill="#8e7487" if ended else "#ef829b" if part["index"] == 0 else "#77c1ed",
                             outline="#dcecff")
            speed = math.hypot(*observation["velocity"])
            draw.text((offset + 14, 296), f"{episode.scenario_family} | speed {speed:.1f} cells/s", font=small, fill="#c4d1e7")
            draw.text((offset + 14, 313), "Receipt poses/local cells; owner feel remains unjudged", font=small, fill="#9badc8")
            if panel:
                draw.line((offset, 0, offset, height), fill="#4b5c75")
        frames.append(frame)
    stream = io.BytesIO()
    frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:], duration=100,
                   loop=0, disposal=2, optimize=False)
    return stream.getvalue()


def render_screen(directory: str | Path):
    root = Path(directory)
    summary = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if summary["status"] != "completed":
        raise ValueError("incomplete screens retain their trajectories; do not render them as matched complete trials")
    rows = {(row["scenario_family"], row["policy_id"]): row for row in summary["results"] if row["split"] == "evaluation"}
    clips = []
    for scenario in manifest["config"]["evaluation_families"]:
        for seed in manifest["config"]["training_seeds"]:
            ids = ("baseline", f"bc-seed{seed}-final", f"residual-seed{seed}-final")
            matched = {name: read_episode(root / rows[(scenario["id"], name)]["artifact"]) for name in ids}
            data = render_matched(matched)
            filename = f"{scenario['id']}-seed{seed}-matched-poses.gif"
            path = root / "clips" / filename
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            clips.append({"file": "clips/" + filename, "sha256": hashlib.sha256(data).hexdigest(),
                          "bytes": len(data), "scenario": scenario["id"], "training_seed": seed})
    value = {"format": "nca-matched-receipt-poses/v1", "clips": clips,
             "source_summary_sha256": summary["sha256"], "in_game_renderer": False,
             "post_terminal": "last measured pose held and explicitly labelled; no new observations invented"}
    with (root / "clips.json").open("xb") as stream:
        stream.write((canonical_json(value) + "\n").encode())
    return value


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = render_screen(args.directory)
    print(json.dumps({"clips": len(result["clips"]), "format": result["format"]}))
