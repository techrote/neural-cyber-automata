"""Render paired actual own-part poses; no simulated or invented game frames."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import math
import os
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from nca.contracts import ContractError, strict_json_loads
from nca.export.build import read_json, immutable_json


def _episode(path):
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return strict_json_loads(data, max_bytes=32 * 1024 * 1024)


def _write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pose-", delete=False) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ContractError("immutable_exists", path.name) from exc
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def render(root: Path):
    spec, spec_hash = read_json(root / "clip-spec.json")
    left_bound, top_bound, right_bound, bottom_bound = spec["camera"]["world_bounds"]
    width, height = spec["camera"]["panel_pixels"]
    scale_x, scale_y = width / (right_bound-left_bound), height / (bottom_bound-top_bound)
    if not math.isclose(scale_x, scale_y):
        raise ContractError("pose_camera", "equal spatial scale on both axes required")
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=13)
    records = []
    for pair in spec["pairs"]:
        episodes = [_episode(root / pair[key]) for key in ("left", "right")]
        frames, ticks = [], list(range(0, pair["ticks"] + 1, spec["frame_stride"]))
        indexed = [{frame["tick"]: frame for frame in episode["frames"]} for episode in episodes]
        for tick in ticks:
            canvas = Image.new("RGB", (2 * width + 32, height + 110), "#0d1422")
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 12), f"T109  {pair['id']}  |  completed tick {tick} / {tick/60:.2f}s", font=font, fill="#eef3fc")
            for side, (episode, rows, label) in enumerate(zip(episodes, indexed, pair["labels"])):
                if tick not in rows:
                    raise ContractError("pose_tick", "matched actual frame is absent")
                frame = rows[tick]
                x_origin, y_origin = 8 + side * (width + 16), 65
                panel = Image.new("RGB", (width, height), "#172033")
                scene = ImageDraw.Draw(panel)
                def xy(point):
                    return ((point[0] - left_bound) * scale_x, (point[1] - top_bound) * scale_y)
                draw.text((x_origin + 10, 40), label, font=font, fill="#a8cee9")
                for x in range(math.ceil(left_bound/16)*16, right_bound, 16):
                    scene.line((xy((x, top_bound)), xy((x, bottom_bound))), fill="#212d43")
                for y in range(math.ceil(top_bound/16)*16, bottom_bound, 16):
                    scene.line((xy((left_bound, y)), xy((right_bound, y))), fill="#212d43")
                target = xy(episode["case"]["target"])
                scene.line((target[0]-7, target[1], target[0]+7, target[1]), fill="#f3c66b", width=2)
                scene.line((target[0], target[1]-7, target[0], target[1]+7), fill="#f3c66b", width=2)
                trail = [xy(rows[t]["position"]) for t in sorted(rows) if t <= tick]
                if len(trail) > 1:
                    scene.line(trail, fill="#54748d", width=2)
                for part, authored in zip(frame["parts"], episode["profile"]["parts"]):
                    if part["id"] != authored["id"]:
                        raise ContractError("pose_part", "profile shape and observed part differ")
                    px, py = part["position"]
                    sx, sy = authored["size"]
                    c, s = math.cos(part["rotation"]), math.sin(part["rotation"])
                    polygon = [xy((px+c*x-s*y, py+s*x+c*y))
                               for x, y in ((-sx/2, -sy/2), (sx/2, -sy/2), (sx/2, sy/2), (-sx/2, sy/2))]
                    color = tuple(max(0, min(255, round(v*255))) for v in authored["color"][:3])
                    scene.polygon(polygon, fill=color, outline="#f5f7ff" if not part["detached"] else "#ffc66b")
                scene.text((10, 10), f"position {frame['position'][0]:.2f}, {frame['position'][1]:.2f}  |  {len(frame['parts'])} parts",
                           font=small, fill="#c8d4e7")
                canvas.paste(panel, (x_origin, y_origin))
                draw.rectangle((x_origin, y_origin, x_origin + width, y_origin + height), outline="#33415b")
            draw.text((18, height + 81), "Actual pose reconstruction · same camera and simulation time · terrain/effects are not rendered",
                      font=small, fill="#adbace")
            frames.append(canvas)
        stream = io.BytesIO()
        frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:],
                       duration=spec["frame_duration_ms"], loop=0, optimize=False, disposal=2)
        data = stream.getvalue()
        name = f"clips/{pair['id']}.gif"
        _write_bytes(root / name, data)
        records.append({"file": name, "file_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                        "frames": len(frames), "simulation_ticks": pair["ticks"], "labels": pair["labels"],
                        "source_files": [{"file": pair[key], "file_sha256": hashlib.sha256((root / pair[key]).read_bytes()).hexdigest()}
                                         for key in ("left", "right")]})
    result = {"format": "nca-export-clips/v1", "complete": len(records) == len(spec["pairs"]),
              "clip_spec_file_sha256": spec_hash, "pillow": pillow_version, "clips": records,
              "camera": spec["camera"], "frame_duration_ms": spec["frame_duration_ms"],
              "evidence_class": "actual own-part pose reconstruction", "engine_rendered": False}
    immutable_json(root / "clips.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.root)
    print(f"Rendered {len(result['clips'])} matched pose clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
