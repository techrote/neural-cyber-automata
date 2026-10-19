#!/usr/bin/env python3
"""Prepare or execute only the fixed commissioned Jolt direction-frame follow-up."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "inspect"))
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/first-skill/jolt-direction-frame-v1.json"))
    parser.add_argument("--host", type=Path)
    parser.add_argument("--godot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    from nca.contracts import canonical_json
    from nca.train.goal_frame_config import FOLLOWUP_SHA256, read_followup_config, verify_original_boundary
    config = read_followup_config(args.config)
    verify_original_boundary(config, args.source)
    if args.operation == "prepare":
        print(canonical_json({"config_sha256": FOLLOWUP_SHA256, "training_seeds": config["training_seeds"],
                              "budget": config["budget"], "prerequisite": config["prerequisite"],
                              "host_starts": 0, "policy_calls": 0, "optimizer_updates": 0}))
        return 0
    if args.operation == "inspect":
        if args.checkpoint is None:
            parser.error("inspect requires --checkpoint")
        from nca.controllers.goal_frame import load_goal_frame_policy
        from nca.controllers.learned import policy_state_sha256
        policy, metadata = load_goal_frame_policy(args.checkpoint, expected_config_sha256=FOLLOWUP_SHA256)
        print(canonical_json({"specification": policy.specification(), "metadata": metadata,
                              "policy_sha256": policy_state_sha256(policy)}))
        return 0
    if args.host is None or args.godot is None or args.output is None:
        parser.error("run requires --host, --godot and a new immutable --output directory")
    from nca.train.goal_frame import run_followup
    summary = run_followup(root=args.source, checkout=args.host, godot=args.godot,
                           config_path=args.config, output=args.output)
    print(canonical_json({"status": summary["status"], "sha256": summary["sha256"],
                          "attempted_game_requests": summary["attempted_game_requests"],
                          "wall_seconds": summary["wall_seconds"], "error": summary["error"],
                          "evaluation": summary["evaluation"]}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
