#!/usr/bin/env python3
"""Explicit bounded CPU screen; no game/training run occurs on import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run", "inspect"))
    parser.add_argument("--config", type=Path, default=Path("configs/first-skill/jolt-screen-v1.json"))
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--godot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--accepted-t102")
    parser.add_argument("--accepted-t103")
    parser.add_argument("--accepted-t104")
    args = parser.parse_args()
    from nca.train.artifacts import read_config
    config, digest = read_config(args.config)
    if args.operation == "prepare":
        print(json.dumps({"config_sha256": digest, "training_seeds": config["training_seeds"],
                          "budget": config["budget"], "host_started": False, "weights_trained": False}, indent=2))
        return 0
    if args.operation == "inspect":
        if args.checkpoint is None:
            parser.error("inspect requires --checkpoint")
        from nca.controllers.learned import load_policy, policy_state_sha256
        policy, metadata = load_policy(args.checkpoint, expected_config_sha256=digest)
        print(json.dumps({"specification": policy.specification(), "metadata": metadata,
                          "policy_sha256": policy_state_sha256(policy)}, indent=2))
        return 0
    if any(getattr(args, key) is None for key in ("checkout", "godot", "output", "accepted_t102", "accepted_t103", "accepted_t104")):
        parser.error("run requires checkout, godot, a new output directory and all three accepted dependency commits")
    from nca.train.screen import run_screen
    result = run_screen(checkout=args.checkout, godot=args.godot, output=args.output, config=config,
                        dependency_commits={"t102": args.accepted_t102, "t103": args.accepted_t103, "t104": args.accepted_t104})
    print(json.dumps({"status": result["status"], "error": result["error"],
                      "decisions": result["actual_decisions"], "wall_seconds": result["wall_seconds"],
                      "summary_sha256": result["sha256"], "evaluation": result["evaluation"]}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
