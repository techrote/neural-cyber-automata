"""Isolated fixed-BC evaluation worker, imported against the original learner.

The enclosing published runner supplies PYTHONPATH and a frozen manifest.
This file intentionally imports no nca.export or new learner implementation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--godot", type=Path, required=True)
    parser.add_argument("--wall-budget", type=float, required=True)
    args = parser.parse_args()
    import torch
    from nca.adapters.godot import GodotHost
    from nca.contracts import sha256_json, strict_json_loads
    from nca.controllers.learned import load_policy, policy_state_sha256
    from nca.data import read_episode
    from nca.data.capture import source_identity
    from nca.train.artifacts import immutable_json
    from nca.train.screen import ScreenBudget, ScreenBudgetExceeded, capture_task
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = strict_json_loads((args.output / "manifest.json").read_bytes())
    declaration = manifest["declaration"]
    if (manifest["sha256"] != sha256_json({k: v for k, v in manifest.items() if k != "sha256"})
            or not 0 < args.wall_budget <= 300 or declaration["budget"]["maximum_decisions"] != 320):
        raise ValueError("worker needs the exact bounded parent manifest")
    revision, sources, source_hash = source_identity(Path.cwd())
    if (revision != manifest["policy_execution_revision"]
            or source_hash != manifest["policy_execution_sources_sha256"]):
        raise ValueError("worker imported a different learner source")
    for name, expected in declaration["training_files"].items():
        if hashlib.sha256((args.training_root / name).read_bytes()).hexdigest() != expected:
            raise ValueError("worker input differs from the predeclared original artifact")
    original = strict_json_loads((args.training_root / "manifest.json").read_bytes())
    import numpy
    if (str(torch.__version__) != original["config"]["runtime"]["torch"]
            or str(numpy.__version__) != original["config"]["runtime"]["numpy"]
            or torch.get_num_threads() != 1 or torch.cuda.is_available()):
        raise ValueError("worker requires the declared CPU Torch and NumPy runtime")
    policy, _ = load_policy(args.training_root / "checkpoints/bc-seed1103-final.pt",
                            expected_config_sha256=declaration["learner_config_sha256"])
    if policy.kind != "bc" or policy_state_sha256(policy) != declaration["policy_sha256"]:
        raise ValueError("worker cannot replace the prospectively fixed candidate")
    budget = ScreenBudget(320, args.wall_budget)
    results, errors = [], []
    status, host_starts = "incomplete", 0

    def stop_at_budget(*_):
        raise ScreenBudgetExceeded("enclosing candidate-evaluation wall budget exhausted")

    signal.signal(signal.SIGINT, stop_at_budget)
    signal.signal(signal.SIGTERM, stop_at_budget)
    try:
        budget.check()
        host_starts += 1
        with GodotHost(args.host, args.godot) as host:
            for family in original["config"]["evaluation_families"]:
                _, row, _ = capture_task(host, family,
                    episode_id=f"fixed-candidate-{family['id']}-bc-seed1103-final", split="evaluation",
                    policy_id="bc-seed1103-final", kind="bc", policy=policy, seed=family["host_seed"],
                    budget=budget, output=args.output, config=original["config"], training=False)
                results.append(row)
        status = "completed"
    except BaseException as exc:
        if getattr(exc, "result", None) is not None:
            results.append(exc.result)
        status = "capped" if isinstance(exc, ScreenBudgetExceeded) else "failed"
        errors.append({"type": type(exc).__name__, "code": "candidate_evaluation_execution"})
        import traceback
        traceback.print_exc()
    episodes = [read_episode(args.output / row["artifact"]) for row in results]
    result = {"schema": "nca-fixed-candidate-worker-result/v1", "manifest_sha256": manifest["sha256"],
              "status": status, "errors": errors, "results": results, "host_start_attempts": host_starts,
              "attempted_decisions": budget.decisions, "actual_decisions": sum(e.decisions for e in episodes),
              "actual_ticks": sum(e.consumed_ticks for e in episodes),
              "worker_budget_wall_seconds": time.monotonic() - budget.started,
              "policy_sha256": policy_state_sha256(policy), "new_optimizer_updates": 0,
              "source_files_unchanged": source_identity(Path.cwd()) == (revision, sources, source_hash)}
    result["unconfirmed_decision_reservations"] = max(0, result["attempted_decisions"] - result["actual_decisions"])
    result["unconfirmed_pending_outcome"] = result["unconfirmed_decision_reservations"] > 0
    result["sha256"] = sha256_json(result)
    immutable_json(args.output / "worker-result.json", result)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
