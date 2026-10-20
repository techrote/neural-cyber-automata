"""NumPy-only external controller timing on previously captured local sequences."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from nca.contracts import ContractError
from nca.controllers.baseline import SkillGoal
from nca.export import RuntimeBundle, ExportController, context_for
from nca.export.build import read_json, immutable_json
from .common import durations, process_memory, numpy_build


def measure(bundle_path: Path, bundle_hash: str, inputs_path: Path, count: int):
    if not 1 <= count <= 1000 or "torch" in sys.modules:
        raise ContractError("worker_scope", "fresh NumPy-only process and at most1000 inferences required")
    payload, _ = read_json(bundle_path)
    sequences, inputs_hash = read_json(inputs_path)
    context = context_for(payload["profile"], payload["context"])
    bundle = RuntimeBundle.load(bundle_path, context, expected_sha256=bundle_hash)
    times, resets, outputs = [], [], []
    index = 0
    while len(times) < count:
        sequence = sequences["sequences"][index % len(sequences["sequences"])]
        reset_started = time.perf_counter_ns()
        controller = ExportController(bundle)
        goal = SkillGoal(**sequence["goal"])
        resets.append(time.perf_counter_ns() - reset_started)
        for view in sequence["observations"]:
            start = time.perf_counter_ns()
            result = controller.step(view, goal)
            times.append(time.perf_counter_ns() - start)
            if result.status != "accepted":
                raise ContractError("recorded_sequence", result.reason)
            outputs.append(result.action.to_dict())
            if len(times) == count:
                break
        index += 1
    from nca.contracts import sha256_json
    return {"format": "nca-export-inference-cost/v1", "runtime_context": context.to_dict(),
            "bundle_file_sha256": bundle_hash, "inputs_file_sha256": inputs_hash,
            "scope": "full local-view validation, preprocessing, recurrence, action mapping and source stamp",
            "transport_included": False, "weight_load_included": False,
            "torch_imported": "torch" in sys.modules, "inference": durations(times),
            "numpy_build": numpy_build(),
            "episode_initialization": durations(resets), "outputs_sha256": sha256_json(outputs),
            "numeric_weight_bytes": bundle.network.nbytes, "hidden_state_bytes": controller.hidden.nbytes,
            "process_memory": process_memory(os.getpid()), "memory_method": "Linux proc status VmRSS/VmHWM"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.bundle, args.sha256, args.inputs, args.count)
    immutable_json(args.output, result)
    print('{"status":"completed","scope":"numpy_only_inference"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
