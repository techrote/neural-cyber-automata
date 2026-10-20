"""Standalone Torch deployment call cost; never imports a learner or opens a host."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import torch

from nca.contracts import ContractError, sha256_json
from nca.controllers.baseline import SkillGoal
from nca.export import ExportController, context_for
from nca.export.build import immutable_json, read_json
from nca.export.torch_runtime import TorchRuntimeBundle, BACKEND
from .common import durations, process_memory


def measure(bundle_path: Path, bundle_hash: str, inputs_path: Path, count: int):
    if type(count) is not int or not 1 <= count <= 1000:
        raise ContractError("worker_scope", "at most 1,000 explicit deployment calls required")
    torch.set_num_threads(1)
    payload, _ = read_json(bundle_path)
    inputs, inputs_hash = read_json(inputs_path)
    sequences = inputs.get("sequences")
    if (not isinstance(sequences, list) or not sequences
            or any(not isinstance(row, dict) or not row.get("observations") for row in sequences)):
        raise ContractError("recorded_sequence", "nonempty complete actual-observation sequences required")
    numeric = payload["numeric_bundle"]
    context = context_for(numeric["profile"], numeric["context"])
    bundle = TorchRuntimeBundle.load(bundle_path, context, expected_sha256=bundle_hash)
    times, resets, outputs, index = [], [], [], 0
    while len(times) < count:
        sequence = sequences[index % len(sequences)]
        start = time.perf_counter_ns()
        controller = ExportController(bundle)
        goal = SkillGoal(**sequence["goal"])
        resets.append(time.perf_counter_ns() - start)
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
    learner_imported = any(name == "nca.controllers.learned" or name.startswith("nca.train") for name in sys.modules)
    if learner_imported:
        raise ContractError("worker_scope", "standalone deployment imported learner code")
    return {"format": "nca-torch-deployment-inference-cost/v1", "backend": BACKEND,
            "runtime_context": context.to_dict(), "bundle_file_sha256": bundle_hash,
            "inputs_file_sha256": inputs_hash, "learner_imported": False, "host_starts": 0,
            "training_updates": 0, "scope": "complete validation, preprocessing, recurrence, action and source stamp",
            "weight_load_included": False, "transport_included": False,
            "inference": durations(times), "episode_initialization": durations(resets),
            "outputs_sha256": sha256_json(outputs), "numeric_weight_bytes": bundle.network.nbytes,
            "hidden_state_bytes": controller.hidden.nbytes, "process_memory": process_memory(os.getpid())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    immutable_json(args.output, measure(args.bundle, args.sha256, args.inputs, args.count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
