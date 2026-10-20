"""Recorded-evidence recovery and durable failure output; no host or model work."""

from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import ContractError, canonical_json, sha256_json
from nca.bench.export_postprocess import (load_declaration, recover_variant, validate_frames,
    read_exchange_prefix, forbidden_imports, _pose)
from nca.bench.subprocess_capture import run_captured

ROOT = Path(__file__).resolve().parents[1]


def fixture(model="jev", decision_ticks=12):
    profile = {"id": model, "part_count": 2, "parts": [
        {"id": name, "size": [2., 3.], "color": [.2, .4, .7, 1.]} for name in ("body", "foot")]}
    def observation(tick):
        return {"tick": tick, "run_id": "synthetic-recorded-run", "epoch": 3, "actor_id": 0,
            "observation_id": f"synthetic-recorded-run:3:0:{int(tick > 0)}:{tick}", "model": model,
            "position": [176., 346.5], "velocity": [0., 0.], "rotation": 0., "angular_velocity": 0.,
            "parts": [{"id": part["id"], "index": index, "position": [176. + index, 346.5],
                       "rotation": 0., "detached": False} for index, part in enumerate(profile["parts"])]}
    initial = observation(0)
    case = {"id": "floor-right-24", "spawn": [176., 346.5], "target": [200., 346.5]}
    traces = [{"tick_before": tick - 1, "tick_after": tick, "observation": observation(tick),
               "requested": [], "admission": []} for tick in range(1, 25)]
    def row(operation, fields, response):
        return {"request": {"operation": operation, "fields": fields, "mode": "stepped"}, "response": response}
    exchanges = {
        1: row("reset", {"actor_id": 0, "config": {"actors": [{"model": model, "position": case["spawn"], "producer": "policy"}],
                 "max_ticks": 24, "seed": 73}}, {"observation": initial}),
        2: row("profile", {}, {"profile": profile}),
        3: row("cancel", {"run_id": initial["run_id"], "epoch": initial["epoch"]}, {"observation": initial})}
    steps = []
    for index, start in enumerate(range(0, 24, decision_ticks), 4):
        exchanges[index] = row("step", {"commands": [], "ticks": decision_ticks}, {
            "actual_ticks": decision_ticks, "idle_trace": [], "idle_trace_dropped": 0,
            "unconsumed_terminal_commands": [], "traces": deepcopy(traces[start:start + decision_ticks])})
        steps.append(index)
    episode = {"case": case, "passed": True, "requested_actions": 0, "actual_ticks": 24,
               "frames": [_pose(initial)] + [_pose(row["observation"]) for row in traces], "trace": traces}
    binding = {"reset_serial": 1, "profile_serial": 2, "cancel_serial": 3, "step_serials": steps,
               "model": model, "decision_ticks": decision_ticks, "profile_sha256": sha256_json(profile)}
    return episode, binding, exchanges


class ExportPostprocessTests(unittest.TestCase):
    def test_exact_recorded_profile_is_added_without_changing_any_original_value(self):
        for model, cadence in (("jev", 12), ("astra", 6)):
            episode, binding, exchanges = fixture(model, cadence)
            original = canonical_json(episode)
            derived = recover_variant(episode, binding, exchanges)
            self.assertEqual(canonical_json(episode), original)
            self.assertEqual(set(derived) - set(episode), {"profile", "profile_sha256", "profile_provenance"})
            self.assertEqual({key: derived[key] for key in episode}, episode)
            self.assertEqual(derived["profile"], exchanges[2]["response"]["profile"])
            derived["profile"]["parts"][0]["id"] = "caller-mutated"
            self.assertEqual(exchanges[2]["response"]["profile"]["parts"][0]["id"], "body")

    def test_foreign_profile_reset_actor_replacement_or_missing_receipts_cannot_be_recovered(self):
        for mutate in (lambda e, b, x: x[1]["request"]["fields"]["config"]["actors"][0].update(model="other"),
                       lambda e, b, x: x[2]["response"]["profile"]["parts"][0].update(color=[1., 0., 0., 1.]),
                       lambda e, b, x: x[3]["request"]["fields"].update(epoch=4),
                       lambda e, b, x: x[4]["request"]["fields"]["commands"].append({"move": [1, 0]}),
                       lambda e, b, x: x[4]["response"]["traces"].pop(),
                       lambda e, b, x: e["frames"][10]["parts"][0]["position"].__setitem__(0, 99.),
                       lambda e, b, x: b.update(profile_serial=3)):
            values = fixture()
            mutate(*values)
            with self.assertRaises(ContractError):
                recover_variant(*values)

    def test_all_frame_parts_are_required_even_when_renderer_zip_would_hide_omission(self):
        episode, binding, exchanges = fixture()
        derived = recover_variant(episode, binding, exchanges)
        for mutate in (lambda e: e["frames"][12]["parts"].pop(),
                       lambda e: e["frames"][12]["parts"].reverse(),
                       lambda e: e["profile"]["parts"].pop(),
                       lambda e: e["profile"]["parts"][0].update(size=[float("nan"), 3.]),
                       lambda e: e["frames"].pop()):
            changed = deepcopy(derived)
            mutate(changed)
            with self.assertRaises(ContractError):
                validate_frames(changed, 24)

    def test_exchange_prefix_requires_ordered_confirmed_responses(self):
        requests = [{"event": "request_reserved", "serial": 1, "operation": "profile", "mode": "stepped", "fields": {}},
                    {"event": "response_received", "serial": 1, "response": {"ok": True, "profile": {}}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exchanges.jsonl.gz"
            def write(rows):
                path.write_bytes(gzip.compress("".join(canonical_json(row) + "\n" for row in rows).encode()))
            write(requests)
            self.assertEqual(read_exchange_prefix(path, 1)[1]["response"], requests[1]["response"])
            for rows in (requests[:1], requests[::-1], requests[:1] + requests):
                write(rows)
                with self.assertRaises(ContractError):
                    read_exchange_prefix(path, 1)

    def test_fixed_declaration_rejects_expanded_budget_or_changed_input_identity(self):
        path = ROOT / "configs/export/torch-cpu-clips-completion-v1.json"
        declaration = load_declaration(path)
        for mutate in (lambda d: d["budget"].update(maximum_offline_wall_seconds=90.),
                       lambda d: d["inputs"]["runtime"].update({"summary.json": "0" * 64})):
            changed = deepcopy(declaration)
            mutate(changed)
            with tempfile.TemporaryDirectory() as directory:
                altered = Path(directory) / "declaration.json"
                altered.write_text(canonical_json(changed))
                with self.assertRaisesRegex(ContractError, "postprocess_declaration"):
                    load_declaration(altered)

    def test_processor_import_does_not_import_torch_learner_or_owner_adapter(self):
        code = "import sys; import nca.bench.export_postprocess as p; assert not p.forbidden_imports(); assert 'nca.adapters.godot' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT,
                       env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1"), capture_output=True)

    def test_import_guard_covers_actual_learner_namespaces(self):
        for name in ("nca.train", "nca.train.artifacts", "nca.cellular.training", "nca.controllers.learned", "torch"):
            with patch.dict(sys.modules, {name: None}):
                self.assertIn(name, forbidden_imports())


class CapturedChildTests(unittest.TestCase):
    def run_child(self, output, code, timeout=5):
        return run_captured([sys.executable, "-u", "-c", code], output=output, label="renderer", timeout=timeout,
                            cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))

    def test_nonzero_exit_preserves_complete_stdout_stderr_and_failed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_child(output, "import sys; print('partial output'); print('real failure', file=sys.stderr); sys.exit(7)")
            self.assertEqual((output / "renderer.stdout.log").read_text(), "partial output\n")
            self.assertEqual((output / "renderer.stderr.log").read_text(), "real failure\n")
            receipt = json.loads((output / "renderer.json").read_text())
            self.assertEqual(receipt["returncode"], 7)
            self.assertFalse(receipt["completed"])

    def test_timeout_preserves_flushed_partial_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(subprocess.TimeoutExpired):
                self.run_child(output, "import sys,time; print('before timeout',file=sys.stderr); time.sleep(3)", timeout=1)
            self.assertEqual((output / "renderer.stderr.log").read_text(), "before timeout\n")
            self.assertTrue(json.loads((output / "renderer.json").read_text())["timeout"])

    def test_existing_capture_refuses_without_launching_or_rewriting_previous_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.run_child(output, "print('first invocation')")
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with patch("nca.bench.subprocess_capture.subprocess.run") as child:
                with self.assertRaises(FileExistsError):
                    self.run_child(output, "print('must not run')")
                child.assert_not_called()
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
