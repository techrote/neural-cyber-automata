"""Small checkpoint conformance, never CUDA or learned-quality qualification."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nca.contracts import sha256_json
from nca.ik.checkpoint import (FORMAT, MAX_BYTES, capture_rng, load_checkpoint,
                               restore_checkpoint, restore_rng, write_checkpoint)
from nca.runs import immutable_bytes, immutable_json

OPTIONAL = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None
if OPTIONAL:
    import numpy
    import torch


def metadata():
    return {"config_sha256": "a" * 64, "source_sha256": "b" * 64,
            "profile_sha256": "c" * 64, "dataset_sha256": "d" * 64,
            "environment_sha256": "e" * 64}


def training_state():
    return {"boundary": "optimizer_step", "cursor": 6, "committed_steps": 2,
            "normalization": {"offset": [0., 0.], "scale": [1., 1.]},
            "split": "synthetic-conformance/v1", "generator": "cpu-sampler/v1",
            "remaining_allocation": {"steps": 4, "samples": 12, "seconds": 30.}}


class BareCheckpointTests(unittest.TestCase):
    def test_checkpoint_import_does_not_load_optional_dependencies(self):
        source = Path(__file__).parents[1] / "src"
        program = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import nca.ik.checkpoint; import nca.runs; "
            "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        )
        subprocess.run([sys.executable, "-S", "-c", program, str(source)], check=True, capture_output=True)

    def test_shared_immutable_bytes_cannot_replace_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.bin"
            self.assertEqual(immutable_bytes(path, b"first"), hashlib.sha256(b"first").hexdigest())
            with self.assertRaisesRegex(ValueError, "already exists"):
                immutable_bytes(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")
            self.assertEqual([item.name for item in Path(directory).iterdir()], ["test.bin"])

    def test_corrupt_bytes_are_rejected_before_optional_import_or_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            original = b"original"
            digest = hashlib.sha256(original).hexdigest()
            reference = {"file": path.name, "sha256": digest, "bytes": len(original), "format": FORMAT,
                         "metadata": metadata(), "training_state_sha256": sha256_json(training_state())}
            immutable_json(path.with_name(path.name + ".json"),
                           {"checkpoint": reference, "record_sha256": sha256_json(reference)})
            path.write_bytes(b"corrupt!")
            with patch("nca.ik.checkpoint._runtime", side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    load_checkpoint(path, expected_sha256=digest, expected_metadata=metadata())


@unittest.skipUnless(OPTIONAL, "optional IK Torch/NumPy checkpoint runtime is not installed")
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        random.seed(7)
        numpy.random.seed(7)
        torch.manual_seed(7)

    def components(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(),
                                    torch.nn.Dropout(.2), torch.nn.Linear(4, 2))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=.9)
        return model, optimizer, scheduler

    def step(self, model, optimizer, scheduler, generator):
        inputs = torch.randn((3, 2), generator=generator)
        targets = inputs * .2
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return inputs, float(loss.detach()), random.random(), float(numpy.random.random())

    def test_resume_matches_uninterrupted_sampler_dropout_rng_optimizer_and_scheduler(self):
        model, optimizer, scheduler = self.components()
        generator = torch.Generator(device="cpu").manual_seed(42)
        for _ in range(2):
            self.step(model, optimizer, scheduler, generator)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.pt"
            ref = write_checkpoint(path, model, optimizer, metadata=metadata(), training_state=training_state(),
                                   generators={"sampler": generator}, scheduler=scheduler)
            expected_rows = [self.step(model, optimizer, scheduler, generator) for _ in range(2)]
            expected_weights = {name: value.clone() for name, value in model.state_dict().items()}
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            resumed_model, resumed_optimizer, resumed_scheduler = self.components()
            resumed_generator = torch.Generator(device="cpu").manual_seed(99)
            payload = load_checkpoint(path, expected_sha256=ref["sha256"], expected_metadata=metadata())
            state, limits = restore_checkpoint(payload, resumed_model, resumed_optimizer,
                                              generators={"sampler": resumed_generator}, scheduler=resumed_scheduler)
            actual_rows = [self.step(resumed_model, resumed_optimizer, resumed_scheduler, resumed_generator) for _ in range(2)]
            self.assertEqual(state, training_state())
            self.assertEqual(limits, [])
            for expected, actual in zip(expected_rows, actual_rows):
                self.assertTrue(torch.equal(expected[0], actual[0]))
                self.assertEqual(expected[1:], actual[1:])
            for name, value in resumed_model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_weights[name]))
            for key, values in resumed_optimizer.state_dict()["state"].items():
                for name, value in values.items():
                    self.assertTrue(torch.equal(value, expected_optimizer["state"][key][name]))
            self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())

    def test_checkpoint_requires_committed_boundary_and_sha_identities(self):
        model, optimizer, _ = self.components()
        with tempfile.TemporaryDirectory() as directory:
            for meta, state in (({**metadata(), "source_sha256": "changed"}, training_state()),
                                (metadata(), {**training_state(), "boundary": "partial_backward"})):
                with self.subTest(meta=meta, state=state), self.assertRaises(ValueError):
                    write_checkpoint(Path(directory) / "bad.pt", model, optimizer,
                                     metadata=meta, training_state=state)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_checkpoint_published_twice_cannot_overwrite(self):
        model, optimizer, _ = self.components()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            ref = write_checkpoint(path, model, optimizer, metadata=metadata(), training_state=training_state())
            with self.assertRaisesRegex(ValueError, "already exists"):
                write_checkpoint(path, model, optimizer, metadata=metadata(), training_state=training_state())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), ref["sha256"])

    def test_missing_commit_marker_and_incompatible_identity_fail(self):
        model, optimizer, _ = self.components()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            ref = write_checkpoint(path, model, optimizer, metadata=metadata(), training_state=training_state())
            with patch("nca.ik.checkpoint._runtime", side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    load_checkpoint(path, expected_sha256=ref["sha256"],
                                    expected_metadata={**metadata(), "config_sha256": "f" * 64})
            path.with_name(path.name + ".json").unlink()
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(path, expected_sha256=ref["sha256"], expected_metadata=metadata())

    def test_byte_cap_and_nonfinite_state_fail_before_publication(self):
        model, optimizer, _ = self.components()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            with self.assertRaisesRegex(ValueError, "byte cap"):
                write_checkpoint(path, model, optimizer, metadata=metadata(), training_state=training_state(), max_bytes=32)
            with torch.no_grad():
                next(model.parameters()).fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                write_checkpoint(path, model, optimizer, metadata=metadata(), training_state=training_state())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_model_shape_and_scheduler_presence_fail(self):
        model, optimizer, scheduler = self.components()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            ref = write_checkpoint(path, model, optimizer, metadata=metadata(), training_state=training_state(), scheduler=scheduler)
            payload = load_checkpoint(path, expected_sha256=ref["sha256"], expected_metadata=metadata())
            with self.assertRaisesRegex(ValueError, "types differ"):
                restore_checkpoint(payload, model, optimizer)
            different = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.Tanh(),
                                            torch.nn.Dropout(.2), torch.nn.Linear(5, 2))
            opt = torch.optim.Adam(different.parameters(), lr=.01)
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=.9)
            with self.assertRaisesRegex(ValueError, "shape or dtype"):
                restore_checkpoint(payload, different, opt, scheduler=sched)

    def test_sampler_mismatch_and_cuda_rng_change_require_explicit_handling(self):
        generator = torch.Generator(device="cpu").manual_seed(4)
        state = capture_rng({"sampler": generator})
        with self.assertRaisesRegex(ValueError, "named RNG set"):
            restore_rng(state)
        state["torch_cuda"] = [torch.zeros(16, dtype=torch.uint8)]
        with self.assertRaisesRegex(ValueError, "explicit continuation"):
            restore_rng(state, {"sampler": generator})
        limits = restore_rng(state, {"sampler": generator}, allow_device_change=True)
        self.assertEqual(len(limits), 1)
        self.assertIn("not restored", limits[0])

    def test_invalid_optimizer_tensor_shape_fails_before_model_mutation(self):
        model, optimizer, scheduler = self.components()
        self.step(model, optimizer, scheduler, torch.Generator().manual_seed(5))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            ref = write_checkpoint(path, model, optimizer, metadata=metadata(), training_state=training_state(), scheduler=scheduler)
            payload = load_checkpoint(path, expected_sha256=ref["sha256"], expected_metadata=metadata())
            payload["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
            other, opt, sched = self.components()
            before = {name: value.clone() for name, value in other.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "optimizer tensor shape"):
                restore_checkpoint(payload, other, opt, scheduler=sched)
            self.assertTrue(all(torch.equal(value, before[name]) for name, value in other.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
