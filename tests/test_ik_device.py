"""Bare import/refusal checks and tiny optional CPU placement conformance.

Mocked CUDA capabilities below only test refusal logic and metadata.  This file
never qualifies CUDA hardware or launches a CUDA training/benchmark fixture.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nca.ik.device import (DeviceContext, DeviceError, configure_device, device_report,
                           environment_identity, parse_device, _architecture_route)


def fake_torch(*, available=False, capability=(8, 6), architectures=("sm_86",),
               cuda_version="12.6", version="2.8.0+cu126"):
    """Metadata-only test double, intentionally has no tensor/operator APIs."""
    return SimpleNamespace(
        __version__=version,
        __config__=SimpleNamespace(show=lambda: "synthetic test build"),
        version=SimpleNamespace(cuda=cuda_version, git_version="fixture"),
        get_float32_matmul_precision=lambda: "highest",
        get_num_threads=lambda: 1,
        are_deterministic_algorithms_enabled=lambda: True,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False, benchmark=False, deterministic=True,
                                  version=lambda: 91000),
        ),
        cuda=SimpleNamespace(
            get_arch_list=lambda: list(architectures),
            is_available=lambda: available,
            is_initialized=lambda: False,
            device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(
                name="Synthetic metadata fixture", major=capability[0], minor=capability[1],
                multi_processor_count=10, total_memory=2048),
            mem_get_info=lambda index: (1024, 2048),
        ),
    )


class IKDeviceBareTests(unittest.TestCase):
    def test_module_and_nominal_contracts_do_not_import_optional_runtime(self):
        program = """
import importlib.abc
import sys
class DenyOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch'):
            raise AssertionError('optional import on bare path')
sys.meta_path.insert(0, DenyOptional())
import nca.ik
import nca.ik.device
assert nca.ik.runner_right_sole_profile().effector.id == 'right_sole'
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run([sys.executable, "-c", program], capture_output=True,
                                text=True, env=env, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selection_is_explicit_and_canonical(self):
        self.assertEqual(parse_device("cpu"), ("cpu", None))
        self.assertEqual(parse_device("cuda:0"), ("cuda", 0))
        self.assertEqual(parse_device("cuda:12"), ("cuda", 12))
        for value in (None, "cuda", "cuda:-1", "CUDA:0", "cuda:01", "cpu:0", "mps", "auto", True):
            with self.subTest(value=value), self.assertRaises(DeviceError):
                parse_device(value)

    def test_runtime_absence_is_not_cpu_fallback_or_qualification(self):
        with patch("nca.ik.device.importlib.import_module", side_effect=ImportError):
            for requested in ("cpu", "cuda:0"):
                report = device_report(requested)
                self.assertEqual(report["requested_device"], requested)
                self.assertEqual(report["status"], "unavailable")
                self.assertEqual(report["reason"], "optional_runtime_missing")
                self.assertEqual(report["qualification"], "unmeasured")
                with self.assertRaises(DeviceError) as caught:
                    configure_device(requested, 41)
                self.assertEqual(caught.exception.report["requested_device"], requested)

    def test_configuration_identity_and_budget_inputs_refuse_boolean_or_nonfinite(self):
        for seed in (-1, 2**32, 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(DeviceError):
                configure_device("cpu", seed)
        for fraction in (0, -1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(fraction=fraction), self.assertRaises(DeviceError):
                configure_device("cpu", 41, memory_fraction=fraction)
        for value in ("wrong", "A" * 64, True):
            with self.subTest(value=value), self.assertRaises(DeviceError):
                device_report("cpu", config_sha256=value)

    def test_version_drift_is_explicit_unsupported_environment(self):
        for torch in (fake_torch(version="2.9.0+cu126"), fake_torch(cuda_version="12.8")):
            with patch("nca.ik.device.importlib.import_module",
                       side_effect=[torch, SimpleNamespace(__version__="2.3.5")]):
                report = device_report("cpu")
            self.assertEqual(report["status"], "unsupported")
            self.assertEqual(report["reason"], "runtime_version_mismatch")
        with patch("nca.ik.device.importlib.import_module",
                   side_effect=[fake_torch(), SimpleNamespace(__version__="2.4.0")]):
            self.assertEqual(device_report("cpu")["reason"], "runtime_version_mismatch")

    def _synthetic_report(self, spec="cuda:0", **options):
        with patch("nca.ik.device._driver_version", return_value="580.00"), \
                patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
                patch("nca.ik.device.importlib.import_module",
                      side_effect=[fake_torch(**options), SimpleNamespace(__version__="2.3.5")]):
            return device_report(spec, source_sha256="a" * 64, config_sha256="b" * 64)

    def test_unavailable_cuda_and_missing_index_remain_unqualified(self):
        self.assertEqual(self._synthetic_report()["reason"], "cuda_unavailable")
        self.assertEqual(self._synthetic_report("cuda:1", available=True)["reason"],
                         "cuda_index_unavailable")

    def test_compiled_architectures_are_checked_but_do_not_qualify_operations(self):
        unsupported = self._synthetic_report(available=True, capability=(3, 7))
        self.assertEqual(unsupported["reason"], "cuda_architecture_not_compiled")
        report = self._synthetic_report(available=True)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["qualification"], "unmeasured")
        self.assertEqual(report["gpu"]["capability"], [8, 6])
        self.assertEqual(report["compiled_architectures"], ["sm_86"])
        self.assertEqual(report["source_sha256"], "a" * 64)
        self.assertEqual(report["config_sha256"], "b" * 64)
        self.assertEqual(report["memory"], {"free_bytes": 1024, "total_bytes": 2048})
        self.assertEqual(report["driver"], "580.00")
        self.assertEqual(_architecture_route((8, 9), ["sm_86"]), "sm_86")
        self.assertEqual(_architecture_route((9, 0), ["compute_80"]), "compute_80")
        self.assertIsNone(_architecture_route((9, 0), ["sm_86"]))

    def test_invalid_determinism_environment_is_refused_before_runtime_import(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "invalid"}), \
                patch("nca.ik.device.importlib.import_module") as load:
            report = device_report("cuda:0")
        load.assert_not_called()
        self.assertEqual(report["reason"], "invalid_determinism_configuration")

    def test_environment_identity_excludes_transient_memory_and_source_bindings(self):
        report = self._synthetic_report(available=True)
        changed = dict(report, memory={"free_bytes": 512, "total_bytes": 2048},
                       source_sha256="c" * 64, config_sha256="d" * 64)
        self.assertEqual(environment_identity(report), environment_identity(changed))
        changed["driver"] = "581.00"
        self.assertNotEqual(environment_identity(report), environment_identity(changed))


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "optional pinned Torch runtime is not installed")
class IKDeviceCPUConformanceTests(unittest.TestCase):
    def setUp(self):
        self.context = configure_device("cpu", 41)
        self.torch = self.context.torch

    def test_actual_cpu_forward_backward_optimizer_and_all_tensor_placement(self):
        torch, context = self.torch, self.context
        model = torch.nn.Linear(2, 2, dtype=torch.float32, device=context.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, foreach=False)
        x = torch.tensor([[.1, .2], [.3, -.2]], dtype=torch.float32, device=context.device)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        prediction = model(x)
        loss = prediction.square().mean()
        loss.backward()
        optimizer.step()
        receipt = context.verify_tensors({
            "data": x, "model": model.state_dict(), "prediction": prediction,
            "loss": loss, "gradients": [p.grad for p in model.parameters()],
            "optimizer": optimizer.state_dict()["state"],
        })
        self.assertEqual(receipt["device"], "cpu")
        self.assertTrue(receipt["all_on_selected_device"])
        self.assertEqual(set(receipt["tensor_counts"]),
                         {"data", "model", "prediction", "loss", "gradients", "optimizer"})
        self.assertTrue(any(not torch.equal(before[name], value)
                            for name, value in model.state_dict().items()))
        precision = context.report["precision"]
        self.assertFalse(precision["matmul_allow_tf32"])
        self.assertFalse(precision["cudnn_allow_tf32"])
        self.assertTrue(precision["deterministic_algorithms"])
        self.assertEqual(context.report["qualification"], "unmeasured")

    def test_wrong_precision_nonfinite_and_absent_evidence_are_refused(self):
        for data, code in (
                ({"loss": self.torch.tensor([1.], dtype=self.torch.float64)}, "dtype_mismatch"),
                ({"loss": self.torch.tensor([float("nan")])}, "nonfinite_tensor"),
                ({"gradients": [None]}, "missing_tensor_evidence"),
                ({}, "missing_tensor_evidence")):
            with self.subTest(code=code), self.assertRaises(DeviceError) as caught:
                self.context.verify_tensors(data)
            self.assertEqual(caught.exception.code, code)
        other = DeviceContext(self.torch, self.context.numpy, self.torch.device("cuda:0"), {})
        with self.assertRaises(DeviceError) as caught:
            other.verify_tensors({"data": self.torch.ones(2)})
        self.assertEqual(caught.exception.code, "device_placement_mismatch")

    def test_cpu_timing_and_memory_leave_cuda_measurements_null(self):
        value, timing = self.context.timed(lambda: self.torch.ones(2).sum())
        self.assertEqual(float(value), 2.)
        self.assertGreaterEqual(timing["synchronized_host_seconds"], 0)
        self.assertIsNone(timing["cuda_event_seconds"])
        self.context.reset_peak_memory()
        self.assertIsNone(self.context.memory_stats()["peak_allocated_bytes"])
        self.assertIsNone(self.context.memory_stats()["peak_reserved_bytes"])


if __name__ == "__main__":
    unittest.main()
