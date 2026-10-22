"""Non-physical checks for the opt-in Jolt v3 GitHub long-run gate.

These tests use temporary files and mocked GitHub API responses. No model,
Godot host, resource admission or experiment is started.
© 2026 techrote. All rights reserved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "jolt_ci_workflow_v3", ROOT / "tools/jolt_ci_workflow_v3.py"
)
CI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CI)


class ArtifactEvidenceTests(unittest.TestCase):
    def test_exclusive_publication_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            CI._publish(path, {"accepted": False})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                CI._publish(path, {"accepted": True})
            self.assertEqual(original, path.read_bytes())

    def test_inventory_hashes_available_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "evidence"
            root.mkdir()
            payload = root / "original.bin"
            payload.write_bytes(b"negative result preserved")
            receipt = Path(folder) / "manifest.json"
            code = CI.inventory(argparse.Namespace(root=root, receipt=receipt))
            self.assertEqual(code, 0)
            result = json.loads(receipt.read_bytes())
            self.assertEqual(result["files"]["original.bin"]["sha256"],
                             hashlib.sha256(payload.read_bytes()).hexdigest())
            self.assertTrue(result["complete_readback"])
            self.assertEqual(result["file_count"], 1)

    @unittest.skipIf(os.name == "nt", "Symlink permissions vary on Windows CI")
    def test_inventory_marks_symlink_incomplete_and_does_not_follow_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "evidence"
            root.mkdir()
            (root / "original.bin").write_bytes(b"retained")
            (root / "substitute.bin").symlink_to(root / "original.bin")
            receipt = Path(folder) / "manifest.json"
            self.assertEqual(CI.inventory(argparse.Namespace(root=root, receipt=receipt)), 2)
            result = json.loads(receipt.read_bytes())
            self.assertFalse(result["complete_readback"])
            self.assertNotIn("substitute.bin", result["files"])
            self.assertEqual(result["errors"][0]["entry"], "substitute.bin")


class SingleAttemptClaimTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.declaration = root / "declaration.json"
        self.attestation = root / "attestation.json"
        self.receipt = root / "claim.json"
        self.source = "a" * 40
        self.declaration.write_text(json.dumps({
            "allocation_id": CI.ALLOCATION,
            "source_revision": self.source,
            "runtime": {
                "torch":"2.8.0+cpu", "numpy":"2.3.5", "device":"cpu",
                "threads":1, "interop_threads":1, "python":"3.12.10",
                "system":"Linux", "kernel":"6.17", "machine":"x86_64",
                "libc":["glibc","2.39"], "cpu_model":"Synthetic CPU",
                "logical_cpus_available":1, "deterministic_algorithms":True
            },
            "independent_reviews": [{}, {}],
        }))
        self.attestation.write_text(json.dumps({
            "schema": "nca-jolt-003-ci-frozen-declaration-readback/v1",
            "full_source_runtime_resource_and_reviews_admitted": True,
            "execution_started": False,
            "source_revision": self.source,
            "declaration_sha256": CI._hash(self.declaration),
        }))
        self.args = argparse.Namespace(declaration=self.declaration,
                                       attestation=self.attestation,
                                       receipt=self.receipt)
        self.env = {
            "GITHUB_REPOSITORY": CI.REPOSITORY,
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF_NAME": "main",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "123456",
            "GH_TOKEN": "fake-test-token",
        }

    def tearDown(self):
        self.folder.cleanup()

    def test_rerun_denied_before_any_api_write(self):
        with mock.patch.dict(os.environ, {**self.env, "GITHUB_RUN_ATTEMPT": "2"}, clear=True), \
                mock.patch.object(CI, "_github_api") as api:
            with self.assertRaises(ValueError):
                CI.claim(self.args)
            api.assert_not_called()
        self.assertFalse(self.receipt.exists())

    def test_prior_persistent_claim_blocks_second_invocation(self):
        with mock.patch.dict(os.environ, self.env, clear=True), \
                mock.patch.object(CI, "_prior_claim", return_value=True), \
                mock.patch.object(CI, "_github_api") as api:
            with self.assertRaisesRegex(ValueError, "already been claimed"):
                CI.claim(self.args)
            api.assert_not_called()
        self.assertFalse(self.receipt.exists())

    def test_claim_requires_same_original_declaration_hash(self):
        self.attestation.write_text(json.dumps({
            **json.loads(self.attestation.read_text()),
            "declaration_sha256": "b" * 64,
        }))
        with mock.patch.dict(os.environ, self.env, clear=True), \
                mock.patch.object(CI, "_github_api") as api:
            with self.assertRaises(ValueError):
                CI.claim(self.args)
            api.assert_not_called()

    def test_claim_publishes_then_reads_back_exact_original_comment(self):
        posted_body = []

        def fake_api(url, token, *, body=None):
            self.assertEqual(token, "fake-test-token")
            if body is not None:
                posted_body.append(body["body"])
                return {"id": 123}
            if url.endswith("/issues/comments/123"):
                return {"id": 123, "body": posted_body[0],
                        "html_url": "https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-123"}
            raise AssertionError("unexpected API request")

        with mock.patch.dict(os.environ, self.env, clear=True), \
                mock.patch.object(CI, "_prior_claim", return_value=False), \
                mock.patch.object(CI, "_github_api", side_effect=fake_api) as api:
            self.assertEqual(CI.claim(self.args), 0)
            self.assertEqual(api.call_count, 2)
        record = json.loads(self.receipt.read_text())
        self.assertEqual(record["issue_comment_id"], 123)
        self.assertTrue(record["original_comment_readback_equal"])
        self.assertFalse(record["physical_execution_started_at_claim"])
        self.assertIn(CI.CLAIM_MARKER, posted_body[0])

    def test_incomplete_issue_comment_pagination_fails_closed(self):
        with mock.patch.object(CI, "_github_api", return_value=[{"body": "not a claim"}] * 100):
            with self.assertRaisesRegex(ValueError, "exceeds complete readback bound"):
                CI._prior_claim("fake-test-token")

    def test_existing_comment_marker_is_detected(self):
        with mock.patch.object(CI, "_github_api", return_value=[{"body": CI.CLAIM_MARKER}]):
            self.assertTrue(CI._prior_claim("fake-test-token"))


if __name__ == "__main__":
    unittest.main()
