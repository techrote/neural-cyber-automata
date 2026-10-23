"""One-shot real CPU smoke *contract* tests. No physical or neural work here."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("jolt_real_cpu_trial", ROOT / "tools/jolt_real_cpu_trial.py")
TRIAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRIAL)


class TrialGates(unittest.TestCase):
    def test_separate_scope_does_not_consume_formal_allocation(self):
        self.assertNotEqual(TRIAL.TRIAL_ID, "learn-jolt-003-20261022-a1")
        self.assertEqual(TRIAL.TEACHER_CASES, 8)
        self.assertEqual(TRIAL.PPO_DECISIONS, 256)
        self.assertEqual(TRIAL.SEED, 190103)
        text = (ROOT / "tools/jolt_real_cpu_trial.py").read_text()
        self.assertNotIn('config["cases"]["open"][0]', text)
        self.assertNotIn('config["cases"]["sealed"]', text)
        self.assertIn('"official_open_cases_opened": False', text)
        self.assertIn('"sealed_cases_opened": False', text)
        self.assertIn('"official_learn_jolt_003_allocation_consumed": False', text)

    def test_missing_or_wrong_claim_environment_refuses(self):
        env = {
            "GITHUB_REPOSITORY": TRIAL.GITHUB_REPO,
            "GITHUB_REF_NAME": TRIAL.BRANCH,
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_RUN_ATTEMPT": "2", "GITHUB_RUN_ID": "12345",
            "GITHUB_SHA": "a" * 40,
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                TRIAL._assert_environment()

    def test_prior_comment_claim_prevents_retry(self):
        with mock.patch.object(TRIAL, "_github_api",
                               return_value=[{"body": TRIAL.MARKER + "\nClaim already recorded"}]):
            self.assertTrue(TRIAL._prior_claim("fake-token"))

    def test_exhausted_claim_pagination_fails_closed(self):
        with mock.patch.object(TRIAL, "_github_api",
                               return_value=[{"body": "unrelated"}] * 100):
            with self.assertRaisesRegex(ValueError, "too long"):
                TRIAL._prior_claim("fake-token")

    def test_one_time_publication_and_readback_before_physical_work(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "claim.json"
            messages = []
            def api(url, token, *, body=None):
                if body is not None:
                    messages.append(body["body"])
                    return {"id": 314}
                self.assertTrue(url.endswith("/issues/comments/314"))
                return {"id": 314, "body": messages[0],
                        "html_url": "https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-314"}
            env = {"GH_TOKEN": "stub-token", "GITHUB_RUN_ID": "998"}
            with mock.patch.dict(os.environ, env, clear=True), \
                 mock.patch.object(TRIAL, "_assert_environment"), \
                 mock.patch.object(TRIAL, "_owner_approval", return_value={}), \
                 mock.patch.object(TRIAL, "_hash", return_value="f" * 64), \
                 mock.patch.object(TRIAL, "_git_sha", return_value="a" * 40), \
                 mock.patch.object(TRIAL, "_prior_claim", return_value=False), \
                 mock.patch.object(TRIAL, "_github_api", side_effect=api):
                self.assertEqual(TRIAL.claim(argparse.Namespace(receipt=receipt)), 0)
            data = json.loads(receipt.read_bytes())
            self.assertEqual(data["issue_comment_id"], 314)
            self.assertFalse(data["physical_work_started_at_claim"])
            self.assertFalse(data["official_learn_jolt_003_allocation_consumed"])
            self.assertIn(TRIAL.MARKER, messages[0])

    def test_existing_claim_refuses_before_new_api_publication(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.dict(os.environ, {"GH_TOKEN": "stub-token"}, clear=True), \
             mock.patch.object(TRIAL, "_assert_environment"), \
             mock.patch.object(TRIAL, "_owner_approval", return_value={}), \
             mock.patch.object(TRIAL, "_prior_claim", return_value=True), \
             mock.patch.object(TRIAL, "_github_api") as api:
            with self.assertRaisesRegex(ValueError, "already been claimed"):
                TRIAL.claim(argparse.Namespace(receipt=Path(directory) / "claim.json"))
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
