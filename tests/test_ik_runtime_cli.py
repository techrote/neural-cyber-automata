"""Real CLI/subprocess conformance in an ephemeral, committed source fixture.

The synthetic Git identity and three tiny CPU steps test software plumbing.
They do not qualify hardware, measure performance, or publish a formal run.
"""

import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

import nca


def pinned_runtime_available():
    try:
        return (metadata.version("torch").split("+", 1)[0] == "2.8.0"
                and metadata.version("numpy") == "2.3.5")
    except metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(pinned_runtime_available() and shutil.which("git"),
                     "optional pinned IK Torch/NumPy runtime and Git are required")
class IKRuntimeCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.allocation = self.root / "allocation"
        repository = Path(__file__).resolve().parents[1]
        package = Path(nca.__file__).resolve().parent
        files = sorted(set(package.rglob("*.py")) | set(package.rglob("*.gd"))
                       | set(package.rglob("*.json")))
        for original in files:
            destination = self.source / "src/nca" / original.relative_to(package)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
        for relative in ("pyproject.toml", "tools/capture_replay.py", "tools/setup_ik.sh",
                         "tools/setup_ik.ps1", "configs/ik/runtime-smoke-v1.json"):
            destination = self.source / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / relative, destination)
        self.environment = dict(os.environ, NCA_SOURCE_ROOT=str(self.source),
                                PYTHONPATH=str(self.source / "src"),
                                PYTHONDONTWRITEBYTECODE="1", GIT_CONFIG_NOSYSTEM="1",
                                GIT_CONFIG_GLOBAL=os.devnull)
        for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
            self.environment.pop(name, None)
        for arguments in (("init", "--quiet"), ("config", "core.autocrlf", "false"),
                          ("add", "--force", "."),
                          ("-c", "user.name=IK conformance fixture", "-c",
                           "user.email=conformance@example.invalid", "-c", "commit.gpgsign=false",
                           "commit", "--quiet", "-m", "Synthetic CLI software conformance fixture")):
            subprocess.run(["git", "-C", str(self.source), *arguments], check=True,
                           capture_output=True, text=True, env=self.environment, timeout=10)
        config = json.loads((self.source / "configs/ik/runtime-smoke-v1.json").read_text())
        config.update(steps=3, batch_size=3, max_optimizer_steps=3,
                      max_samples=9, max_wall_seconds=45)
        self.config = self.root / "fixture-config.json"
        self.config.write_text(json.dumps(config), encoding="utf-8")

    def stop_process(self, process):
        """Allow supervisor cleanup, then kill its descendants on a stuck test."""
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=5, check=False)
        else:
            # The worker owns another session, so killing only our CLI process
            # group would leave it running if the supervisor were stuck.
            table = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True, timeout=5)
            parents = {int(pid): int(parent) for pid, parent in
                       (line.split() for line in table.splitlines())}
            descendants = {process.pid}
            while True:
                expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
                if expanded == descendants:
                    break
                descendants = expanded
            process.send_signal(signal.SIGINT)
            try:
                process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            finally:
                for pid in descendants:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)

    def cli(self, *arguments, expected_code=0):
        process = subprocess.Popen([sys.executable, "-m", "nca.ik.runtime", *arguments],
                                   cwd=self.root, env=self.environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=os.name == "posix")
        try:
            stdout, stderr = process.communicate(timeout=50)
        finally:
            self.stop_process(process)
        self.assertEqual(process.returncode, expected_code, (arguments, stdout, stderr))
        return json.loads(stdout)

    def read(self, relative):
        return json.loads((self.allocation / relative).read_text(encoding="utf-8"))

    def test_real_cli_resumes_committed_boundary_and_retains_the_allocation(self):
        prepared = self.cli("prepare", "--config", str(self.config), "--device", "cpu",
                            "--output", str(self.allocation))
        digest = hashlib.sha256((self.allocation / "declaration.json").read_bytes()).hexdigest()
        self.assertEqual(prepared["declaration_sha256"], digest)
        self.assertEqual(prepared["declaration"]["device"], "cpu")
        allocation = ("--run", str(self.allocation), "--declaration-sha256", digest)
        first = self.cli("smoke", *allocation, "--stop-after", "1")
        first_record = (self.allocation / "records/attempt-0001.json").read_bytes()
        self.assertEqual(first["attempt_count"], 1)
        self.assertEqual(first["committed_steps"], 1)
        self.assertEqual(first["charged_steps"], 1)
        self.assertEqual(first["charged_samples"], 3)
        self.assertEqual(first["cursor"], 3)
        self.assertEqual(first["qualification"], "unmeasured")

        resumed = self.cli("resume", *allocation)
        state = self.cli("inspect", "--run", str(self.allocation))
        for key in ("attempt_count", "committed_steps", "charged_steps", "charged_samples",
                    "charged_seconds", "cursor", "qualification"):
            self.assertEqual(state[key], resumed[key])
        self.assertEqual(state["attempt_count"], 2)
        self.assertEqual(state["committed_steps"], 3)
        self.assertEqual(state["charged_steps"], 3)
        self.assertEqual(state["charged_samples"], 9)
        self.assertEqual(state["cursor"], 9)
        self.assertEqual(state["remaining_steps"], 0)
        self.assertEqual(state["device"], "cpu")
        self.assertEqual(state["qualification"], "cpu_smoke_passed")
        self.assertFalse(state["active"])
        self.assertFalse(state["unresolved_attempt"])
        self.assertEqual(state["lineage_limits"], [])
        self.assertEqual([record["status"] for record in state["attempts"]], ["completed", "completed"])
        self.assertEqual((self.allocation / "records/attempt-0001.json").read_bytes(), first_record)
        self.assertGreater(state["charged_seconds"], first["charged_seconds"])
        self.assertLessEqual(state["charged_seconds"], 45)
        self.assertEqual(state["charged_seconds"], sum(record["runtime_seconds"] for record in state["attempts"]))
        request = self.read("attempts/attempt-0002/request.json")
        self.assertEqual(request["prior_charged_seconds"], first["charged_seconds"])
        self.assertEqual(request["prior_charged_steps"], 1)
        self.assertEqual(request["prior_committed_steps"], 1)
        self.assertIsNotNone(request["checkpoint"])
        receipts = [json.loads(path.read_text()) for path in sorted(self.allocation.glob("attempts/*/step-*.json"))]
        self.assertEqual([receipt["committed_steps"] for receipt in receipts], [1, 2, 3])
        self.assertEqual([receipt["cursor"] for receipt in receipts], [3, 6, 9])
        self.assertEqual(len(list(self.allocation.glob("attempts/*/reserve-*.json"))), 3)
        sample_ids = [sample for receipt in receipts for sample in receipt["sample_ids"]]
        self.assertEqual(len(sample_ids), 9)
        self.assertEqual(len(set(sample_ids)), 9)
        for receipt in receipts:
            self.assertEqual(receipt["result"]["placement"]["device"], "cpu")
            self.assertTrue(receipt["result"]["placement"]["all_on_selected_device"])
            self.assertTrue(receipt["result"]["weights_changed"])

        records = {path.name: path.read_bytes() for path in (self.allocation / "records").iterdir()}
        refused = self.cli("resume", *allocation, expected_code=1)
        self.assertEqual(refused["admission_refused"], "completed_or_exhausted_allocation")
        self.assertEqual(refused["attempt_count"], 2)
        self.assertEqual(refused["charged_seconds"], state["charged_seconds"])
        self.assertEqual(refused["charged_steps"], 3)
        self.assertEqual(refused["charged_samples"], 9)
        self.assertEqual(len(list((self.allocation / "attempts").iterdir())), 2)
        self.assertEqual({path.name: path.read_bytes() for path in (self.allocation / "records").iterdir()}, records)


if __name__ == "__main__":
    unittest.main()
