"""Persistent external client for the accepted Cybersand Godot owners.

The launcher is owned here and passed as an external --script. Cybersand's
public configure/step/observation methods retain all physics/material authority.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from .protocol import (
    DEFAULT_DECISION_TICKS, DEFAULT_LEASE_TICKS, MAX_REPEAT_TICKS, PROTOCOL,
    EpisodeFinished, HostError, HostIntegrityError, JsonChannel, ProtocolError,
    RewardDifference, integer, make_physical_command, make_sampled_command, owned_json,
)

PINNED_HOST_REVISION = "f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497"
PINNED_HOST_TREE = "445a185e9f9824b16953fec27bf0157dfeedca77"
PINNED_GODOT_VERSION = "4.7.stable.official.5b4e0cb0f"
PINNED_GODOT_LINUX = "f85bbc6b15e22416c7d797cd60b63286dd67b9cb13498847056c18520ae55a75"
PINNED_RAPIER_LINUX = "147ab5673b9687ccb7a3b12fec07ea55cc674ad8a75f529d1085bf7d61fbd831"
PHYSICAL_SCHEDULE = "synchronous-mask-actuation-effects-material-feedback-terrain-rapier/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_host(checkout: str | Path, godot: str | Path) -> dict[str, Any]:
    """Reject source/runtime drift before executing host code; return portable evidence."""
    root = Path(checkout).resolve()
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    try:
        revision, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
        if revision != PINNED_HOST_REVISION or tree != PINNED_HOST_TREE:
            raise HostError("host revision is not the audited accepted source")
        if git("diff", "--name-only", "HEAD"):
            raise HostError("tracked host source/runtime differs from the accepted revision")
        lock = json.loads((root / "third_party/godot-runtime.lock.json").read_text())
        manifest_path = "godot/addons/cybersand_native/runtime-provenance.linux.json"
        native = json.loads((root / manifest_path).read_text())
        if os.name == "nt":
            # Windows source support is prepared; validation is a distinct lane.
            manifest_path = "godot/addons/cybersand_native/runtime-provenance.json"
            native = json.loads((root / manifest_path).read_text())
        native_digest = sha256_file(root / native["artifact"])
        if native_digest != native["sha256"]:
            raise HostError("native artifact does not match its accepted provenance")
        source_inputs: dict[str, str] = {}
        for name, expected in native["source_inputs"].items():
            # Provenance defines canonical LF Git text, including on Windows.
            actual = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            if actual != expected:
                raise HostError(f"native source provenance mismatch: {name}")
            source_inputs[name] = actual
        rapier_name = ("godot/addons/godot-rapier2d/bin/libgodot_rapier.windows.x86_64-pc-windows-msvc.dll"
                       if os.name == "nt" else
                       "godot/addons/godot-rapier2d/bin/libgodot_rapier.linux.x86_64-unknown-linux-gnu.so")
        rapier_digest = sha256_file(root / rapier_name)
        if os.name != "nt" and rapier_digest != PINNED_RAPIER_LINUX:
            raise HostError("Rapier artifact does not match its accepted identity")
        # The committed LFS pointer supplies the expected digest on both platforms.
        pointer = git("show", f"HEAD:{rapier_name}")
        if f"oid sha256:{rapier_digest}" not in pointer:
            raise HostError("Rapier bytes differ from the accepted Git LFS pointer")
        executable = str(Path(shutil.which(str(godot)) or str(godot)).resolve())
        version = subprocess.check_output([executable, "--version"], text=True, timeout=15).strip()
        if version != lock["godot_version"] or version != PINNED_GODOT_VERSION:
            raise HostError("Godot version differs from the host lock")
        godot_digest = sha256_file(Path(executable))
        if os.name != "nt" and godot_digest != PINNED_GODOT_LINUX:
            raise HostError("Godot executable differs from the verified locked archive")
        scripts = {}
        for name in ("physical_actor_arena", "physical_actor_session", "physical_actor_rig",
                     "physical_actor_catalog", "physical_actor_script", "physical_actor_effects",
                     "rapier_physics_bridge", "rigid_body_coupling", "thunderbowl_arena",
                     "thunderbowl_contract", "actor_rig_motor"):
            path = f"godot/scripts/{name}.gd"
            scripts[path] = sha256_file(root / path)
        return {"repository": "techrote/cybersand", "host_revision": revision, "host_tree": tree,
                "godot_version": version, "godot_binary_sha256": godot_digest,
                "native_artifact_sha256": native_digest, "native_source_commit": native["source_commit"],
                "native_source_inputs": source_inputs, "rapier_artifact_sha256": rapier_digest,
                "host_script_sha256": scripts,
                "bridge_sha256": sha256_file(Path(__file__).with_name("host_bridge.gd"))}
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as exc:
        raise HostError(f"cannot establish the pinned host/runtime identity: {exc}") from exc


class GodotHost:
    """One persistent Godot process, one exclusive policy actor, one request in flight.

    Use as a context manager. The low-level request interface is for conformance;
    normal policies use CybersandEnv and receive local observations only.
    """

    def __init__(self, checkout: str | Path, godot: str | Path, *, lane: str = "act004",
                 mode: str = "stepped", timeout: float = 30.0) -> None:
        if lane not in {"act004", "sampled"} or mode not in {"stepped", "realtime"}:
            raise ValueError("unknown environment lane or stepping mode")
        if not 0 < timeout <= 120:
            raise ValueError("host deadline must be in (0,120] seconds")
        self.identity = inspect_host(checkout, godot)
        self.lane, self.mode = lane, mode
        self._request_id = 0
        self._closed = False
        self._channel: JsonChannel | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._diagnostics: collections.deque[str] = collections.deque(maxlen=100)
        self._temporary = tempfile.TemporaryDirectory(prefix="nca-host-")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(timeout)
            session_token = secrets.token_hex(32)
            environment = os.environ.copy()
            # Task-scoped runtime directories; never repurpose HOME.
            if os.name != "nt":
                for key, folder in (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"),
                                    ("XDG_CACHE_HOME", "cache")):
                    path = Path(self._temporary.name) / folder
                    path.mkdir()
                    environment[key] = str(path)
            command = [str(godot), "--headless", "--path", str(Path(checkout).resolve() / "godot"),
                       "--script", str(Path(__file__).with_name("host_bridge.gd").resolve()), "--",
                       f"--port={listener.getsockname()[1]}", f"--token={session_token}",
                       f"--lane={lane}", f"--mode={mode}"]
            self._process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, env=environment)
            self._reader = threading.Thread(target=self._drain_diagnostics, daemon=True)
            self._reader.start()
            connection, address = listener.accept()
            if address[0] != "127.0.0.1":
                connection.close()
                raise ProtocolError("nonlocal host connection")
            connection.settimeout(timeout)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._channel = JsonChannel(connection)
            hello = self._channel.receive()
            if (hello.get("protocol") != PROTOCOL or hello.get("id") != 0 or hello.get("token") != session_token
                    or hello.get("lane") != lane or hello.get("mode") != mode
                    or hello.get("godot_version", {}).get("major") != 4
                    or hello.get("godot_version", {}).get("minor") != 7
                    or not str(hello.get("godot_version", {}).get("hash", "")).startswith("5b4e0cb0f")
                    or not hello.get("native_available")
                    or (lane == "act004" and (not hello.get("rapier_available")
                        or hello.get("schedule") != PHYSICAL_SCHEDULE or hello.get("physics_engine") != "Rapier2D"))):
                raise ProtocolError("host capability handshake mismatch")
            hello.pop("token")
            self.capabilities = owned_json(hello)
            self.identity.update(lane=lane, mode=mode, schedule=hello["schedule"])
        except BaseException as exc:
            self.close(graceful=False)
            diagnostics = "\n".join(self._diagnostics)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise HostError(f"persistent host did not start: {exc}\n{diagnostics}") from exc
        finally:
            listener.close()

    def _drain_diagnostics(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for data in iter(self._process.stdout.readline, b""):
            self._diagnostics.append(data.decode("utf-8", errors="replace").rstrip()[:2000])

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    @property
    def pid(self) -> int:
        if self._process is None:
            raise HostError("host not running")
        return self._process.pid

    def request(self, operation: str, **fields: Any) -> dict[str, Any]:
        if {"id", "protocol", "op"}.intersection(fields):
            raise ValueError("request fields cannot replace protocol correlation")
        if self._closed or self._channel is None:
            raise HostError("host is closed")
        with self._lock:
            self._request_id += 1
            request = {"protocol": PROTOCOL, "id": self._request_id, "op": operation, **fields}
            try:
                self._channel.send(request)
                result = self._channel.receive()
                if result.get("protocol") != PROTOCOL or result.get("id") != self._request_id:
                    raise ProtocolError("host response correlation mismatch")
                if type(result.get("ok")) is not bool:
                    raise ProtocolError("missing host response status")
            except ProtocolError:
                self.close(graceful=False)
                raise
            if not result["ok"]:
                raise HostError(str(result.get("error", "host refused request")))
            return result

    def close(self, *, graceful: bool = True) -> None:
        if self._closed:
            return
        if graceful and self._channel is not None and self._process is not None and self._process.poll() is None:
            try:
                self._channel.connection.settimeout(2.0)
                self.request("close")
            except HostError:
                pass
        self._closed = True
        if self._channel is not None:
            self._channel.connection.close()
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
            if hasattr(self, "_reader"):
                self._reader.join(timeout=2)
            if self._process.stdout is not None:
                self._process.stdout.close()
        self._temporary.cleanup()

    def __enter__(self) -> GodotHost:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class CybersandEnv:
    """Gymnasium-style owned reset/step results over a persistent actual host.

    ACT004 has no game score. Its reward is explicitly 0.0 (no task reward),
    suitable for conformance or a separately declared task reward wrapper.
    ACT003's cumulative owner rewards are converted into interval differences.
    """

    def __init__(self, host: GodotHost, config: Mapping[str, Any], *, actor_id: int = 0,
                 decision_ticks: int = DEFAULT_DECISION_TICKS, lease_ticks: int = DEFAULT_LEASE_TICKS):
        integer(actor_id, 0, 7, "actor_id")
        integer(decision_ticks, 1, MAX_REPEAT_TICKS, "decision_ticks")
        integer(lease_ticks, 1, 18, "lease_ticks")
        self.host, self.config = host, owned_json(dict(config))
        self.actor_id, self.decision_ticks, self.lease_ticks = actor_id, decision_ticks, lease_ticks
        self._view: dict[str, Any] | None = None
        self._finished = True
        self._sequence = -1
        self._reward = RewardDifference()
        self._last_transition_tick = 0
        self._identity: tuple[Any, ...] | None = None

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        config = owned_json(self.config)
        if options:
            if set(options) != {"config"}:
                raise ValueError("reset options support only an explicit replacement config")
            config = owned_json(options["config"])
        if seed is not None:
            config["seed"] = integer(seed, 0, 2_147_483_647, "seed")
        response = self.host.request("reset", config=config, actor_id=self.actor_id)
        self._validate_transition(response, resetting=True)
        self.config = config
        self._view = owned_json(response["observation"])
        self._identity = self._stamp(self._view)
        self._finished = False
        self._sequence = int(self._view.get("last_sequence", -1))
        self._last_transition_tick = int(self._view["tick"])
        if self.host.lane == "sampled":
            self._reward.reset(self._view["run_id"], self._last_transition_tick, response["cumulative_reward"])
        return owned_json(self._view), self._info(response)

    @staticmethod
    def _stamp(view: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(view.get(key) for key in ("run_id", "epoch", "actor_id", "model", "profile_version"))

    def observe(self) -> dict[str, Any]:
        response = self.host.request("observe")
        self._validate_transition(response)
        self._view = owned_json(response["observation"])
        self._finished = bool(response["finished"])
        return owned_json(self._view)

    def command(self, action: Mapping[str, Any], *, observation: Mapping[str, Any] | None = None,
                lease_ticks: int | None = None, **sampled_options: Any) -> dict[str, Any]:
        if self._view is None or self._finished:
            raise EpisodeFinished("reset is required before creating a command")
        view = self._view if observation is None else observation
        if self._stamp(view) != self._identity:
            raise ProtocolError("action refers to an old actor, episode or profile")
        self._sequence += 1
        maker = make_physical_command if self.host.lane == "act004" else make_sampled_command
        return maker(view, self._sequence, action, self.lease_ticks if lease_ticks is None else lease_ticks, **sampled_options)

    def step(self, action: Mapping[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.step_packet(self.command(action))

    def step_packet(self, packet: Mapping[str, Any] | None, *, ticks: int | None = None) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Conformance surface: raw packets are never rewritten or freshly re-stamped."""
        if self._view is None or self._finished:
            raise EpisodeFinished("reset required; the terminal state cannot be rewarded twice")
        repeat = self.decision_ticks if ticks is None else integer(ticks, 1, 18, "ticks")
        response = self.host.request("step", commands=[] if packet is None else [owned_json(dict(packet))], ticks=repeat)
        self._validate_transition(response)
        actual = response.get("actual_ticks")
        start = response.get("start_tick")
        end = response.get("tick")
        traces = response.get("traces")
        late_terminal = (self.host.mode == "realtime" and response.get("finished")
                         and actual == 0 and type(end) is int and end > self._last_transition_tick)
        if (type(actual) is not int or not (1 <= actual <= repeat or late_terminal) or type(start) is not int
                or type(end) is not int or end - start != actual or start < self._last_transition_tick
                or not isinstance(traces, list) or len(traces) != actual):
            self._finished = True
            raise ProtocolError("host returned a noncausal or incomplete action-repeat interval")
        for offset, trace in enumerate(traces):
            if (trace.get("tick_before") != start + offset or trace.get("tick_after") != start + offset + 1
                    or trace.get("completed_monotonic_usec", -1) < trace.get("received_monotonic_usec", 0)
                    or (self.host.lane == "act004" and (trace.get("world_tick") != start + offset + 1
                        or trace.get("cellular_sample_tick") != start + offset))):
                self._finished = True
                raise ProtocolError("host trace violates the accepted causal schedule")
        self._view = owned_json(response["observation"])
        reward = 0.0
        if self.host.lane == "sampled":
            reward = self._reward.consume(self._view["run_id"], int(response["tick"]), response["cumulative_reward"])
        info = self._info(response)
        info["elapsed_ticks"] = int(response["tick"]) - self._last_transition_tick
        info["delay_ticks_before_repeat"] = int(response["start_tick"]) - self._last_transition_tick
        self._last_transition_tick = int(response["tick"])
        self._finished = bool(response["finished"])
        self._sequence = max(self._sequence, int(self._view.get("last_sequence", -1)))
        return owned_json(self._view), reward, bool(response["terminated"]), bool(response["truncated"]), info

    def cancel(self) -> dict[str, Any]:
        if self._view is None:
            raise HostError("reset required before cancellation")
        if self.host.lane == "sampled":
            neutral = {"movement": [0.0, 0.0], "aim": [0.0, 0.0], "effort": 0.0,
                       "trim": 0.0, "reel": 0.0, "primary": False, "secondary": False, "auxiliary": False}
            if self._view["control_mode"] == "metal":
                neutral = {"actuators": [0.0] * 4, "tag_aim": [0.0, 0.0]}
            return self.step_packet(self.command(neutral, cancel=True), ticks=1)[0]
        response = self.host.request("cancel", run_id=self._view["run_id"], epoch=self._view["epoch"])
        if not response.get("cancelled"):
            raise HostError("host refused cancellation")
        self._view = owned_json(response["observation"])
        return owned_json(self._view)

    def _validate_transition(self, response: Mapping[str, Any], *, resetting: bool = False) -> None:
        if response.get("failed"):
            self._finished = True
            raise HostIntegrityError(f"invalid host tick: {response.get('reason')}; no bootstrap observation")
        view = response.get("observation")
        if not isinstance(view, dict) or not view or view.get("actor_id") != self.actor_id:
            self._finished = True
            raise ProtocolError("missing or misaddressed local host observation")
        if response.get("schedule") != self.host.capabilities["schedule"] or view.get("tick") != response.get("tick"):
            raise ProtocolError("host observation and completed schedule disagree")
        stamp = self._stamp(view)
        # ACT003 explicitly increments its input epoch when _finish suspends
        # the same actor session. It is a terminal revocation, not a reset.
        sampled_terminal_fence = (
            not resetting and self.host.lane == "sampled" and response.get("finished")
            and self._identity is not None and stamp[0] == self._identity[0]
            and stamp[1] == self._identity[1] + 1 and stamp[2:] == self._identity[2:]
        )
        if not resetting and stamp != self._identity and not sampled_terminal_fence:
            self._finished = True
            raise ProtocolError("host changed episode, body or profile without reset")
        if resetting and self._view is not None and view["run_id"] == self._view["run_id"]:
            raise ProtocolError("reset failed to invalidate old observation identity")
        if resetting and (response.get("tick") != 0 or response.get("actual_ticks") != 0):
            raise ProtocolError("reset did not return the initial completed tick")
        if response.get("actual_ticks", -1) < 0:
            raise ProtocolError("host clock moved backwards")

    def _info(self, response: Mapping[str, Any]) -> dict[str, Any]:
        return owned_json({"environment_id": f"cybersand-{self.host.lane}/{self.host.mode}/v1",
                           "host_identity": self.host.identity, "decision_ticks": self.decision_ticks,
                           "helper_lease_ticks": self.lease_ticks, "reward_kind": response["reward_kind"],
                           "actual_ticks": response["actual_ticks"], "world_tick": response["world_tick"],
                           "reason": response["reason"], "trace": response["traces"],
                           "idle_trace": response.get("idle_trace", []),
                           "idle_trace_dropped": response.get("idle_trace_dropped", 0),
                           "consumed_trace_complete": response.get("idle_trace_dropped", 0) == 0,
                           "unconsumed_terminal_commands": response.get("unconsumed_terminal_commands", []),
                           "evidence_class": "actual-game" if self.host.lane == "act004" else "actual-sampled-host",
                           "owner_feel_accepted": False})

    def close(self) -> None:
        self.host.close()

    def __enter__(self) -> CybersandEnv:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def physical_config(model: str = "ripper", *, ticks: int = 600, seed: int = 0,
                    companions: tuple[str, ...] = ()) -> dict[str, Any]:
    """Explicit ACT004 configuration: one owned policy actor, optional authored demos."""
    return {"version": 2, "seed": integer(seed, 0, 2_147_483_647, "seed"),
            "max_ticks": integer(ticks, 1, 36000, "ticks"), "local_patch_radius": 6,
            "actors": [{"model": model, "producer": "policy"}]
            + [{"model": name, "producer": "demo"} for name in companions]}


def sampled_config(model: str = "ripper", *, ticks: int = 600, seed: int = 0,
                   metal: bool = False) -> dict[str, Any]:
    return {"version": 1, "seed": seed, "max_ticks": ticks, "layout": "training_pit",
            "evaluation": "mechanics", "actors": [{"model": model, "team": 0,
                "control_mode": "metal" if metal else "wire", "observation_mode": "local",
                "controller": "metal_pulse" if metal else "wire_seek"}]}
