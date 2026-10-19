"""Fixed trace playback, deliberately unable to impersonate a live game host."""

from __future__ import annotations

import math
from typing import Any, Mapping

from nca.adapters.protocol import EpisodeFinished, ProtocolError, owned_json


class RecordedEnv:
    """Replay exact requested inputs and owned outputs; do not synthesize transitions.

    The recording has a reset pair and a list of action/observation/reward/
    terminated/truncated/info records. Different inputs fail explicitly.
    """

    def __init__(self, recording: Mapping[str, Any]):
        self._recording = owned_json(dict(recording))
        if self._recording.get("format") != "nca-recorded-host/1":
            raise ProtocolError("unknown recording format")
        if not isinstance(self._recording.get("reset"), dict) or not isinstance(self._recording.get("transitions"), list):
            raise ProtocolError("recording requires reset and transitions")
        if not isinstance(self._recording["reset"].get("observation"), dict):
            raise ProtocolError("recording reset observation must be owned object data")
        finished = False
        for step in self._recording["transitions"]:
            if (not isinstance(step, dict) or finished or set(step) !=
                    {"action", "observation", "reward", "terminated", "truncated", "info"}):
                raise ProtocolError("invalid, incomplete or post-terminal recorded transition")
            if type(step["terminated"]) is not bool or type(step["truncated"]) is not bool:
                raise ProtocolError("recorded terminal flags must be Boolean")
            if type(step["reward"]) not in (int, float) or not math.isfinite(float(step["reward"])):
                raise ProtocolError("invalid recorded reward")
            finished = step["terminated"] or step["truncated"]
        self._index = 0
        self._finished = True

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        if seed is not None or options:
            raise ValueError("recorded playback cannot reseed or reconfigure a game")
        self._index, self._finished = 0, False
        record = self._recording["reset"]
        return owned_json(record["observation"]), self._info(record.get("info", {}))

    def _info(self, source: Mapping[str, Any]) -> dict[str, Any]:
        result = owned_json(dict(source))
        result.update(environment_id="recorded/playback/v1", evidence_class="recorded-playback",
                      source_environment=self._recording.get("source_environment", "unspecified"))
        return result

    def step(self, action: Mapping[str, Any]):
        if self._finished or self._index >= len(self._recording["transitions"]):
            raise EpisodeFinished("recording ended or requires reset")
        record = self._recording["transitions"][self._index]
        if owned_json(action) != record["action"]:
            raise ProtocolError("requested action differs from the fixed recording")
        self._index += 1
        self._finished = record["terminated"] or record["truncated"]
        return (owned_json(record["observation"]), float(record["reward"]), record["terminated"],
                record["truncated"], self._info(record["info"]))

    def close(self) -> None:
        self._finished = True
