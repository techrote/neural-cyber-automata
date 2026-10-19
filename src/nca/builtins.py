"""A minimal registered recorded lane for offline contract conformance.

The adapter returns recorded states; it does not simulate motion or reward.
"""

from .contracts import (Command, LifecycleStamp, Observation,
                        recorded_fixture_profile)
from .registry import Registry


class RecordedAdapter:
    def __init__(self):
        self.profile = recorded_fixture_profile()
        self.lifecycle = LifecycleStamp("recorded-example", "actor-0", 0)
        self.index = 0
        self.last_sequence = -1

    def observe(self) -> Observation:
        return Observation(f"recorded-{self.index}", self.index * 6,
                           self.lifecycle, self.profile.sha256, (0.0, None),
                           (True, False))

    def step(self, command: Command) -> Observation:
        command.validate_against(self.profile, self.observe(), self.index * 6,
                                 self.lifecycle, last_sequence=self.last_sequence)
        self.last_sequence = command.sequence
        self.index += 1
        return self.observe()


class IdlePolicy:
    def __call__(self, observation: Observation, sequence: int) -> Command:
        profile = recorded_fixture_profile()
        observation.validate_profile(profile)
        return Command(f"idle-{sequence}", observation.lifecycle,
                       observation.profile_sha256, observation.id,
                       observation.tick, sequence, 12, (0.0, 0.0),
                       profile.channel_mask)


class RecordedEvaluator:
    def __call__(self, observation: Observation) -> dict:
        observation.validate_profile(recorded_fixture_profile())
        return {"contract_valid": True, "gameplay_success": None,
                "evidence_class": "recorded protocol fixture"}


def builtin_registry() -> Registry:
    # These modules use the standard library and do not launch a runtime on
    # import. Construction of the host remains an explicit caller action.
    from .adapters.godot import CybersandEnv, GodotHost
    from .envs.recorded import RecordedEnv
    from .controllers import AuthoredController
    from .data import replay_episode

    registry = Registry()
    registry.register("profile", "recorded-rig", 1, recorded_fixture_profile,
                      ("recorded_fixture", "wire", "local-observation"))
    registry.register("adapter", "recorded", 1, RecordedAdapter, ("recorded_fixture",))
    registry.register("adapter", "recorded-playback", 1, RecordedEnv,
                      ("recorded_fixture", "reset-step", "owned-observation"))
    registry.register("adapter", "cybersand-host", 1, GodotHost,
                      ("persistent-process", "verified-runtime", "loopback"))
    registry.register("adapter", "cybersand-env", 1, CybersandEnv,
                      ("reset-step", "owned-observation", "duration-receipts"))
    registry.register("policy", "idle", 1, IdlePolicy, ("recorded_fixture",))
    registry.register("policy", "authored-feedback", 1, AuthoredController,
                      ("local-observation", "wire", "bounded-feedback", "cancellation"))
    registry.register("evaluator", "recorded", 1, RecordedEvaluator, ("recorded_fixture",))
    registry.register("evaluator", "consumed-replay", 1, replay_episode,
                      ("act004", "same-runtime", "original-observation-age"))
    return registry
