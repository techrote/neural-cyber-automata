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


def first_skill_screen(**kwargs):
    """Load optional training dependencies only on explicit invocation."""
    from .train.screen import run_screen
    return run_screen(**kwargs)


def learned_jolt_policy(*args, **kwargs):
    """Load the versioned optional learned runtime on explicit construction."""
    from .controllers.jolt_learned import JoltController
    return JoltController(*args, **kwargs)


def learn_jolt_screen(**kwargs):
    """Execute only through the new trainer's frozen admission contract."""
    from .train.jolt_screen import run
    return run(**kwargs)


def learn_jolt_screen_v3(**kwargs):
    """Select the fresh execution contract without reinterpreting trainer v2."""
    from .contracts import ContractError
    version = kwargs.pop("execution_version", 3)
    if type(version) is not int or version != 3:
        raise ContractError("jolt_execution_version", "trainer learn-jolt@3 requires execution v3")
    from .train.jolt_screen import run
    return run(execution_version=3, **kwargs)


def cpu_export_policy(*args, **kwargs):
    """Construct the optional NumPy runtime only on explicit invocation."""
    from .export import ExportController
    return ExportController(*args, **kwargs)


def torch_cpu_export_policy(*args, **kwargs):
    """Load the separately declared Torch CPU deployment backend explicitly."""
    from .export.torch_runtime import torch_export_controller
    return torch_export_controller(*args, **kwargs)


def padded_morphology_policy(**kwargs):
    """Construct the optional untrained padded candidate explicitly."""
    from .controllers.morphology.models import MaskedRecurrentPolicy
    return MaskedRecurrentPolicy(**kwargs)


def graph_morphology_policy(**kwargs):
    """Construct the optional untrained local graph candidate explicitly."""
    from .controllers.morphology.models import SharedGraphPolicy
    return SharedGraphPolicy(**kwargs)


def ik_runner_profile():
    """Load the source-derived point-IK descriptor without optional runtimes."""
    from .ik.profiles import runner_right_sole_profile
    return runner_right_sole_profile()


def ik_cpu_fk(**kwargs):
    from .ik.kinematics import CPUFK
    return CPUFK(**kwargs)


def ik_torch_fk(**kwargs):
    from .ik.kinematics import TorchFK
    return TorchFK(**kwargs)


def ik_reference_solver(**kwargs):
    from .ik.reference import DampedLeastSquares
    return DampedLeastSquares(**kwargs)


def ik_analytic_solver(**kwargs):
    from .ik.reference import AnalyticTwoLink
    return AnalyticTwoLink(**kwargs)


def ik_compact_model(**kwargs):
    from .ik.models import build_model
    return build_model(**kwargs)


def ik_compact_training_step(**kwargs):
    from .train.ik import IKTrainer
    return IKTrainer(**kwargs)


def ik_polar_model(**kwargs):
    from .ik.polar import build_model
    return build_model(**kwargs)


def ik_polar_training_step(**kwargs):
    from .ik.polar import build_trainer
    return build_trainer(**kwargs)


def ik_compact_continuation(**kwargs):
    from .train.ik_continuation import supervise
    return supervise(**kwargs)


def ik_authenticated_final(**kwargs):
    from .evaluation.ik_final_corpus import open_final_corpus
    return open_final_corpus(**kwargs)


def ik_compact_final_comparison(**kwargs):
    from .evaluation.ik_continuation_eval import supervise
    return supervise(**kwargs)


def ik_frozen_recovery(**kwargs):
    from .evaluation.ik_recovery import supervise
    return supervise(**kwargs)


def builtin_registry() -> Registry:
    # These modules use the standard library and do not launch a runtime on
    # import. Construction of the host remains an explicit caller action.
    from .adapters.godot import CybersandEnv, GodotHost
    from .envs.recorded import RecordedEnv
    from .controllers import AuthoredController
    from .controllers.jolt_arrival import JoltArrivalController
    from .controllers.jolt_recovery import JoltRecoveryController
    from .controllers.jolt_recovery_v2 import JoltRecoveryControllerV2
    from .controllers.jolt_recovery_v3 import JoltRecoveryControllerV3
    from .controllers.jolt_recovery_v4 import JoltRecoveryControllerV4
    from .data import replay_episode
    from .skills import register_skills
    from .cellular import register as register_cellular
    from .adapters.skill_control import register_decisions

    registry = Registry()
    registry.register("profile", "recorded-rig", 1, recorded_fixture_profile,
                      ("recorded_fixture", "wire", "local-observation"))
    registry.register("profile", "runner-right-sole", 1, ik_runner_profile,
                      ("nominal-geometry", "point-ik-2d", "root-local", "fixed-chain"))
    registry.register("kinematics", "planar-cpu-f64", 1, ik_cpu_fk,
                      ("nominal-geometry", "point-ik-2d", "jacobian", "binary64"))
    registry.register("kinematics", "planar-torch", 1, ik_torch_fk,
                      ("nominal-geometry", "point-ik-2d", "jacobian", "differentiable", "explicit-device"))
    registry.register("ik_solver", "runner-dls", 1, ik_reference_solver,
                      ("nominal-geometry", "bounded-reference", "source-bound", "point-ik-2d"))
    registry.register("ik_solver", "planar-analytic-2r", 1, ik_analytic_solver,
                      ("nominal-geometry", "independent-comparator", "multiple-branches"))
    registry.register("policy", "runner-compact-ik", 1, ik_compact_model,
                      ("optional-torch", "nominal-geometry", "bounded-reference", "explicit-device", "masked"))
    registry.register("policy", "runner-compact-ik", 2, ik_polar_model,
                      ("optional-torch", "polar-target", "bounded-reference", "explicit-device", "masked"))
    registry.register("trainer", "runner-compact-ik-step", 2, ik_polar_training_step,
                      ("optional-torch", "polar-target", "raw-cartesian-loss", "single-update"))
    registry.register("trainer", "runner-compact-ik-step", 1, ik_compact_training_step,
                      ("optional-torch", "training-only", "single-update", "explicit-device", "software-foundation"))
    registry.register("trainer", "runner-compact-ik-continuation", 1, ik_compact_continuation,
                      ("amendment-bound", "original-allocation", "titan-only", "supervised"))
    registry.register("evaluator", "ik-authenticated-final-corpus", 1, ik_authenticated_final,
                      ("original-byte-authentication", "candidate-bound-opening", "no-pending-journal"))
    registry.register("evaluator", "ik-compact-final-comparison", 1, ik_compact_final_comparison,
                      ("three-seed-freeze", "original-allocation", "supervised", "matched-cpu-timing"))
    registry.register("evaluator", "ik-frozen-recovery", 1, ik_frozen_recovery,
                      ("same-frozen-candidates", "append-only-costs", "no-training", "titan-only"))
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
    registry.register("policy", "authored-jolt-arrival", 1, JoltArrivalController,
                      ("local-observation", "wire", "bounded-feedback", "arrival-diagnostic", "cancellation"))
    registry.register("policy", "authored-jolt-recovery", 1, JoltRecoveryController,
                      ("local-observation", "wire", "contact-recovery", "separate-supervisor", "cancellation"))
    registry.register("policy", "authored-jolt-recovery", 2, JoltRecoveryControllerV2,
                      ("local-observation", "wire", "contact-recovery", "arrival-advisory", "separate-supervisor", "cancellation"))
    registry.register("policy", "authored-jolt-recovery", 3, JoltRecoveryControllerV3,
                      ("local-observation", "wire", "contact-recovery", "arrival-advisory", "fresh-contact-transition", "separate-supervisor", "cancellation"))
    registry.register("policy", "authored-jolt-recovery", 4, JoltRecoveryControllerV4,
                      ("local-observation", "wire", "contact-recovery", "arrival-advisory", "interior-capture", "fresh-contact-transition", "separate-supervisor", "cancellation"))
    registry.register("evaluator", "recorded", 1, RecordedEvaluator, ("recorded_fixture",))
    registry.register("evaluator", "consumed-replay", 1, replay_episode,
                      ("act004", "same-runtime", "original-observation-age"))
    registry.register("trainer", "first-jolt-screen", 1, first_skill_screen,
                      ("optional-cpu", "bounded-experiment", "recurrent", "source-attribution"))
    registry.register("policy", "jolt-learned", 2, learned_jolt_policy,
                      ("optional-torch", "cpu", "recurrent", "learned-aim-buttons", "separate-supervisor", "cancellation"))
    registry.register("trainer", "learn-jolt", 2, learn_jolt_screen,
                      ("optional-cpu", "frozen-admission", "bounded-experiment", "recurrent", "source-attribution"))
    registry.register("trainer", "learn-jolt", 3, learn_jolt_screen_v3,
                      ("optional-cpu", "frozen-admission", "safe-retirement", "original-clock", "bounded-streaming"))
    registry.register("policy", "jolt-cpu-export", 1, cpu_export_policy,
                      ("optional-numpy", "evaluated-bundle", "runtime-context", "cpu"))
    registry.register("policy", "jolt-torch-cpu-export", 1, torch_cpu_export_policy,
                      ("optional-torch", "evaluated-bundle", "explicit-backend", "cpu"))
    registry.register("policy", "morphology-padded-recurrent", 1, padded_morphology_policy,
                      ("optional-cpu", "gated-preparation", "masked", "recurrent"))
    registry.register("policy", "morphology-shared-graph", 1, graph_morphology_policy,
                      ("optional-cpu", "gated-preparation", "local-messages", "recurrent"))
    register_skills(registry)
    register_cellular(registry)
    register_decisions(registry)
    return registry
