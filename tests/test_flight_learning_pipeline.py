import copy
from pathlib import Path
import random
import tempfile
import unittest

from nca.adapters.flight import CONTROLLER_SHA256, TRAINER_PROFILE_SHA256
from nca.contracts import ContractError, sha256_json
from nca.train.flight_accounting import (
    DEPENDENCIES,
    FlightAccountingLedger,
    FlightDeclaration,
    preparation_declaration,
)
from nca.train.flight_learning import (
    ACTION_SPEC,
    BC_EPOCHS,
    BC_SCHEDULE,
    BC_SCHEDULE_SHA256,
    DEVELOPMENT_SEEDS,
    MAX_CAPTURE_EPISODES,
    MAX_CAPTURE_LABELS,
    MAX_OPTIMIZER_UPDATES_PER_SEED,
    MAX_SAMPLE_PRESENTATIONS_PER_SEED,
    SHUFFLE_SEED_XOR,
    PREPROCESSING_SPEC,
    SCENARIO_FAMILY,
    build_capture_dataset,
    build_export_bundle,
    load_checkpoint_bytes,
    load_export_bundle,
    policy_input,
    policy_sha256,
    serialize_checkpoint,
    train_behavior_clone,
    validate_declared_capture_episode,
)


SOURCE = {
    "host_revision": "1" * 40,
    "host_build_sha256": "2" * 64,
    "profile_sha256": TRAINER_PROFILE_SHA256,
    "controller_sha256": CONTROLLER_SHA256,
    "teacher_sha256": "5" * 64,
}
TEACHER = {"id": "4ly003-teacher", "sha256": SOURCE["teacher_sha256"]}


def accounting_fixture():
    value = preparation_declaration().to_dict()
    value["status"] = "prospective"
    value["identities"] = {
        "learner_revision": "6" * 40,
        "learner_source_sha256": "7" * 64,
        "host_revision": SOURCE["host_revision"],
        "host_build_sha256": SOURCE["host_build_sha256"],
        "profile_sha256": SOURCE["profile_sha256"],
        "controller_sha256": SOURCE["controller_sha256"],
        "teacher_sha256": SOURCE["teacher_sha256"],
    }
    value["scenario"]["manifest_sha256"] = sha256_json(["fixture", "scenarios"])
    value["splits"].update({
        "development_sha256": sha256_json(["fixture", "development"]),
        "tuning_sha256": sha256_json(["fixture", "tuning"]),
        "final_commitment_sha256": sha256_json(["fixture", "sealed-final-commitment"]),
    })
    value["runtime"].update({
        "platform": "fixture-linux-x86_64",
        "python": "fixture-python",
        "torch": "fixture-torch-cpu",
        "threads": 1,
    })
    value["work_ceilings"].update({
        "host_calls": 100,
        "startup_operations": 4,
        "cleanup_operations": 4,
    })
    value["resource_ceilings"].update({
        "maximum_wall_seconds": 60.0,
        "maximum_resident_bytes": 1024 * 1024,
        "maximum_storage_bytes": 16 * 1024 * 1024,
        "maximum_publication_bytes": 4 * 1024 * 1024,
    })
    evidence = {}
    for name in DEPENDENCIES:
        digest = sha256_json(["fixture-evidence", name])
        value["dependencies"][name] = {
            "status": "accepted", "evidence_sha256": digest,
        }
        evidence[name] = digest
    declaration = FlightDeclaration.from_mapping(value)
    return declaration, evidence


ACCOUNTING_DECLARATION, ACCOUNTING_EVIDENCE = accounting_fixture()
DECLARATION = ACCOUNTING_DECLARATION.sha256


def observation(tick, *, suffix=None):
    suffix = tick if suffix is None else suffix
    flat = [0.0] * 54
    flat[0] = tick / 100.0
    return {
        "version": "4ly-observation/v1",
        "action_version": "4ly-action/v1",
        "run_id": "run-1",
        "actor_id": 0,
        "epoch": 1,
        "observation_id": f"obs-{suffix}",
        "observed_tick": tick,
        "control_generation": 0,
    }


def flat_observation(owner):
    values = [0.0] * 54
    values[0] = owner["observed_tick"] / 100.0
    return {
        "version": "4ly-flat-observation/v1",
        "field_count": 54,
        "source_observation_id": owner["observation_id"],
        "values": values,
    }


def action(before, *, status="accepted", move=(0.5, -0.25)):
    return {
        "source_id": TEACHER["id"],
        "source_sha256": TEACHER["sha256"],
        "observation_id": before["observation_id"],
        "status": status,
        "requested_action": list(move) if status == "accepted" else None,
        "admitted_action": list(move) if status == "accepted" else None,
        "reason": "accepted" if status == "accepted" else status,
    }


def outcome(status, *, terminated=False, truncated=False, failed=False, success=False, reason=None):
    return {
        "status": status,
        "reason": reason or status,
        "terminated": terminated,
        "truncated": truncated,
        "failed": failed,
        "success": success,
    }


def transition(index, before, after, *, teacher_status="accepted", physical_status="confirmed",
               terminated=False, truncated=False, failed=False, success=False):
    terminal_status = "failed" if failed else "completed" if terminated else "inconclusive"
    return {
        "index": index,
        "observation": before,
        "flat_observation": flat_observation(before),
        "teacher_action": action(before, status=teacher_status),
        "physical": {
            "status": physical_status,
            "consumed_ticks": 6 if physical_status == "confirmed" else None,
            "reason": physical_status,
        },
        "next_observation": after,
        "next_flat_observation": None if after is None else flat_observation(after),
        "outcome": outcome(
            terminal_status, terminated=terminated, truncated=truncated,
            failed=failed, success=success, reason="terminal" if terminated else terminal_status),
    }


def episode(*, episode_id="ep-1", transitions=None, final=None, split="development",
            initial=None, capture_id="capture-a", scenario_id="arrival_braking",
            host_seed=101):
    transitions = [] if transitions is None else transitions
    if initial is None and transitions:
        initial = transitions[0]["observation"]
    if final is None:
        final = outcome("refused", reason="reset_refused") if initial is None else outcome(
            "completed", terminated=True, success=True, reason="arrival")
    return {
        "schema": "nca-4ly-capture-episode/v1",
        "declaration_sha256": DECLARATION,
        "capture_id": capture_id,
        "episode_id": episode_id,
        "profile": "4ly_trainer",
        "split": split,
        "scenario_id": scenario_id,
        "host_seed": host_seed,
        "source": copy.deepcopy(SOURCE),
        "teacher": copy.deepcopy(TEACHER),
        "initial_observation": initial,
        "initial_flat_observation": None if initial is None else flat_observation(initial),
        "transitions": transitions,
        "outcome": final,
    }


def valid_capture(episode_id="ep-valid", *, scenario_id="arrival_braking", host_seed=101):
    before = observation(0)
    after = observation(6)
    row = transition(0, before, after, terminated=True, success=True)
    return validate_declared_capture_episode(
        episode(
            episode_id=episode_id,
            scenario_id=scenario_id,
            host_seed=host_seed,
            initial=before,
            transitions=[row],
            final=outcome("completed", terminated=True, success=True, reason="arrival"),
        ),
        declaration=ACCOUNTING_DECLARATION,
    )


class FlightCaptureTests(unittest.TestCase):
    def test_valid_capture_uses_admitted_action_and_only_54_policy_values(self):
        captured = valid_capture()
        self.assertEqual(captured.status, "completed")
        self.assertEqual(len(captured.samples), 1)
        self.assertEqual(captured.samples[0].action, (0.5, -0.25))
        self.assertEqual(len(captured.samples[0].observation), 54)
        self.assertEqual(captured.to_dict()["source"], SOURCE)

    def test_failed_and_refused_episodes_are_retained_but_bad_transitions_are_not_labels(self):
        zero, six, twelve = observation(0), observation(6), observation(12)
        first = transition(0, zero, six)
        second = transition(1, six, twelve, failed=True)
        failed = validate_declared_capture_episode(
            episode(
                episode_id="ep-failed",
                initial=zero,
                transitions=[first, second],
                final=outcome("failed", failed=True, reason="physical_fault"),
            ),
            declaration=ACCOUNTING_DECLARATION,
        )
        refused = validate_declared_capture_episode(
            episode(
                episode_id="ep-refused",
                scenario_id="arrival_braking",
                host_seed=103,
                initial=None,
                transitions=[],
                final=outcome("refused", reason="reset_refused"),
            ),
            declaration=ACCOUNTING_DECLARATION,
        )
        self.assertEqual(failed.status, "failed")
        self.assertEqual(len(failed.samples), 1)
        self.assertIn("1:failed_transition", failed.exclusions)
        self.assertEqual(refused.status, "refused")
        self.assertEqual(refused.samples, ())
        self.assertIn("episode_without_authoritative_initial_observation", refused.exclusions)
        dataset = build_capture_dataset((failed, refused))
        self.assertEqual(len(dataset.episodes), 2)
        self.assertEqual(len(dataset.samples), 1)
        # The refused episode retains its nominal half of this family's episode
        # mass instead of silently transferring weight to the successful row.
        self.assertAlmostEqual(dataset.sample_weights[0], 1.0 / 12.0, places=12)

    def test_confirmed_zero_tick_failure_is_retained_without_supervision(self):
        before = observation(0)
        failed = transition(0, before, copy.deepcopy(before), failed=True)
        failed["physical"]["consumed_ticks"] = 0
        captured = validate_declared_capture_episode(
            episode(
                episode_id="ep-zero-tick-failed",
                initial=before,
                transitions=[failed],
                final=outcome("failed", failed=True, reason="expired_lease"),
            ),
            declaration=ACCOUNTING_DECLARATION,
        )
        self.assertEqual(captured.samples, ())
        self.assertIn("0:failed_transition", captured.exclusions)
        self.assertEqual(
            captured.to_dict()["transitions"][0]["physical"]["consumed_ticks"], 0)

    def test_consumed_physical_failure_may_revoke_generation_once(self):
        before = observation(0)
        after = observation(6)
        after["control_generation"] = 1
        after["observation_id"] = "obs-6-revoked"
        failed = transition(0, before, after, failed=True)
        failed["next_flat_observation"] = flat_observation(after)
        captured = validate_declared_capture_episode(
            episode(
                episode_id="ep-consumed-failed-revoked",
                initial=before,
                transitions=[failed],
                final=outcome("failed", failed=True, reason="physical_fault"),
            ),
            declaration=ACCOUNTING_DECLARATION,
        )
        self.assertEqual(captured.samples, ())
        self.assertIn("0:failed_transition", captured.exclusions)

    def test_unknown_consumption_and_refused_teacher_never_become_supervision(self):
        before = observation(0)
        unknown = transition(
            0, before, None, teacher_status="accepted", physical_status="unknown", failed=True)
        captured = validate_declared_capture_episode(
            episode(
                episode_id="ep-unknown",
                initial=before,
                transitions=[unknown],
                final=outcome("failed", failed=True, reason="lost_reply"),
            ),
            declaration=ACCOUNTING_DECLARATION,
        )
        self.assertEqual(captured.samples, ())
        self.assertIn("0:unknown_physical_consumption", captured.exclusions)

        after = observation(6)
        refused = transition(0, before, after, teacher_status="refused", terminated=True)
        captured = validate_declared_capture_episode(
            episode(
                episode_id="ep-teacher-refused",
                initial=before,
                transitions=[refused],
                final=outcome("completed", terminated=True, reason="stopped"),
            ),
            declaration=ACCOUNTING_DECLARATION,
        )
        self.assertEqual(captured.samples, ())
        self.assertIn("0:teacher_refused", captured.exclusions)

    def test_schema_final_and_identity_mismatch_refuse(self):
        base = valid_capture().to_dict()
        for mutation in (
            lambda value: value.update(split="final"),
            lambda value: value.update(split="training"),
            lambda value: value.update(declaration_sha256="e" * 64),
            lambda value: value["source"].update(host_revision="f" * 40),
            lambda value: value["initial_observation"].update(version="legacy"),
        ):
            value = copy.deepcopy(base)
            mutation(value)
            with self.assertRaises(ContractError):
                validate_declared_capture_episode(
                    value, declaration=ACCOUNTING_DECLARATION)

    def test_policy_input_does_not_traverse_evaluator_only_state(self):
        class Explodes:
            def __call__(self):
                raise AssertionError("evaluator/teacher must not be called")
        view = observation(0)
        view["evaluator"] = Explodes()
        view["navigation_teacher"] = Explodes()
        self.assertEqual(len(policy_input(view, flat_observation(view))), 54)


@unittest.skipUnless(__import__("importlib").util.find_spec("torch") is not None, "optional Torch unavailable")
class FlightLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.captures = tuple(
            valid_capture(
                f"ep-{scenario_id}-{host_seed}",
                scenario_id=scenario_id,
                host_seed=host_seed,
            )
            for scenario_id in sorted(SCENARIO_FAMILY)
            for host_seed in DEVELOPMENT_SEEDS
        )
        cls.capture = next(
            item for item in cls.captures
            if item.scenario_id == "arrival_braking" and item.host_seed == 101
        )
        cls.dataset = build_capture_dataset(cls.captures)
        assert len(cls.captures) == 44

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ledger_index = 0

    def tearDown(self):
        self.temp.cleanup()

    def train(self, seed):
        self.ledger_index += 1
        attempt_id = f"fixture-{self.ledger_index}"
        ledger = FlightAccountingLedger.create(
            Path(self.temp.name) / f"{attempt_id}.jsonl",
            ACCOUNTING_DECLARATION,
            attempt_id=attempt_id,
            current_evidence=ACCOUNTING_EVIDENCE,
        )
        result = train_behavior_clone(
            self.dataset, seed=seed, accounting=ledger)
        snapshot = ledger.close("synthetic_learning_fixture")
        self.assertEqual(
            snapshot["charged"]["optimizer_updates"],
            result.optimizer_updates,
        )
        self.assertEqual(
            snapshot["charged"]["sample_presentations"],
            result.sample_presentations,
        )
        return result

    def test_clone_refuses_without_source_bound_accounting(self):
        with self.assertRaisesRegex(ContractError, "accounting"):
            train_behavior_clone(self.dataset, seed=74011)

    def test_clone_refuses_an_incomplete_preregistered_capture_set(self):
        partial = build_capture_dataset(self.captures[:-1])
        ledger = FlightAccountingLedger.create(
            Path(self.temp.name) / "partial.jsonl",
            ACCOUNTING_DECLARATION,
            attempt_id="partial-fixture",
            current_evidence=ACCOUNTING_EVIDENCE,
        )
        try:
            with self.assertRaisesRegex(ContractError, "44-case"):
                train_behavior_clone(partial, seed=74011, accounting=ledger)
        finally:
            ledger.close("synthetic_incomplete_capture_fixture")

    def test_preregistered_cloning_ceiling_contract(self):
        self.assertEqual(MAX_CAPTURE_EPISODES, 44)
        self.assertEqual(MAX_CAPTURE_LABELS, 11400)
        self.assertEqual(MAX_OPTIMIZER_UPDATES_PER_SEED, 1800)
        self.assertEqual(MAX_SAMPLE_PRESENTATIONS_PER_SEED, 228000)
        self.assertEqual(BC_SCHEDULE["checkpoint_epochs"], list(range(21)))
        self.assertEqual(BC_SCHEDULE["tuning_candidate_epochs"], [5, 10, 15, 20])
        self.assertEqual(BC_SCHEDULE["shuffle_seed_xor"], SHUFFLE_SEED_XOR)
        self.assertEqual(BC_SCHEDULE["required_capture_episodes"], 44)
        self.assertEqual(BC_SCHEDULE["maximum_capture_labels"], 11400)
        self.assertEqual(BC_SCHEDULE["maximum_capture_ticks"], 68400)
        self.assertEqual(BC_SCHEDULE["maximum_capture_decisions"], 11400)
        self.assertEqual(BC_SCHEDULE["scenario_max_ticks"]["slalom"], 2400)
        self.assertEqual(BC_SCHEDULE["scenario_max_ticks"]["wall_recovery"], 2100)
        self.assertEqual(BC_SCHEDULE_SHA256, sha256_json(BC_SCHEDULE))

    def test_preregistered_family_weights_are_equal(self):
        totals = {}
        for sample, weight in zip(self.dataset.samples, self.dataset.sample_weights):
            totals[sample.family] = totals.get(sample.family, 0.0) + weight
        self.assertEqual(
            set(totals),
            {"hover", "free_space", "reversal", "arrival", "obstacles", "recovery"},
        )
        self.assertAlmostEqual(sum(self.dataset.sample_weights), 1.0, places=12)
        for value in totals.values():
            self.assertAlmostEqual(value, 1.0 / 6.0, places=12)

    def test_bounded_clone_is_deterministic_for_the_declared_seed(self):
        first = self.train(74011)
        second = self.train(74011)
        self.assertEqual(first.completed_epochs, BC_EPOCHS)
        self.assertEqual([item.epoch for item in first.checkpoints], list(range(21)))
        self.assertEqual(first.checkpoints[0].optimizer_updates, 0)
        self.assertEqual(first.checkpoints[0].sample_presentations, 0)
        expected_presentations = len(self.dataset.samples) * BC_EPOCHS
        self.assertEqual(first.optimizer_updates, BC_EPOCHS)
        self.assertEqual(first.sample_presentations, expected_presentations)
        self.assertEqual(first.checkpoints[-1].optimizer_updates, first.optimizer_updates)
        self.assertEqual(
            first.checkpoints[-1].sample_presentations, first.sample_presentations)
        self.assertEqual(first.epoch_losses, second.epoch_losses)
        self.assertEqual(policy_sha256(first.model), policy_sha256(second.model))

    def test_checkpoint_round_trip_preserves_model_optimizer_rng_and_accounting_lineage(self):
        import torch
        trained = self.train(74013)
        initial_data, initial_receipt = serialize_checkpoint(
            trained,
            checkpoint_id="seed74013-epoch00",
            learner_revision="6" * 40,
            learner_source_sha256="7" * 64,
            checkpoint_epoch=0,
        )
        initial = load_checkpoint_bytes(initial_data, initial_receipt)
        self.assertEqual(initial.lineage["completed_epochs"], 0)
        self.assertEqual(initial.lineage["optimizer_updates"], 0)
        self.assertEqual(initial.lineage["sample_presentations"], 0)

        data, receipt = serialize_checkpoint(
            trained,
            checkpoint_id="seed74013-epoch20",
            learner_revision="6" * 40,
            learner_source_sha256="7" * 64,
            checkpoint_epoch=20,
            parent_checkpoint_sha256=initial_receipt["sha256"],
        )
        before_global = torch.get_rng_state().clone()
        restored = load_checkpoint_bytes(data, receipt)
        self.assertTrue(torch.equal(before_global, torch.get_rng_state()))
        self.assertTrue(torch.equal(trained.torch_rng_state, restored.torch_rng_state))
        self.assertEqual(policy_sha256(restored.model), receipt["policy_sha256"])
        self.assertEqual(restored.lineage["declaration_sha256"], DECLARATION)
        self.assertEqual(restored.lineage["completed_epochs"], 20)
        self.assertEqual(
            restored.lineage["parent_checkpoint_sha256"], initial_receipt["sha256"])
        self.assertEqual(restored.lineage["capture_dataset_sha256"], self.dataset.sha256)
        self.assertEqual(restored.lineage["capture_id"], "capture-a")
        self.assertEqual(restored.lineage["optimizer_updates"], BC_EPOCHS)
        self.assertEqual(
            restored.lineage["sample_presentations"],
            len(self.dataset.samples) * BC_EPOCHS,
        )
        self.assertEqual(
            restored.lineage["accounting_attempt_id"],
            trained.accounting_attempt_id,
        )
        self.assertEqual(
            restored.lineage["accounting_chain_sha256"],
            trained.accounting_chain_sha256,
        )
        self.assertNotEqual(trained.accounting_chain_sha256, "0" * 64)

        original_probe = random.Random()
        original_probe.setstate(trained.shuffle_rng.getstate())
        self.assertEqual(original_probe.random(), restored.shuffle_rng.random())

        original_state = trained.optimizer.state_dict()
        restored_state = restored.optimizer.state_dict()
        self.assertEqual(original_state["param_groups"], restored_state["param_groups"])
        self.assertEqual(set(original_state["state"]), set(restored_state["state"]))
        for key in original_state["state"]:
            for name, value in original_state["state"][key].items():
                other = restored_state["state"][key][name]
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, other))
                else:
                    self.assertEqual(value, other)

    def test_export_round_trip_is_exact_bounded_and_teacher_free(self):
        trained = self.train(74017)
        data, receipt = serialize_checkpoint(
            trained,
            checkpoint_id="seed74017-epoch20",
            learner_revision="8" * 40,
            learner_source_sha256="9" * 64,
        )
        restored = load_checkpoint_bytes(data, receipt)
        wrapper = build_export_bundle(restored)
        self.assertEqual(wrapper["bundle"]["preprocessing"], PREPROCESSING_SPEC)
        self.assertEqual(wrapper["bundle"]["action"], ACTION_SPEC)
        self.assertIsNone(wrapper["bundle"]["preprocessing"]["additional_fitted_statistics"])
        policy = load_export_bundle(wrapper)

        calls = []
        view = observation(0)
        view["navigation_teacher"] = lambda *_: calls.append("teacher")
        view["evaluator_only"] = {"success": True, "reward": 999}
        result = policy.act(view, flat_observation(view))
        self.assertEqual(calls, [])
        self.assertLessEqual(sum(value * value for value in result["move"]), 1.0 + 1e-7)
        self.assertEqual(policy.policy_sha256, receipt["policy_sha256"])

    def test_export_and_policy_schema_mutations_fail_closed(self):
        trained = self.train(74011)
        data, receipt = serialize_checkpoint(
            trained,
            checkpoint_id="seed74011-epoch20",
            learner_revision="a" * 40,
            learner_source_sha256="b" * 64,
        )
        wrapper = build_export_bundle(load_checkpoint_bytes(data, receipt))
        changed = copy.deepcopy(wrapper)
        changed["bundle"]["preprocessing"]["field_count"] = 53
        changed["bundle_sha256"] = sha256_json(changed["bundle"])
        with self.assertRaises(ContractError):
            load_export_bundle(changed)
        bad_view = observation(0)
        bad_view["version"] = "legacy"
        with self.assertRaises(ContractError):
            policy_input(bad_view, flat_observation(bad_view))

    def test_dataset_refuses_mixed_accounting_or_capture_identity(self):
        second_value = valid_capture("ep-second").to_dict()
        second_value["capture_id"] = "capture-b"
        second = validate_declared_capture_episode(
            second_value, declaration=ACCOUNTING_DECLARATION)
        with self.assertRaises(ContractError):
            build_capture_dataset((self.capture, second))


if __name__ == "__main__":
    unittest.main()
