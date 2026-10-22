"""Reproduce an offline allocation proposal; this file cannot admit or run it.

Run from any directory with Python's standard library. Output goes to stdout.
No host, framework, checkpoint, network or executable admission is imported.
© 2026 techrote. All rights reserved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = "f23401eb793f9708040c4047a8146ab01ec3befc"


def read(relative):
    data = (ROOT / relative).read_bytes()
    return json.loads(data), {"path": relative, "sha256": hashlib.sha256(data).hexdigest()}


def calculate():
    original, configuration = read("configs/learning/jolt-learn-002-v1.json")
    costs, cost_source = read("docs/evidence/jolt-learn-002-costs-20261021.json")
    forensic, forensic_source = read("docs/evidence/jolt-learn-002-forensic-review-20261021.json")
    seeds, epochs, episodes, unroll = 3, 40, 64, 16
    updates, rollout, ppo_epochs, maximum_episode, cadence = 8, 256, 4, 40, 6
    stages = {
        "teacher": {"episodes_started": episodes, "decisions": episodes * maximum_episode},
        "ppo": {"episodes_started": seeds * updates * rollout, "decisions": seeds * updates * rollout},
        "open": {"episodes_started": 4 * 8, "decisions": 4 * 8 * maximum_episode},
        "sealed": {"episodes_started": 16 * 14, "decisions": 16 * 14 * maximum_episode},
    }
    for stage in stages.values():
        stage["requested_ticks"] = cadence * stage["decisions"]
    total_decisions = sum(stage["decisions"] for stage in stages.values())
    total_episodes = sum(stage["episodes_started"] for stage in stages.values())

    # Every partition of a 256-row update into positive physical episodes <=40.
    # A complete fixed-window prefix costs 0,16,48 rows for lengths <=16,<=32,<=40.
    prefix = [sum(range(0, length, unroll)) for length in range(maximum_episode + 1)]
    best = [0] + [-1] * rollout
    witness = [[] for _ in range(rollout + 1)]
    for rows in range(1, rollout + 1):
        for length in range(1, min(rows, maximum_episode) + 1):
            candidate = best[rows - length] + prefix[length]
            if candidate > best[rows]:
                best[rows] = candidate
                witness[rows] = witness[rows - length] + [length]
    ppo_prefix = best[rollout] * seeds * updates * ppo_epochs

    learned_physical_rows = stages["ppo"]["decisions"] + 4 * 6 * maximum_episode + 16 * 12 * maximum_episode
    learned_episodes = stages["ppo"]["episodes_started"] + 4 * 6 + 16 * 12
    inference = 2 * stages["ppo"]["decisions"] + 4 * (3 + 2 * 3) * maximum_episode + 16 * (6 + 2 * 6) * maximum_episode
    counters = dict(original["budget"]["counters"])
    counters.update(checkpoint_executions=learned_physical_rows + learned_episodes,
                    inference_forward=inference, ppo_burn_in=ppo_prefix, ppo_base_burn_in=ppo_prefix)
    owner_formula = (total_decisions + 3 * total_episodes + 2)
    assert counters["owner_requests"] == owner_formula
    assert stages == original["budget"]["stages"]
    assert counters["bc_optimizer"] == seeds * epochs * episodes * 3
    assert counters["bc_burn_in"] == seeds * epochs * episodes * sum(range(0, maximum_episode, unroll))
    assert best[rollout] == 352 and ppo_prefix == 33792

    forward_names = ("bc_forward", "bc_burn_in", "ppo_forward", "ppo_base_forward",
                     "ppo_burn_in", "ppo_base_burn_in", "ppo_replay_forward",
                     "ppo_replay_base_forward", "inference_forward", "bootstrap_forward", "audit_forward")
    forward_total = sum(counters[name] for name in forward_names)
    observed = costs["original_reported_costs"]["counts"]
    internal = costs["internal_ledger_seconds"]
    prior = costs["lineage_subtotal"]["recovery_v4_increment"]
    same_blend = internal * total_decisions / observed["decisions"]
    diagnostic_proxy = prior["whole_process_seconds"] * total_decisions / prior["decisions"]
    reservations = costs["original_reported_costs"]["reservation_count"]
    reservation_bytes = forensic["original_files"]["budget-reservations.json"]["bytes"]
    count_backstop = sum(counters.values())
    mean_bytes = reservation_bytes / reservations
    fixed_journal_notes = 2 + 36 + 54 + 3 + 2
    journal_base = 2 * total_episodes + 2 * owner_formula + total_decisions + fixed_journal_notes
    journal_notes = journal_base + total_episodes + 1
    worker_files = 3 * total_episodes + 36 + 27 + 13
    maximum_wall, cleanup, verification, terminal = 7200, 1200, 300, 60
    assert maximum_wall > cleanup > verification > terminal > 0

    return {
        "schema": "nca-learn-jolt-003-allocation-proposal/v1",
        "reviewed_base_revision": BASE,
        "status": "proposal_for_implementation_and_admission_preparation",
        "offline_scope_only": True,
        "prospective_experiment_admitted": False,
        "owner_execution_commission_recorded": False,
        "new_physical_or_neural_work_performed": False,
        "proposed_experiment": "LEARN-JOLT-003",
        "source_bindings": [configuration, cost_source, forensic_source],
        "calculator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixed_training": {"seeds": original["training_seeds"], "teacher_episodes": 64,
                           "bc_epochs_per_seed": 40, "ppo_updates_per_seed": 8,
                           "ppo_decisions_per_update": 256, "ppo_epochs_per_update": 4,
                           "unroll": 16, "maximum_episode_decisions": 40,
                           "decision_ticks": 6, "open_rows": 32, "sealed_rows": 224,
                           "final_checkpoints_before_any_sealed_outcome": 6},
        "proposed_budget": {
            "maximum_attempts": 1, "automatic_retry": False, "automatic_extension": False,
            "device": "cpu", "gpu_allowed": False, "torch_threads": 1,
            "maximum_owner_transport_seconds": 15,
            "maximum_wall_seconds": maximum_wall, "cleanup_reserve_seconds": cleanup,
            "executor_verification_reserve_seconds": verification,
            "executor_finalization_reserve_seconds": terminal,
            "counters": counters, "stages": stages,
        },
        "absolute_phase_offsets_seconds": {
            "ordinary_work_stopped": maximum_wall - cleanup,
            "worker_exit_cleanup_and_streams_complete": maximum_wall - verification,
            "external_inventory_exit_cleanup_and_streams_complete": maximum_wall - terminal,
            "terminal_publication_stdout_and_observed_exit_complete": maximum_wall,
        },
        "nonoverlapping_intervals_seconds": {
            "active_including_startup_training_and_all_physical_evaluation": maximum_wall - cleanup,
            "worker_shutdown_neural_readback_and_publication": cleanup - verification,
            "external_inventory_verification_and_cleanup": verification - terminal,
            "terminal_publication_stdout_and_observed_exit": terminal,
        },
        "derived_bounds": {
            "physical_decisions": total_decisions,
            "physical_requested_ticks": total_decisions * cadence,
            "episodes": total_episodes,
            "owner_requests": owner_formula,
            "learned_physical_rows": learned_physical_rows,
            "learned_episode_starts": learned_episodes,
            "retirement_attempt_allowance_without_forward": learned_episodes,
            "ppo_prefix_rows_per_update_per_epoch": best[rollout],
            "ppo_prefix_witness_episode_lengths": witness[rollout],
            "ppo_prefix_rows_per_module_all_updates_epochs": ppo_prefix,
            "optimizer_calls_bc_plus_ppo": counters["bc_optimizer"] + counters["ppo_optimizer"],
            "module_forwards_with_conservative_bootstrap_and_audit_caps": forward_total,
            "checkpoint_writes": 3 * (2 + 1 + 8 + 1),
            "checkpoint_loads": {"evaluation_cache": 12, "independent_audit": 36, "total": 48},
            "reservation_records_loose_upper_bound": count_backstop,
            "journal_note_bound_terms": {"episode_start_and_finish": 2 * total_episodes,
                                         "owner_request_and_response": 2 * owner_formula,
                                         "decision_proposed": total_decisions,
                                         "fixed_notes": fixed_journal_notes,
                                         "one_retirement_note_per_episode": total_episodes,
                                         "possible_allocation_stop_note": 1},
            "journal_notes_with_one_new_retirement_note_per_episode": journal_notes,
            "worker_file_bound_terms": {"episode_sample_metrics": 3 * total_episodes,
                                       "checkpoints": 36, "learning": 27, "fixed": 13},
            "worker_files_current_shape": worker_files,
            "executor_lifecycle_files_current_shape": 16,
            "outer_stdout_and_exit_receipts_additional": True,
            "conditional_tighter_bootstrap_forwards": 48,
            "conditional_tighter_audit_forwards": inference + 48,
            "conditional_tighter_total_module_forwards": forward_total - counters["bootstrap_forward"] - counters["audit_forward"] + 48 + inference + 48,
        },
        "time_planning_sensitivities_not_runtime_guarantees": {
            "incomplete_learning_internal_seconds": internal,
            "incomplete_learning_decisions": observed["decisions"],
            "complete_process_duration_of_original_learning_run": None,
            "same_blend_projection_seconds": same_blend,
            "active_to_same_blend_projection_ratio": (maximum_wall - cleanup) / same_blend,
            "different_workload_v4_seconds": prior["whole_process_seconds"],
            "different_workload_v4_decisions": prior["decisions"],
            "v4_whole_diagnostic_rate_projection_seconds": diagnostic_proxy,
            "active_margin_over_v4_projection_seconds": maximum_wall - cleanup - diagnostic_proxy,
            "projections_are_additive": False,
            "full_new_journal_and_neural_audit_measured": False,
            "completion_under_cap_guaranteed": False,
        },
        "serialization_review_inputs_not_byte_bounds": {
            "historical_reservation_records": reservations,
            "historical_reservation_bytes_per_file": reservation_bytes,
            "historical_mean_bytes_per_record": mean_bytes,
            "proposed_records_at_historical_mean_bytes_per_file": count_backstop * mean_bytes,
            "reservation_reader_limit_bytes": 512 * 1024 * 1024,
            "reservation_limit_mean_bytes_per_record_excluding_envelope": 512 * 1024 * 1024 / count_backstop,
            "ordinary_json_and_uncompressed_sample_reader_limit_bytes": 64 * 1024 * 1024,
            "raw_episode_reader_limit_bytes": 32 * 1024 * 1024,
            "journal_record_limit_bytes": 1024 * 1024,
            "aggregate_artifact_and_peak_resident_memory_bound_established": False,
        },
        "conditions": [
            "Only independently verified emergency_recovery_deadline may complete as a safe unsuccessful episode; all other failures stop the allocation.",
            "Each PPO episode must contain at least one confirmed retained decision; at most 256 starts/update and 6144 total. Zero progress aborts.",
            "One retirement attempt per learned episode is charged even if no neural forward or physical dispatch follows.",
            "Keep fixed physical windows, masks, reward, six-tick cadence and forty-decision episode maximum; the prefix proof depends on these.",
            "The 20000-decision/120000-tick aggregate margin authorizes no work above the unchanged stage caps.",
            "Bootstrap12288 and audit37888 are conservative backstops; tighter48/25296 estimates require final-source proof.",
            "Source/runtime/split/serialization/memory/storage/lifecycle admission remains pending, along with a later explicit owner commission.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(calculate(), indent=2, sort_keys=True))
