"""Versioned IK development contracts; no final access and no implicit execution.

Receipt RMS and binned bias are independently implemented from the equations
described by the 2026-10-23 independent review, not copied third-party code.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import statistics

from nca.contracts import sha256_json
from nca.train.ik_corpus import PairPreservingSampler

SEEDS = (41011, 42013, 43019)
LENGTH = 6.550781059358212


@dataclass(frozen=True)
class DevConfig:
    head: str = "tanh"
    lr_schedule: str = "constant"
    epochs: int = 40
    batch_order: str = "stage"
    hidden_width: int = 64

    def __post_init__(self):
        if (self.head not in ("tanh", "interior_linear")
                or self.lr_schedule not in ("constant", "cosine", "constant_3e-4")
                or type(self.epochs) is not int or not 1 <= self.epochs <= 1000
                or self.batch_order not in ("stage", "shuffled")
                or self.hidden_width not in (64, 128)):
            raise ValueError("invalid development recipe")

    @property
    def id(self):
        return f"{self.head}-{self.lr_schedule}-e{self.epochs}-{self.batch_order}-w{self.hidden_width}"

    def to_dict(self):
        return asdict(self)

    def lr(self, step, total):
        if not 0 <= step < total:
            raise ValueError("step outside declared run")
        if self.lr_schedule == "constant_3e-4":
            return .0003
        if self.lr_schedule == "constant":
            return .001
        return .00001 + .5 * (.001 - .00001) * (1 + math.cos(math.pi * step / max(1, total - 1)))

    def evaluation_epochs(self):
        cadence = 1 if self.epochs == 40 else 5
        return tuple(e for e in range(1, self.epochs + 1) if e % cadence == 0 or e == self.epochs)


def grid():
    core = [DevConfig(h, lr, e) for h in ("tanh", "interior_linear")
            for lr in ("constant", "cosine") for e in (40, 200, 1000)]
    return core + [DevConfig("interior_linear", "cosine", 200, "shuffled"),
                   DevConfig("interior_linear", "cosine", 200, hidden_width=128),
                   DevConfig("interior_linear", "constant_3e-4", 200)]


SELECTION_RULE = {
    "schema": "nca-ik-dev-selection/v1",
    "primary": "median_across_all_three_seeds_of_best_complete_dev_nominal_p95",
    "checkpoint": "lowest_complete_dev_p95_then_earliest_epoch",
    "ties": ["maximum_selected_dev_worst_across_seeds", "fewer_epochs",
             "width64_before128", "stage_before_shuffled", "tanh_before_interior_linear",
             "constant_before_cosine_before_constant_3e-4", "config_id"],
    "shortlist": 3, "missing_seed": "ineligible_report_failure_no_substitution",
    "tuning": "one_call_per_shortlisted_seed_checkpoint_after_dev_ranking_only",
    "final_access": False, "evidence": "development evidence; no acceptance claim",
}


def split(samples, *, held_out=10):
    samples = tuple(samples)
    if not samples or any(s.task.partition != "training" for s in samples):
        raise ValueError("training families only")
    families = sorted({s.task.family_id for s in samples},
                      key=lambda f: (sha256_json(["ik54-dev-split/v1", f]), f))
    if not 0 < held_out < len(families):
        raise ValueError("nonempty train and dev required")
    dev = frozenset(families[:held_out])
    train_rows = tuple(s for s in samples if s.task.family_id not in dev)
    dev_rows = tuple(s for s in samples if s.task.family_id in dev)
    record = {"schema": "nca-ik-dev-split/v1", "family_order": families,
              "dev_families": sorted(dev), "train_families": sorted(set(families) - dev),
              "train_sample_hashes": [s.sha256 for s in train_rows],
              "dev_sample_hashes": [s.sha256 for s in dev_rows],
              "normalizer": "unchanged_original_training_only_fit",
              "limitation": "dev shares training hip lanes; within-lane accuracy only"}
    return train_rows, dev_rows, record


def sampler(samples, seed, config):
    return PairPreservingSampler(samples, seed, epochs=config.epochs, batch_size=128)


def epoch_batches(plan, config, epoch):
    start = sum(plan.batches_per_epoch[:epoch])
    batches = plan._batches[start:start + plan.batches_per_epoch[epoch]]
    if config.batch_order == "shuffled":
        batches = tuple(b for i, b in sorted(enumerate(batches), key=lambda ib: (
            sha256_json(["ik-complete-batch-shuffle/v1", plan.seed, epoch, ib[0],
                         [plan.samples[j].task.id for j in ib[1]]]), ib[0])))
    return batches


def point_stage_rms(receipts, *, last=5):
    values = [r["terms"]["endpoint"] for r in receipts if r["stage"] == 0][-last:]
    return math.sqrt(statistics.fmean(values)) * LENGTH if values else None


def bias_metrics(samples, predictions):
    bins = defaultdict(list)
    joint_errors = ([], [])
    for s in samples:
        if s.task.state.kind == "stress" or not s.label.pose_loss:
            continue
        q = predictions[s.task.id].raw_reference
        if q is None or not all(math.isfinite(x) for x in q):
            return {"hip_p95": None, "ankle_p95": None, "hip_max_abs_bin_mean": None, "hip_eta2": None}
        target = s.label.teacher.configuration
        err = [a-b for a, b in zip(q, target)]
        for j in range(2):
            joint_errors[j].append(abs(err[j]))
        bins[min(11, max(0, int((target[0] + .6) / .1)))].append(err[0])
    from nca.evaluation.ik_training import nearest_rank
    values = [v for group in bins.values() for v in group]
    if not values:
        raise ValueError("dev nominal population is empty")
    mean = statistics.fmean(values)
    total = math.fsum((x - mean)**2 for x in values)
    means = {b: statistics.fmean(v) for b, v in bins.items()}
    between = math.fsum(len(bins[b]) * (means[b] - mean)**2 for b in bins)
    return {"hip_p95": nearest_rank(joint_errors[0], .95),
            "ankle_p95": nearest_rank(joint_errors[1], .95),
            "hip_max_abs_bin_mean": max(abs(v) for v in means.values()),
            "hip_eta2": between / total if total else 0.,
            "hip_bin_means": {str(k): v for k, v in sorted(means.items())}}


def batched_predictions(context, model, samples, *, batch_size=1024):
    from nca.evaluation.ik_training import Prediction, _refusal
    from nca.ik.data import runtime_features
    torch = context.torch
    result, admitted = {}, []
    for s in samples:
        refusal = _refusal(s.task)
        if refusal:
            result[s.task.id] = Prediction(None, refusal)
        else:
            admitted.append(s)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(admitted), batch_size):
            rows = admitted[start:start + batch_size]
            features = torch.tensor([runtime_features(s.task.inputs) for s in rows],
                                    dtype=torch.float32, device=context.device)
            output = model(features)
            q = output.reference.cpu().tolist()
            codes = output.invalid_code.cpu().tolist()
            for s, reference, code in zip(rows, q, codes):
                result[s.task.id] = (Prediction(tuple(reference)) if code == 0 else
                                    Prediction(None, "nonfinite_output" if code & 32 else "refused"))
    return result


def parity(context, model, samples, *, predictions=None):
    from nca.evaluation.ik_training import collect_predictions, model_predictor
    batch = predictions or batched_predictions(context, model, samples)
    single = collect_predictions(samples, model_predictor(context, model))
    maximum = 0.
    for s in samples:
        a, b = batch[s.task.id], single[s.task.id]
        if a.status != b.status or (a.raw_reference is None) != (b.raw_reference is None):
            raise ValueError("batched refusal/status parity failed")
        if a.raw_reference is not None:
            maximum = max(maximum, *(abs(x-y) for x, y in zip(a.raw_reference, b.raw_reference)))
    if maximum > 2e-5:
        raise ValueError("batched joint parity exceeds 2e-5 rad")
    return {"rows": len(samples), "max_joint_difference_radians": maximum, "device": str(context.device)}


def dev_score(samples, predictions):
    """Use existing binary64 row/population/continuity semantics on frozen dev IDs."""
    from nca.evaluation.ik_training import _score, _population, _continuity
    from nca.ik.kinematics import chain_from_profile
    from nca.ik.profiles import runner_right_sole_profile
    chain = chain_from_profile(runner_right_sole_profile())
    rows = [_score(s.task, predictions.get(s.task.id), chain, "trained_raw_reference",
                   sample_missing=False) for s in samples]
    nominal = _population([r for r in rows if r["kind"] != "stress"])
    complete = (set(predictions) == {s.task.id for s in samples}
                and nominal["complete_raw_outputs"]
                and all(not (r["missing_prediction"] or r["malformed_output"] or r["nonfinite_output"]
                             or r["protocol_violations"] or r["unexpected_acceptance"]
                             or r["status"] == "prediction_exception") for r in rows)
                and all(r["expected_refusal_observed"] for r in rows if r["expected_refusal"] is not None))
    return {"nominal": nominal, "complete": complete,
            "continuity": _continuity([s.task for s in samples], rows),
            "bias": bias_metrics(samples, predictions)}


def rank_configs(results):
    ranked = []
    for config in grid():
        runs = [r for r in results if r["config"] == config.to_dict() and r.get("status") == "complete"]
        if sorted(r["seed"] for r in runs) != sorted(SEEDS):
            continue
        primary = statistics.median(r["selected"]["nominal"]["p95_error_cells"] for r in runs)
        worst = max(r["selected"]["nominal"]["worst_error_cells"] for r in runs)
        key = (primary, worst, config.epochs, config.hidden_width,
               config.batch_order != "stage", config.head != "tanh",
               ("constant", "cosine", "constant_3e-4").index(config.lr_schedule), config.id)
        ranked.append((key, config, runs))
    return sorted(ranked, key=lambda x: x[0])

# Copyright 2026 techrote. All rights reserved.
