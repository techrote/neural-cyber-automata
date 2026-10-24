"""Read-only review replication over explicitly allowlisted historical JSON.

No checkpoint deserialization, corpus/archive/registry access or optional tensor
runtime import. Formulas independently reimplemented from review §4/§5.
"""
import hashlib
import json
from pathlib import Path
import statistics

from nca.evaluation.ik_training import nearest_rank
from nca.train.ik_dev import point_stage_rms, SEEDS


def reproduce_review(allowlist_path):
    allowlist = json.loads(Path(allowlist_path).read_text())["evidence_files"]
    authenticated = {}

    def read(path):
        entry = allowlist[str(path)]
        data = Path(path).read_bytes()
        observed = hashlib.sha256(data).hexdigest()
        if observed != entry["sha256"] or len(data) != entry["bytes"]:
            raise ValueError("allowlisted historical bytes changed")
        authenticated[str(path)] = observed
        return json.loads(data)

    result = {}
    selected_epochs = (37, 28, 40)
    for seed, epoch in zip(SEEDS, selected_epochs):
        events = sorted(p for p, v in allowlist.items() if v["role"] == "polar-v2 training ledger event"
                        and f"seed-{seed}" in p)
        receipts = []
        for path in events:
            event = read(path)
            record = event["record"]
            if record["kind"] != "complete":
                continue
            receipt = record["payload"].get("receipt", {})
            if receipt.get("schema") != "nca-ik-training-step/v1":
                continue
            receipts.append(receipt)
        if len(receipts) != 1720:
            raise ValueError("complete original optimizer receipt population required")
        rms = []
        for e in range(35, 40):
            rows = [dict(r, stage=0 if i < 34 else 1 if i == 34 else 2 if i < 42 else 3)
                    for i, r in enumerate(receipts[e*43:(e+1)*43])]
            rms.append(point_stage_rms(rows))
        paths = [p for p, v in allowlist.items() if v["role"] == "polar-v2 tuning report"
                 and f"seed-{seed}" in p and p.endswith(f"tuning-epoch-{epoch:04d}.json")]
        if len(paths) != 1:
            raise ValueError("one allowlisted historical selected report required")
        report = read(paths[0])
        errors = [r["raw_error_cells"] for r in report["rows"] if r["kind"] != "stress"]
        p95 = nearest_rank(errors, .95)
        if abs(p95 - report["nominal"]["p95_error_cells"]) > 1e-12:
            raise ValueError("historical scorer mismatch")
        result[str(seed)] = {"selected_epoch": epoch, "p95": p95,
                             "point_rms_epochs36_40": rms, "point_rms_mean": statistics.fmean(rms),
                             "last_point_receipts": [[r["terms"]["endpoint"] for r in receipts[e*43+29:e*43+34]]
                                                     for e in range(35, 40)]}
    return {"schema": "nca-ik-review-replication/v1", "results": result,
            "authenticated_inputs": authenticated,
            "provenance": "independent arithmetic implementation; no third-party source copied"}

# Copyright 2026 techrote. All rights reserved.
