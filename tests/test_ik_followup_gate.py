from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from nca.contracts import canonical_json, strict_json_loads
from nca.train.ik_followup_gate import open_after_tuning, tuning_stop_gate

def reports():
    report={"schema":"nca-ik-direct-score/v1","partition":"tuning",
            "comparator":"trained_raw_reference","scheduled_calls":2020,
            "received_samples":2020,"received_predictions":2020,"complete":True,
            "rows":[{"task_id":str(i),"nonfinite_output":False,"hard_bound_violation":False,
                     "command_bound_violation":False,"unexpected_acceptance":False,
                     "protocol_violations":[]} for i in range(2020)],
            "nominal":{"scheduled":1880,"complete_raw_outputs":True,"p95_error_cells":.01,
                       "worst_error_cells":.02,"feasible_success_fraction":1.},
            "continuity":{"complete":True,"over_0_25_radian_pairs":0,"admitted_branch_switches":0}}
    return {s:deepcopy(report) for s in (41011,42013,43019)}

class FollowupGateTests(unittest.TestCase):
    def test_passing_gate_round_trips_through_canonical_publication(self):
        gate = tuning_stop_gate(reports())
        published = strict_json_loads(canonical_json(gate))
        self.assertEqual(published["per_seed"], {
            "41011": "pass", "42013": "pass", "43019": "pass"})
        self.assertTrue(published["final_opening_allowed"])

    def test_no_go_finish_publishes_gate_before_closing_without_final_access(self):
        from nca.train.ik_followup import finish_experiment
        scores = reports()
        scores[42013]["nominal"]["feasible_success_fraction"] = .94
        selections = {seed: {"step": 1720, "checkpoint_sha256": format(seed, "064x")}
                      for seed in scores}
        attempts = {seed: Mock() for seed in scores}
        with tempfile.TemporaryDirectory() as directory, \
                patch("nca.train.ik_followup.select_training",
                      return_value=(selections, scores, {})), \
                patch("nca.train.ik_followup.open_final") as opener, \
                patch("nca.train.ik_followup.cuda_context") as cuda:
            gate = finish_experiment(directory, {"sha256": "a" * 64}, None,
                                     "unused", "unused", {}, attempts)
            published = strict_json_loads(Path(directory, "tuning-stop.json").read_bytes())
            self.assertFalse(gate["final_opening_allowed"])
            self.assertEqual(published["per_seed"], {
                "41011": "pass", "42013": "no_go", "43019": "pass"})
            self.assertEqual(published["declaration_sha256"], "a" * 64)
            self.assertEqual(set(published["selections"]), {"41011", "42013", "43019"})
            for attempt in attempts.values():
                attempt.finish.assert_called_once_with(
                    "completed", reason="bounded_followup_tuning_no_go_final_sealed",
                    publication_reserve_seconds=60.)
            opener.assert_not_called()
            cuda.assert_not_called()

    def test_one_failed_seed_prevents_any_open(self):
        for key,value in (("p95_error_cells",.060001),("worst_error_cells",.180001),
                          ("feasible_success_fraction",.94999)):
            r=reports();r[42013]['nominal'][key]=value;open_=Mock()
            if key=='p95_error_cells':r[42013]['nominal']['worst_error_cells']=value
            with self.assertRaisesRegex(ValueError,'must remain sealed'):open_after_tuning(r,open_)
            open_.assert_not_called()

    def test_missing_seed_or_incomplete_report_prevents_open(self):
        r=reports();del r[43019];open_=Mock()
        with self.assertRaises(ValueError):open_after_tuning(r,open_)
        r=reports();r[41011]['rows'].pop()
        with self.assertRaises(ValueError):open_after_tuning(r,open_)
        open_.assert_not_called()

    def test_nonfinite_negative_boolean_or_inconsistent_summary_never_opens(self):
        for key in ("p95_error_cells", "worst_error_cells", "feasible_success_fraction"):
            for value in (float("nan"), float("inf"), -float("inf"), -1., True, None):
                with self.subTest(key=key,value=value):
                    r=reports();r[41011]["nominal"][key]=value;opener=Mock()
                    with self.assertRaises(ValueError):open_after_tuning(r,opener)
                    opener.assert_not_called()
        for key,value in (("p95_error_cells",.03),("feasible_success_fraction",1.01)):
            r=reports();r[41011]["nominal"][key]=value;opener=Mock()
            with self.assertRaises(ValueError):open_after_tuning(r,opener)
            opener.assert_not_called()
        for value in (False,1,None):
            r=reports();r[41011]["complete"]=value;opener=Mock()
            with self.assertRaises(ValueError):open_after_tuning(r,opener)
            opener.assert_not_called()

    def test_all_original_gates_pass_then_open_once(self):
        open_=Mock(return_value='opened');self.assertEqual(open_after_tuning(reports(),open_),'opened')
        open_.assert_called_once_with()

# Copyright 2026 techrote. All rights reserved.
