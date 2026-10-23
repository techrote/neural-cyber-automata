from copy import deepcopy
import unittest
from unittest.mock import Mock
from nca.train.ik_followup_gate import open_after_tuning

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
