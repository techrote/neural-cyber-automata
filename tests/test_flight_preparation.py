import unittest
from copy import deepcopy
from nca.adapters.flight import FlightSession, velocity_action, finite_vector
from nca.train.flight import training_admission, export_preprocessing


class Owner:
    def __init__(self):
        self.tick = 0
        self.run = 0
        self.bad = None
        self.closed = False

    def request(self, command):
        if command['command'] == 'reset':
            self.run += 1
            self.tick = 0
        else:
            self.tick += 6
        result = {'sequence': command['sequence'], 'observation': {
            'version': '4ly-observation/v1', 'action_version': '4ly-action/v1',
            'run_id': str(self.run), 'actor_id': 'Trainer', 'epoch': self.run,
            'observation_id': f'{self.run}-{self.tick}', 'observed_tick': self.tick,
            'control_generation': 0, 'flat': [0.] * 54},
            'transition': dict(consumed_ticks=6, terminated=True, truncated=False, failed=False, success=True)}
        if self.bad:
            self.bad(result)
        return result

    def close(self):
        self.closed = True


class FlightTests(unittest.TestCase):
    def test_actual_terminal_and_copied_observation(self):
        owner = Owner(); session = FlightSession(owner)
        obs = session.reset(scenario='arrival_braking', seed=1)
        obs['flat'][0] = 999
        self.assertEqual(session.observation['flat'][0], 0)
        terminal = session.step([1, 1], observation_id=obs['observation_id'])
        self.assertEqual(terminal['observation']['observed_tick'], 6)
        with self.assertRaises(ValueError): session.step([0, 0], observation_id='1-6')
        session.close(); self.assertTrue(owner.closed)

    def test_final_reset_and_invalid_actions_never_reach_owner(self):
        owner = Owner(); session = FlightSession(owner)
        for split in ('final', 'sealed', 'original_opened_final', 'polar_sealed_final'):
            with self.assertRaises(ValueError): session.reset(scenario='x', seed=1, split=split)
        self.assertEqual(owner.run, 0)
        for action in ([True, 0], [float('nan'), 0], [2, 0], [0]):
            with self.assertRaises(ValueError): velocity_action(action)

    def test_bad_replies_poison_session(self):
        changes = [lambda r:r.update(sequence=0),
            lambda r:r['observation'].update(version='legacy'),
            lambda r:r['observation'].update(flat=[0]*53),
            lambda r:r['observation'].update(flat=[float('inf')]*54)]
        for change in changes:
            owner=Owner(); owner.bad=change; session=FlightSession(owner)
            with self.assertRaises(ValueError): session.reset(scenario='x', seed=1)
            self.assertTrue(session.failed)

    def test_stale_authority_and_tick_discontinuity(self):
        for key,value in [('epoch',99),('observed_tick',8),('control_generation',1),('observation_id','1-0')]:
            owner=Owner(); session=FlightSession(owner); obs=session.reset(scenario='x',seed=1)
            owner.bad=lambda r:r['observation'].update({key:value})
            with self.assertRaises(ValueError):session.step([0,0],observation_id=obs['observation_id'])
            self.assertTrue(session.failed)

    def test_export_and_admission(self):
        self.assertFalse(training_admission()['admitted'])
        flat, action=export_preprocessing([0]*54,[1,1])
        self.assertAlmostEqual(sum(x*x for x in action['move']),1)
        self.assertEqual(len(flat),54)

    def test_consumed_reset_missing_observation_poisoned(self):
        owner=Owner(); owner.bad=lambda r:r.pop('observation'); session=FlightSession(owner)
        with self.assertRaises(KeyError): session.reset(scenario='x',seed=1)
        self.assertTrue(session.failed)
        with self.assertRaises(ValueError): session.reset(scenario='x',seed=1)
        self.assertEqual(owner.run,1)

    def test_truncation_preserves_final_observation(self):
        owner=Owner(); session=FlightSession(owner); obs=session.reset(scenario='x',seed=1)
        owner.bad=lambda r:r['transition'].update(terminated=False,truncated=True,success=False)
        reply=session.step([0,0],observation_id=obs['observation_id'])
        self.assertTrue(reply['transition']['truncated'])
        self.assertEqual(reply['observation']['observed_tick'],6)
        self.assertTrue(session.terminal)

    def test_short_nonterminal_interval_refused(self):
        owner=Owner(); session=FlightSession(owner); obs=session.reset(scenario='x',seed=1)
        def short(reply):
            reply['transition'].update(consumed_ticks=1,terminated=False,success=False)
            reply['observation']['observed_tick']=1
        owner.bad=short
        with self.assertRaises(ValueError):session.step([0,0],observation_id=obs['observation_id'])
        self.assertTrue(session.failed)


if __name__ == '__main__': unittest.main()
