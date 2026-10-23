"""4LY-004 preparation protocol. No host launch or experiment execution API."""
from copy import deepcopy
import math
from nca.registry import registry


def finite_vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError('wrong vector shape')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError('finite numeric vector required')
    return tuple(float(x) for x in value)


def velocity_action(value):
    x, y = finite_vector(value, 2)
    if max(abs(x), abs(y)) > 1:
        raise ValueError('normalized velocity outputs required')
    scale = max(1., math.hypot(x, y))
    return {'move': [x / scale, y / scale]}


class FlightSession:
    """Injected persistent owner interface; a fixture is never actual-host evidence.

    Transport must implement request(dict)->dict and close(). Until accepted
    host identities and a separately tested bridge exist, this is conformance
    preparation, not an admitted environment.
    """
    def __init__(self, transport):
        self.transport = transport
        self.sequence = 0
        self.observation = None
        self.closed = False
        self.failed = False
        self.terminal = True

    def _request(self, command):
        if self.closed or self.failed:
            raise ValueError('closed or failed session')
        self.sequence += 1
        try:
            reply = self.transport.request(dict(command, sequence=self.sequence))
            if type(reply) is not dict or type(reply.get('sequence')) is not int or reply['sequence'] != self.sequence:
                raise ValueError('dropped, duplicate or reordered reply')
            return deepcopy(reply)
        except BaseException:
            self.failed = True
            self.observation = None
            raise

    def reset(self, *, scenario, seed, split='training'):
        if split not in ('training', 'tuning'):
            raise ValueError('all final suites excluded')
        if type(seed) is not int or not isinstance(scenario, str) or not scenario:
            raise ValueError('explicit scenario and seed required')
        previous = self.observation
        self.observation = None
        reply = self._request(dict(command='reset', profile='Trainer', scenario=scenario, seed=seed, split=split))
        try:
            self._observe(reply['observation'], fresh=True)
            if previous and (self.observation['run_id'], self.observation['epoch']) == (previous['run_id'], previous['epoch']):
                raise ValueError('reset did not revoke previous authority')
            if self.observation['observed_tick'] != 0:
                raise ValueError('reset must begin at tick zero')
        except BaseException:
            self.failed = True
            self.observation = None
            raise
        self.terminal = False
        return deepcopy(self.observation)

    def _observe(self, observation, *, fresh=False):
        try:
            if observation.get('version') != '4ly-observation/v1' or observation.get('action_version') != '4ly-action/v1':
                raise ValueError('incompatible flight schema')
            for key in ('run_id', 'actor_id', 'observation_id'):
                if not isinstance(observation.get(key), str) or not observation[key]:
                    raise ValueError('missing owner identity')
            for key in ('epoch', 'observed_tick', 'control_generation'):
                if type(observation.get(key)) is not int or observation[key] < 0:
                    raise ValueError('invalid owner counter')
            if not fresh and self.observation:
                old = self.observation
                if any(observation[k] != old[k] for k in ('run_id', 'actor_id', 'epoch', 'control_generation')):
                    raise ValueError('owner authority changed')
                if observation['observed_tick'] <= old['observed_tick'] or observation['observation_id'] == old['observation_id']:
                    raise ValueError('stale observation')
            finite_vector(observation['flat'], 54)
            self.observation = deepcopy(observation)
        except BaseException:
            self.failed = True
            self.observation = None
            raise

    def step(self, outputs, *, observation_id):
        if self.terminal or self.observation is None or observation_id != self.observation['observation_id']:
            raise ValueError('live matching observation required')
        old_tick = self.observation['observed_tick']
        if old_tick % 6:
            raise ValueError('decision boundary required')
        reply = self._request(dict(command='step', observation_id=observation_id,
            run_id=self.observation['run_id'], epoch=self.observation['epoch'],
            action=velocity_action(outputs), repeat_ticks=6))
        try:
            transition = reply['transition']
            for key in ('terminated', 'truncated', 'failed', 'success'):
                if type(transition.get(key)) is not bool:
                    raise ValueError('explicit terminal classification required')
            ticks = transition['consumed_ticks']
            if type(ticks) is not int or not 1 <= ticks <= 6:
                raise ValueError('invalid confirmed tick count')
            self._observe(reply['observation'])
            if self.observation['observed_tick'] != old_tick + ticks:
                raise ValueError('physical tick discontinuity')
            self.terminal = any(transition[k] for k in ('terminated', 'truncated', 'failed'))
            if transition['success'] and not transition['terminated']:
                raise ValueError('success without terminal')
            return deepcopy(reply)
        except BaseException:
            self.failed = True
            self.observation = None
            raise

    def close(self):
        self.closed = True
        self.observation = None
        self.transport.close()


registry.register('adapter', '4ly-preparation', 1, FlightSession,
                  capabilities=('flight_protocol_fixture', 'no_execution_admission'))
