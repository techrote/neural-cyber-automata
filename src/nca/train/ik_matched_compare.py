"""Authenticated checkpoint/state comparison; no inference or optimizer work."""
import hashlib
from pathlib import Path
from nca.contracts import canonical_json, sha256_json
from nca.train.ik_experiment import _read, SEEDS
from nca.runs import sha256_file


def authenticated_reports(run,declaration,seed):
    from nca.train.ik_matched_supervisor import ledger
    bound=ledger(Path(run),declaration,seed);status=bound.inspect()
    latest=status['latest_checkpoint']
    if not latest: raise ValueError('authenticated final checkpoint required')
    history=latest['state']['extra_state']['tuning_history']
    if len(history)!=40: raise ValueError('all forty authenticated tuning records required')
    reports=[]
    for epoch,item in enumerate(history,1):
        name=f'tuning-epoch-{epoch:04d}.json'
        if item['epoch']!=epoch or item['step']!=43*epoch or item['file']!=name:
            raise ValueError('declared tuning cadence changed')
        path=bound.root/name
        report=_read(path,maximum=16*1024**2)
        if sha256_file(path)!=item['file_sha256'] or sha256_json(report)!=item['report_sha256']:
            raise ValueError('tuning report differs from durable checkpoint history')
        reports.append(report)
    return reports


def state_hash(value):
    def encode(v):
        if hasattr(v,'detach'):
            t=v.detach().cpu().contiguous()
            return ['tensor',str(t.dtype),list(t.shape),hashlib.sha256(
                t.reshape(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()]
        if isinstance(v,dict):
            return ['dict',[[encode(k),encode(v[k])] for k in sorted(v,key=lambda k:(type(k).__name__,repr(k)))]]
        if isinstance(v,(tuple,list)): return [type(v).__name__,[encode(x) for x in v]]
        if v is None or type(v) in (str,int,float,bool): return [type(v).__name__,v]
        raise ValueError('unsupported checkpoint state type')
    return sha256_json(encode(value))


def evidence(run,declaration):
    from nca.ik.checkpoint import load_checkpoint
    from nca.train.ik_matched_supervisor import ledger
    result={}
    for seed in SEEDS:
        bound=ledger(Path(run),declaration,seed); status=bound.inspect()
        if (status['active'] or status['unresolved_work'] or status['attempt_count']!=1
                or status['charged']['optimizer_steps']!=1720
                or status['charged']['sample_presentations']!=210400
                or status['charged']['tuning_calls']!=80800
                or status['charged']['evaluation_calls']!=0):
            raise ValueError('complete reconciled fixed workload required')
        paths=sorted(bound.root.glob('attempts/attempt-*/checkpoints/*.pt'))
        if len(paths)!=41: raise ValueError('all forty-one checkpoints required')
        checkpoints=[]
        for path in paths:
            marker=_read(path.with_name(path.name+'.json'))['checkpoint']
            payload=load_checkpoint(path,expected_sha256=marker['sha256'],expected_metadata=bound.identity)
            state=payload['training_state']
            checkpoints.append(state_hash({k:payload[k] for k in ('model','optimizer','rng')} |
                                          {'sampler':state['sampler'],'trainer':state['trainer']}))
        reports=authenticated_reports(run,declaration,seed)
        events=[]
        for path in sorted(bound.root.glob('attempts/attempt-*/event-*.json')):
            record=_read(path)['record']
            if record['kind']=='complete' and 'ordered_sample_ids_sha256' in record['payload']['receipt']:
                events.append(record['payload']['receipt'])
        # The durable ledger has already authenticated all events; compare only
        # operation identities/sample order, excluding timings and declaration IDs.
        if len(events)!=1720: raise ValueError('every ordered batch receipt required')
        ordered=[{k:v for k,v in r.items() if k in ('ordered_sample_ids_sha256','rows','attempted_steps','committed_steps','sample_presentations','teacher_status_counts','pose_rows','endpoint_rows','continuity_pairs')} for r in events]
        result[str(seed)]={'checkpoints':checkpoints,'tuning':state_hash(reports),
                           'sampler_schedule':declaration['schedules'][str(seed)],
                           'ordered_receipts':state_hash(ordered)}
    return result


def compare(serial,parallel):
    return dict(schema='nca-ik-matched-tensor-comparison/v1',
                exact_match=serial==parallel,serial_sha256=sha256_json(serial),
                parallel_sha256=sha256_json(parallel),qualified_for_consistent_parallel=serial==parallel)

# Copyright 2026 techrote. All rights reserved.
