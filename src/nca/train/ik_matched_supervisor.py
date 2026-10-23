"""Fail-closed matched trial supervisor. Admission checking never launches work."""
from __future__ import annotations
import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from nca.contracts import canonical_json, sha256_json
from nca.runs import immutable_json, sha256_file
from nca.train.ik_matched_prepare import (GIB, TITAN_FINGERPRINT, allocation,
    candidate_config, identity_for, inspect, prepared_source)
from nca.train.ik_experiment import _read, _titan_visibility, SEEDS
from nca.train.ik_run_state import IKRunLedger


def board_sample():
    """Targeted NVML metadata only, before any possible CUDA initialization."""
    _titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    result = subprocess.run(['nvidia-smi', '-i', os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
    fields = result.stdout.strip().split(',')
    if len(fields) != 4:
        raise ValueError('missing board telemetry')
    values = [float(x.strip()) for x in fields]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('invalid board telemetry')
    used, free, utilization, temperature = values
    if utilization > 100 or used * 1024**2 > 20 * GIB:
        raise ValueError('aggregate board memory envelope exceeded')
    return dict(monotonic=time.monotonic(), used_bytes=int(used*1024**2),
                free_bytes=int(free*1024**2), utilization=utilization, temperature=temperature)


def admission(packet, reviews):
    """Require two separately authored prospective records, never manufacture one."""
    source = prepared_source()
    root = Path(packet)
    value = _read(root/'packet.json')
    if value.get('sha256') != sha256_json({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('packet identity changed')
    if value['source_sha256'] != source['source_sha256']:
        raise ValueError('packet source is stale')
    reviewers = set()
    for path in reviews:
        review = _read(path)
        if (review.get('schema') != 'nca-ik-matched-admission/v1'
                or review.get('head') != source['revision']
                or review.get('packet_sha256') != value['sha256']
                or review.get('source_sha256') != source['source_sha256']
                or review.get('modes') != ['serial','parallel']
                or review.get('gpu_visibility_sha256') != TITAN_FINGERPRINT
                or review.get('verdict') != 'admit_both_matched_trials'
                or review.get('independent_non_author') is not True
                or canonical_json(review.get('checks')) != canonical_json({k:True for k in (
                    'exact_head_ci','corpus_and_final_exclusion','accounting_and_stop',
                    'supervisor_and_watchdog','aggregate_resource_feasibility')})):
            raise ValueError('fresh independent TITAN admission required')
        reviewer = review.get('reviewer')
        if type(reviewer) is not str or not reviewer or reviewer in reviewers:
            raise ValueError('two distinct independent reviewers required')
        reviewers.add(reviewer)
    if len(reviewers) != 2:
        raise ValueError('exactly two admissions required')
    for mode in ('serial','parallel'):
        path=root/mode/'declaration.json'
        declaration=_read(path)
        if (sha256_file(path)!=value['declaration_file_sha256'][mode]
                or declaration['source'] != source
                or declaration['candidate']!=candidate_config()
                or declaration['allocations']!={str(s):allocation() for s in SEEDS}):
            raise ValueError('exact committed declaration source differs')
    return value


def stop_tree(process):
    if process.poll() is None:
        if os.name == 'nt':
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                           capture_output=True, check=False, timeout=10)
        else:
            process.kill()
        process.wait(timeout=10)


def ledger(root, declaration, seed):
    return IKRunLedger.open(root/f'seed-{seed}', expected_identity=identity_for(declaration,seed),
                           expected_allocation=allocation())


def validate_authorization(auth, packet_sha256):
    if canonical_json(auth) != canonical_json(dict(schema='nca-ik-matched-owner-execution/v1',
                    packet_sha256=packet_sha256,modes=['serial','parallel'],execute=True)):
        raise ValueError('owner execution instruction required')


def process_metrics():
    import ctypes
    from ctypes import wintypes
    if os.name!='nt': raise ValueError('qualified Windows process telemetry required')
    class IO(ctypes.Structure):
        _fields_=[(k,ctypes.c_ulonglong) for k in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
    class Memory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(k,ctypes.c_size_t) for k in
            ('peak_working_set','working_set','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile','private')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    handle=kernel.GetCurrentProcess(); io=IO(); memory=Memory(); memory.cb=ctypes.sizeof(memory)
    kernel.GetProcessIoCounters.argtypes=[wintypes.HANDLE,ctypes.POINTER(IO)]
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
    if not kernel.GetProcessIoCounters(handle,ctypes.byref(io)) or not psapi.GetProcessMemoryInfo(handle,ctypes.byref(memory),memory.cb):
        raise ValueError('missing process resource telemetry')
    times=os.times()
    return dict(cpu_user_seconds=times.user,cpu_system_seconds=times.system,
                peak_resident_bytes=memory.peak_working_set,read_bytes=io.read_bytes,write_bytes=io.write_bytes)


def supervise(packet, mode, corpus, registry, reviews, authorization):
    began = time.monotonic()
    value = admission(packet, reviews)
    root = Path(packet)
    auth = _read(authorization)
    validate_authorization(auth,value['sha256'])
    # This single persistent claim excludes BOTH modes and is never auto-cleared
    # following failure, interruption or unknown work. Successful mode receipts
    # allow only the other untouched mode through a new explicit invocation.
    board_sample()
    claim = uuid.uuid4().hex
    lock = root/'group-active.json'
    immutable_json(lock, dict(claim=claim,mode=mode,packet_sha256=value['sha256'],
        reviews=[dict(path=str(Path(p).resolve()),sha256=sha256_file(p)) for p in reviews],
        authorization=str(Path(authorization).resolve()),authorization_sha256=sha256_file(authorization)))
    run = root/mode
    declaration = _read(run/'declaration.json')
    pending, active, completed = list(SEEDS), {}, {}
    telemetry = run/'board-telemetry.jsonl'
    try:
        with telemetry.open('x',encoding='utf-8') as monitor:
            while pending or active:
                sample = board_sample()
                monitor.write(canonical_json(sample)+'\n'); monitor.flush(); os.fsync(monitor.fileno())
                while pending and len(active) < declaration['topology']['max_workers']:
                    seed = pending.pop(0)
                    state = ledger(run,declaration,seed).inspect()
                    if state['attempt_count'] or state['active'] or state['unresolved_work']:
                        raise ValueError('no automatic retry or resume')
                    immutable_json(run/f'seed-{seed}'/'group-start.json',
                                   dict(claim=claim,started=began,packet_sha256=value['sha256']))
                    command=[sys.executable,'-m','nca.train.ik_matched_supervisor','_worker','--packet',str(root),
                             '--mode',mode,'--seed',str(seed),'--corpus',str(corpus),
                             '--registry',str(registry),'--claim',claim,'--started',repr(began)]
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                    log=(run/f'seed-{seed}'/'worker.log').open('xb')
                    try: process=subprocess.Popen(command,stdout=log,stderr=log,env=env)
                    finally: log.close()
                    active[seed]=process
                if time.monotonic()-began > 7190:
                    raise TimeoutError('whole group wall bound exceeded')
                for seed,process in list(active.items()):
                    if process.poll() is None: continue
                    state=ledger(run,declaration,seed).inspect()
                    result=_read(run/f'seed-{seed}'/'worker-result.json') if process.returncode==0 else {}
                    if (process.returncode!=0 or state['active'] or state['unresolved_work']
                            or not result.get('training_complete')
                            or time.monotonic()-began > state['charged_wall_seconds']):
                        raise ValueError('failed or unaccounted worker; group stopped')
                    completed[str(seed)]=result
                    del active[seed]
                if active: time.sleep(.25)
        from nca.train.ik_followup_gate import tuning_stop_gate
        reports={}
        for seed in SEEDS:
            saved=[_read(run/f'seed-{seed}'/f'tuning-epoch-{epoch:04d}.json',maximum=16*1024**2) for epoch in range(1,41)]
            reports[seed]=min(enumerate(saved),key=lambda pair:(pair[1]['selection_metric'],pair[0]))[1]
        gate=tuning_stop_gate(reports)
        immutable_json(run/'group-result.json',dict(claim=claim,
            makespan_seconds=time.monotonic()-began,workers=completed,tuning_gate=gate,
            verified=True,final_opened=False))
        if any(time.monotonic()-began > ledger(run,declaration,s).inspect()['charged_wall_seconds'] for s in SEEDS):
            raise ValueError('publication exceeded charged reserve')
        lock.unlink()
    except BaseException:
        cleanup_errors=[]
        for process in active.values():
            try: stop_tree(process)
            except BaseException as exc: cleanup_errors.append(type(exc).__name__)
        immutable_json(root/f'group-failure-{claim}.json',dict(claim=claim,mode=mode,
            unknown_work_blocks_reuse=True,cleanup_errors=cleanup_errors,elapsed=time.monotonic()-began))
        raise


def worker(args):
    root=Path(args.packet); run=root/args.mode
    claim=_read(root/'group-active.json')
    if claim.get('claim')!=args.claim or claim.get('mode')!=args.mode:
        raise ValueError('live group ownership required before CUDA')
    for record in claim['reviews']:
        if sha256_file(record['path'])!=record['sha256']: raise ValueError('review changed')
    admission(root,[r['path'] for r in claim['reviews']])
    if sha256_file(claim['authorization'])!=claim['authorization_sha256']:
        raise ValueError('authorization changed')
    validate_authorization(_read(claim['authorization']),claim['packet_sha256'])
    declaration=_read(run/'declaration.json')
    if declaration['source']!=prepared_source(): raise ValueError('source changed')
    _titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    from nca.ik.declaration import runtime_expectation
    from nca.ik.polar import fit_training_normalization
    from nca.train.ik_corpus import read_polar_followup_corpus
    from nca.train.ik_experiment import corpus_identity, train_attempt, _qualified_titan_gpu_bytes, validate_plan, PLAN_PATH
    from nca.ik.declaration import _project
    from nca.ik.device import device_report, configure_device
    if runtime_expectation()!=declaration['runtime']: raise ValueError('runtime changed')
    corpus=read_polar_followup_corpus(args.corpus,args.registry,source_sha256=declaration['source']['source_sha256'])
    norm=fit_training_normalization(corpus)
    if corpus_identity(corpus)!=declaration['dataset'] or norm.to_dict()!=declaration['normalization']:
        raise ValueError('corpus or normalizer changed')
    bound=ledger(run,declaration,args.seed)
    attempt=bound.begin_attempt(started=args.started)
    try:
        expected=validate_plan(_read(_project(None)/PLAN_PATH))['qualified_environment']
        size=_qualified_titan_gpu_bytes(device_report('cuda:0'),expected)
        context=configure_device('cuda:0',args.seed,memory_fraction=5*GIB/size,
            source_sha256=declaration['source']['source_sha256'],config_sha256=sha256_json(candidate_config()))
        context.torch.set_num_threads(1); context.torch.set_num_interop_threads(1)
        result=train_attempt(bound,attempt,corpus,context,epochs=40,batch_size=128,
            polar_normalization=norm,matched_candidate=True,
            execution=dict(candidate_sha256=sha256_json(candidate_config()),claim=args.claim))
        result['process_resources']=process_metrics()
        immutable_json(run/f'seed-{args.seed}'/'worker-result.json',result)
    except BaseException:
        if bound.inspect()['active']: attempt.finish('failed',reason='matched_worker_failure')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('check-admission','run','_worker'))
    parser.add_argument('--packet',required=True)
    parser.add_argument('--mode',choices=('serial','parallel'))
    parser.add_argument('--review',action='append',default=[])
    parser.add_argument('--authorization'); parser.add_argument('--corpus'); parser.add_argument('--registry')
    parser.add_argument('--seed',type=int,choices=SEEDS); parser.add_argument('--claim'); parser.add_argument('--started',type=float)
    args=parser.parse_args()
    if args.command=='check-admission': admission(args.packet,args.review)
    else:
        raise ValueError('matched supervisor draft: execution admission withheld pending validation')


if __name__=='__main__': main()
# Copyright 2026 techrote. All rights reserved.
