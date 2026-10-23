"""Separately admitted allocation-only qualification; no learner imports/data."""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
ENTRY_STARTED=time.monotonic()
from nca.contracts import canonical_json,sha256_json
from nca.runs import immutable_json,sha256_file
from nca.train.ik_matched_prepare import prepared_source,TITAN_FINGERPRINT,GIB
from nca.train.ik_experiment import _read,_titan_visibility
from nca.train.ik_matched_supervisor import board_sample,stop_tree,process_metrics


def validate_predecessor(previous):
    previous=Path(previous)
    if sha256_file(previous/'result.json')!=limits()['prior_failed_result_sha256']:
        raise ValueError('exact retained failed qualification required')
    result=_read(previous/'result.json')
    if result['outcome']!='failed' or result['cleanup_errors']:
        raise ValueError('settled failed predecessor required')
    for name,expected in result['artifacts'].items():
        if sha256_file(previous/name)!=expected: raise ValueError('predecessor evidence changed')
    return _read(previous/'declaration.json')


def retire_predecessor(root,value):
    previous=Path(value['previous_probe']);old=validate_predecessor(previous)
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    termination=[]
    for index in range(3):
        pid=_read(previous/f'worker-{index}-ready.json')['pid']
        ctypes.set_last_error(0);handle=kernel.OpenProcess(0x1000,False,pid)
        if handle:
            kernel.CloseHandle(handle);raise ValueError('predecessor PID still exists')
        if ctypes.get_last_error()!=87: raise ValueError('predecessor termination unverified')
        termination.append(dict(pid=pid,open_process_error=87))
    from nca.train.ik_gpu_ownership import path
    expected=dict(kind='allocation_probe',declaration_sha256=old['sha256'])
    if _read(path())!=expected: raise ValueError('unexpected retained TITAN owner')
    from nca.runs import immutable_bytes
    old_bytes=path().read_bytes()
    copy=Path(root)/'retired-predecessor-ownership.json'
    immutable_bytes(copy,old_bytes)
    receipt=dict(previous_result_sha256=sha256_file(previous/'result.json'),old_ownership_file_sha256=sha256_file(copy),
        successor_declaration_sha256=value['sha256'],termination_proofs=termination,
        admission_file_sha256=_read(Path(root)/'claim.json')['review_sha256'],all_predecessor_pids_absent=True)
    immutable_json(Path(root)/'ownership-retirement.json',receipt)
    if (copy.read_bytes()!=old_bytes or _read(Path(root)/'ownership-retirement.json')!=receipt
            or path().read_bytes()!=old_bytes): raise ValueError('ownership retirement readback changed')
    path().unlink()


def validate_resources(resources):
    for key in ('peak_resident_bytes','private_commit_bytes','pagefile_bytes','peak_pagefile_bytes'):
        n=resources.get(key)
        if type(n)!=int or not 0<=n<=8*GIB: raise ValueError('host memory cap or missing commit telemetry')
    if max(resources['private_commit_bytes'],resources['pagefile_bytes'])>resources['peak_pagefile_bytes']:
        raise ValueError('inconsistent process commit peak')


def limits():
    return dict(workers=3,chunk_bytes=256*1024**2,chunks=20,allocator_bytes_per_worker=5*GIB,
                board_bytes=20*GIB,wall_seconds=300,active_seconds=240,hold_seconds=30,
                host_peak_bytes_per_process=8*GIB,minimum_available_commit_bytes=36*GIB,
                minimum_available_physical_bytes=32*GIB,maximum_temperature_celsius=80,
                one_attempt=True,no_training=True,no_data=True,touch_zero=True,warmup_bytes=1024**2,
                amendment='host_commit_v2',prior_failed_result_sha256='77220c2773c6f9e36a4288639c7cb5c384a0355c07f21c476e60722d05827e35')


def host_headroom():
    import ctypes
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
            ('total_physical','available_physical','total_commit','available_commit','total_virtual','available_virtual','available_extended')]
    value=Status();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)): raise ValueError('host headroom unavailable')
    if value.available_physical<32*GIB or value.available_commit<36*GIB:
        raise ValueError('declared host aggregate headroom insufficient')
    return dict(available_physical_bytes=value.available_physical,available_commit_bytes=value.available_commit)


def prepare(root,previous):
    source=prepared_source();_titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    from nca.ik.declaration import runtime_expectation
    value=dict(schema='nca-ik-memory-probe/v1',source=source,limits=limits(),
               runtime=runtime_expectation(),gpu_visibility_sha256=TITAN_FINGERPRINT,previous_probe=str(Path(previous).resolve()),host_headroom=host_headroom())
    validate_predecessor(previous)
    value['sha256']=sha256_json(value)
    Path(root).mkdir(parents=True,exist_ok=False)
    immutable_json(Path(root)/'declaration.json',value)
    return value


def verify(root,reviews):
    value=_read(Path(root)/'declaration.json')
    if (value['source']!=prepared_source() or value['limits']!=limits()
            or value['sha256']!=sha256_json({k:v for k,v in value.items() if k!='sha256'})):
        raise ValueError('probe source/declaration changed')
    reviewers=set()
    for path in reviews:
        review=_read(path)
        if (review.get('schema')!='nca-ik-memory-probe-admission/v1'
                or review.get('declaration_sha256')!=value['sha256']
                or review.get('source_sha256')!=value['source']['source_sha256']
                or review.get('head')!=value['source']['revision']
                or review.get('verdict')!='admit_allocation_only_probe'
                or review.get('independent_non_author') is not True
                or review.get('exact_head_ci') is not True):
            raise ValueError('two prospective allocation-only admissions required')
        reviewer=review.get('reviewer')
        if type(reviewer)!=str or not reviewer or reviewer in reviewers: raise ValueError('distinct reviewers required')
        reviewers.add(reviewer)
    if len(reviewers)!=2: raise ValueError('exactly two probe admissions required')
    _titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    return value


def worker(root,index,reviews):
    if type(index)!=int or index not in (0,1,2): raise ValueError('exact declared worker index required')
    root=Path(root); declaration=verify(root,reviews)
    immutable_json(root/f'worker-{index}-claim.json',dict(index=index,declaration_sha256=declaration['sha256']))
    claim=_read(root/'claim.json')
    from nca.train.ik_gpu_ownership import verify as owner_verify
    owner_verify(dict(kind='allocation_probe',declaration_sha256=declaration['sha256']))
    if claim['declaration_sha256']!=declaration['sha256']: raise ValueError('probe ownership differs')
    from nca.ik.process_limits import memory_limit
    memory_limit(8192)
    from nca.ik.declaration import runtime_expectation
    if declaration['runtime']!=runtime_expectation(): raise ValueError('runtime changed')
    from nca.ik.device import configure_device,device_report
    from nca.train.ik_experiment import _qualified_titan_gpu_bytes,validate_plan,PLAN_PATH
    from nca.ik.declaration import _project
    expected=validate_plan(_read(_project(None)/PLAN_PATH))['qualified_environment']
    size=_qualified_titan_gpu_bytes(device_report('cuda:0'),expected)
    context=configure_device('cuda:0',61003+index,memory_fraction=5*GIB/size)
    warmup=context.torch.zeros(1024**2,device='cuda:0',dtype=context.torch.uint8)
    context.torch.cuda.synchronize();del warmup;context.torch.cuda.empty_cache()
    buffers=[]
    immutable_json(root/f'worker-{index}-ready.json',dict(pid=os.getpid(),monotonic=time.monotonic(),context=context.report,resources=process_metrics()))
    for step in range(1,21):
        while not (root/f'approve-{index}-{step}.json').exists():
            if time.monotonic()-claim['started']>240: raise TimeoutError('probe deadline')
            time.sleep(.05)
        approval=_read(root/f'approve-{index}-{step}.json')
        if approval!=dict(step=step,index=index,declaration_sha256=declaration['sha256']):
            raise ValueError('probe allocation approval differs')
        buffers.append(context.torch.zeros(256*1024**2,device='cuda:0',dtype=context.torch.uint8))
        context.torch.cuda.synchronize()
        resources=process_metrics()
        validate_resources(resources)
        immutable_json(root/f'worker-{index}-{step}.json',dict(pid=os.getpid(),monotonic=time.monotonic(),memory=context.memory_stats(),resources=resources))
    while not (root/'release.json').exists():
        if time.monotonic()-claim['started']>240: raise TimeoutError('probe deadline')
        time.sleep(.05)
    immutable_json(root/f'worker-{index}-exit.json',dict(pid=os.getpid(),monotonic=time.monotonic()))


def run(root,reviews):
    root=Path(root);start=ENTRY_STARTED; value=verify(root,reviews)
    immutable_json(root/'claim.json',dict(declaration_sha256=value['sha256'],started=start,
        review_sha256=[sha256_file(p) for p in reviews]))
    from nca.train.ik_gpu_ownership import acquire,release
    ownership=dict(kind='allocation_probe',declaration_sha256=value['sha256'])
    processes=[];logs=[]
    def sample(stream):
        if time.monotonic()-start>240: raise TimeoutError('probe active deadline')
        if any(p.poll() is not None for p in processes): raise ValueError('probe child ended early')
        queried=time.monotonic();row=board_sample()
        if time.monotonic()-queried>2: raise ValueError('stale probe telemetry')
        if row['temperature']>80: raise ValueError('temperature cap')
        stream.write(canonical_json(row)+'\n');stream.flush();os.fsync(stream.fileno())
        return row
    outcome='failed';errors=[];baseline=None;released=None
    owned=False
    try:
        immutable_json(root/'host-headroom-at-launch.json',host_headroom())
        retire_predecessor(root,value)
        acquire(ownership);owned=True
        from nca.ik.process_limits import memory_limit
        memory_limit(8192,kill_children_on_exit=True)
        with (root/'telemetry.jsonl').open('x',encoding='utf-8') as stream:
            sample(stream)
            for index in range(3):
                log=(root/f'worker-{index}.log').open('xb');logs.append(log)
                cmd=[sys.executable,'-m','nca.train.ik_memory_probe','_worker','--root',str(root),'--index',str(index)]
                for review in reviews: cmd+=['--review',str(Path(review).resolve())]
                processes.append(subprocess.Popen(cmd,stdout=log,stderr=log,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')))
                while not (root/f'worker-{index}-ready.json').exists(): sample(stream);time.sleep(.05)
            baseline=sample(stream)
            if baseline['used_bytes']+15*GIB+512*1024**2>20*GIB:
                raise ValueError('context baseline leaves insufficient full-envelope margin')
            for step in range(1,21):
                for index in range(3):
                    current=sample(stream)
                    if current['used_bytes']+256*1024**2+512*1024**2>20*GIB:
                        raise ValueError('insufficient projected allocation headroom')
                    immutable_json(root/f'approve-{index}-{step}.json',dict(step=step,index=index,declaration_sha256=value['sha256']))
                    while not (root/f'worker-{index}-{step}.json').exists(): sample(stream);time.sleep(.05)
                    sample(stream)
            held=time.monotonic()
            while time.monotonic()-held<30: sample(stream);time.sleep(.25)
            immutable_json(root/'release.json',dict(release=True))
            for process in processes:
                process.wait(timeout=max(.1,290-(time.monotonic()-start)))
                if process.returncode: raise ValueError('probe exit failure')
            released=board_sample()
            if released['used_bytes']>baseline['used_bytes']+512*1024**2:
                raise ValueError('probe memory not released within declared margin')
            outcome='pass'
    finally:
        for process in processes:
            try: stop_tree(process)
            except BaseException as exc: errors.append(type(exc).__name__)
        for log in logs: log.close()
        elapsed=time.monotonic()-start
        immutable_json(root/'result.json',dict(schema='nca-ik-memory-probe-result/v1',
            declaration_sha256=value['sha256'],outcome=outcome if not errors and elapsed<300 else 'failed',
            whole_seconds=elapsed,charged_wall_ceiling_seconds=300,cleanup_errors=errors,
            context_baseline=baseline,released_board=released,
            artifacts={p.relative_to(root).as_posix():sha256_file(p) for p in root.rglob('*') if p.is_file()},
            qualification='three_contexts_and_touched_buffers_only_not_training_performance',training=False))
        if time.monotonic()-start>=300:
            immutable_json(root/'publication-overrun.json',dict(qualified=False,unknown_work=True))
            raise TimeoutError('probe publication exceeded paid wall bound')
        if outcome=='pass' and not errors and owned: release(ownership)
        if time.monotonic()-start>=300:
            immutable_json(root/'cleanup-overrun.json',dict(qualified=False,unknown_work=True))
            raise TimeoutError('probe cleanup exceeded paid wall bound')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','_worker'))
    parser.add_argument('--root',required=True);parser.add_argument('--previous-probe');parser.add_argument('--review',action='append',default=[])
    parser.add_argument('--index',type=int,choices=(0,1,2))
    args=parser.parse_args()
    if args.command=='prepare': print(canonical_json(prepare(args.root,args.previous_probe)))
    elif args.command=='run': run(args.root,args.review)
    else: worker(args.root,args.index,args.review)


def qualification_record(root,source):
    root=Path(root).resolve();declaration=_read(root/'declaration.json');result=_read(root/'result.json')
    if not {'source','limits','sha256'}.issubset(declaration):
        raise ValueError('complete qualification declaration required')
    if (declaration['source']!=source or canonical_json(declaration['limits'])!=canonical_json(limits())
            or declaration['sha256']!=sha256_json({k:v for k,v in declaration.items() if k!='sha256'})
            or result.get('schema')!='nca-ik-memory-probe-result/v1' or result.get('outcome')!='pass'
            or result.get('charged_wall_ceiling_seconds')!=300
            or result.get('declaration_sha256')!=declaration['sha256']
            or result.get('whole_seconds',301)>=300 or result.get('cleanup_errors')
            or any(root.glob('*overrun.json')) or result.get('training') is not False):
        raise ValueError('current source-bound successful non-training qualification required')
    for name,expected in result['artifacts'].items():
        p=root/name
        if root not in p.resolve().parents or sha256_file(p)!=expected: raise ValueError('qualification evidence changed')
    for index in range(3):
        required=[f'worker-{index}-ready.json',f'worker-{index}-exit.json',f'worker-{index}-claim.json']
        required += [f'worker-{index}-{s}.json' for s in range(1,21)]
        required += [f'approve-{index}-{s}.json' for s in range(1,21)]
        if not set(required).issubset(result['artifacts']): raise ValueError('complete worker evidence required')
        for step in range(1,21):
            item=_read(root/f'worker-{index}-{step}.json')
            validate_resources(item['resources'])
            if (item['memory']['allocated_bytes']!=step*256*1024**2
                    or item['memory']['peak_reserved_bytes']>5*GIB
                    or item['resources']['peak_resident_bytes']>8*GIB):
                raise ValueError('declared touched allocations/host memory not established')
    if not {'telemetry.jsonl','declaration.json','claim.json','release.json'}.issubset(result['artifacts']):
        raise ValueError('complete probe telemetry and ownership required')
    ready=[_read(root/f'worker-{i}-ready.json') for i in range(3)]
    full=[_read(root/f'worker-{i}-20.json') for i in range(3)]
    exits=[_read(root/f'worker-{i}-exit.json') for i in range(3)]
    if (len({r['pid'] for r in ready})!=3
            or min(x['monotonic'] for x in exits)-max(x['monotonic'] for x in full)<30):
        raise ValueError('three distinct concurrent held workers required')
    from nca.contracts import strict_json_loads
    readings=[strict_json_loads(line) for line in (root/'telemetry.jsonl').read_bytes().splitlines()]
    if not readings or any(not math.isfinite(r['used_bytes']) or r['used_bytes']>20*GIB or r['temperature']>80 for r in readings):
        raise ValueError('bounded complete board telemetry required')
    if any(b['monotonic']-a['monotonic']>2.5 for a,b in zip(readings,readings[1:])):
        raise ValueError('probe telemetry contains a stale gap')
    if (result['context_baseline']['used_bytes']+15*GIB+512*1024**2>20*GIB
            or result['released_board']['used_bytes']>result['context_baseline']['used_bytes']+512*1024**2):
        raise ValueError('context headroom or memory release unverified')
    return dict(root=str(root),result_file_sha256=sha256_file(root/'result.json'),
                declaration_sha256=declaration['sha256'],scope=result['qualification'])


if __name__=='__main__': main()
# Copyright 2026 techrote. All rights reserved.

