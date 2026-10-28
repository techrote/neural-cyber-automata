"""One-use, model-free custodian for the new frozen CPU final recipe.

Only hashes/counts and resource receipts are public. No consumed suite is read,
and no model/checkpoint or candidate output is available to this custodian.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import uuid

from nca.contracts import canonical_json, sha256_json, digest
from nca.runs import immutable_json, immutable_bytes, sha256_file, utc_now
from nca.ik.data import iter_tasks, generate_sample, validate_sample, _segment_distance
from nca.ik.frozen_final_v1 import FrozenFinalConfig
from nca.ik.declaration import runtime_expectation
from nca.ik.process_limits import memory_limit
from nca.train.ik_experiment import _source, _read
from nca.train.ik_followup_seal import process_peak_bytes
from nca.ik.frozen_source_binding_v1 import source_binding

TRAIN_DECLARATION='5e0ce7c4e36bbcbebe7543f8d230c5df166cf33d291a01d847efc6ed9ad2c4f8'
STARTED=time.monotonic()


def geometry():
    config=FrozenFinalConfig(); tasks=tuple(iter_tasks(config,'final'))
    if len(tasks)!=2940:raise ValueError('full final schedule required')
    return config,tasks,{'config_sha256':config.sha256,'task_count':len(tasks),
        'ordered_task_ids_sha256':sha256_json([t.id for t in tasks]),
        'ordered_geometry_sha256':sha256_json([t.to_dict() for t in tasks])}


def segments(tasks, *, joint):
    result=[];trajectories=defaultdict(dict)
    for task in tasks:
        p=task.target_configuration if joint else task.inputs.target_root
        if p is not None:result.append((p,p))
        if task.state.kind=='trajectory':trajectories[task.trajectory_id][task.state.step]=p
    for steps in trajectories.values():
        if set(steps)!=set(range(20)):raise ValueError('whole trajectory required for isolation')
        result.extend((steps[i],steps[i+1]) for i in range(19))
    return result


def separated(left,right,guard):
    """Closed continuous-segment guard, with a conservative spatial broad phase."""
    cell=guard*20;grid=defaultdict(list)
    def cells(a,b):
        return ((x,y) for x in range(math.floor((min(a[0],b[0])-guard)/cell),math.floor((max(a[0],b[0])+guard)/cell)+1)
            for y in range(math.floor((min(a[1],b[1])-guard)/cell),math.floor((max(a[1],b[1])+guard)/cell)+1))
    for i,(a,b) in enumerate(left):
        for key in cells(a,b):grid[key].append(i)
    checked=0
    for a,b in right:
        candidates=set(i for key in cells(a,b) for i in grid.get(key,()))
        for i in candidates:
            checked+=1
            if _segment_distance(a,b,*left[i])<=guard:
                raise ValueError('new geometry enters an existing train/tuning continuous-path guard')
    return {'left_segments':len(left),'right_segments':len(right),'near_pairs_checked':checked,
            'clearance_strictly_greater_than':guard,'pass':True}


def isolation(training_declaration,tasks):
    if sha256_file(training_declaration)!=TRAIN_DECLARATION:raise ValueError('training declaration differs')
    d=_read(training_declaration,16*1024**2)
    from nca.train.ik_corpus import read_polar_followup_corpus
    corpus=read_polar_followup_corpus(d['corpus'],d['registry'],source_sha256=d['source']['source_sha256'])
    if corpus.identity!=d['corpus_identity']:raise ValueError('original train/tuning identity differs')
    previous=tuple(s.task for s in corpus.training+corpus.tuning)
    for field in ('id','input_sha256','family_id','trajectory_id'):
        if {getattr(t,field) for t in previous}&{getattr(t,field) for t in tasks}:
            raise ValueError('new final identity or permitted-input duplicate')
    return {'schema':'nca-ik-frozen-final-isolation/v1','training_declaration_file_sha256':TRAIN_DECLARATION,
        'corpus_identity':corpus.identity,'original_rows':len(previous),'new_rows':len(tasks),
        'ordered_new_geometry_sha256':sha256_json([t.to_dict() for t in tasks]),
        'family_trajectory_input_and_id_disjoint':True,
        'root_point_and_complete_path_guard':separated(segments(previous,joint=False),segments(tasks,joint=False),.0005),
        'joint_point_and_complete_path_guard':separated(segments(previous,joint=True),segments(tasks,joint=True),.002),
        'prior_suites_read':False,'prior_final_geometry_claim':'new namespace,seed and nominal recipe ranges; no whole-prior-suite guard claim',
        'models_or_candidate_outputs_used':False}


def validate_isolation(receipt,identity,corpus_identity):
    if (receipt.get('schema')!='nca-ik-frozen-final-isolation/v1'
        or receipt.get('training_declaration_file_sha256')!=TRAIN_DECLARATION
        or receipt.get('corpus_identity')!=corpus_identity or receipt.get('original_rows')!=7280
        or receipt.get('new_rows')!=2940 or receipt.get('ordered_new_geometry_sha256')!=identity['ordered_geometry_sha256']
        or receipt.get('family_trajectory_input_and_id_disjoint') is not True
        or receipt.get('prior_suites_read') is not False or receipt.get('models_or_candidate_outputs_used') is not False):
        raise ValueError('complete model-free train/tuning isolation required')
    for name,left,right,guard in (('root_point_and_complete_path_guard',8800,3700,.0005),
                                 ('joint_point_and_complete_path_guard',8760,3680,.002)):
        g=receipt.get(name,{})
        if (g.get('pass') is not True or g.get('clearance_strictly_greater_than')!=guard
            or g.get('left_segments')!=left or g.get('right_segments')!=right
            or type(g.get('near_pairs_checked')) is not int or g['near_pairs_checked']<0):
            raise ValueError('complete continuous-path isolation guard required')


def declare(root,training_declaration):
    root=Path(root).resolve();checkout=Path(__file__).resolve().parents[3]
    if root==checkout or checkout in root.parents or root.exists():raise ValueError('new external custodian allocation required')
    source=_source(checkout)
    if sha256_file(training_declaration)!=TRAIN_DECLARATION:raise ValueError('training declaration differs')
    training=_read(training_declaration,16*1024**2)
    # Freeze the recipe and finite generation allowance before isolation/labels.
    root.mkdir(parents=True,exist_ok=False);config=FrozenFinalConfig()
    declaration={'schema':'nca-ik-frozen-seal-declaration/v1','id':uuid.uuid4().hex,'source':source,
        'runtime':runtime_expectation(),'config':config.to_dict(),'allocation':config.budget.to_dict(),
        'attempts':1,'retry_or_resume':False,'training_declaration_file_sha256':TRAIN_DECLARATION,
        'training_corpus_identity':training['corpus_identity'],
        'preparation_allocation':{'wall_seconds':3600,'process_commit_bytes':1024**3},
        'candidate_conditioned_filtering':False,'created_utc':utc_now()}
    declaration['sha256']=sha256_json(declaration);immutable_json(root/'declaration.json',declaration)
    try:
        _,tasks,identity=geometry()
        receipt=isolation(training_declaration,tasks)
        validate_isolation(receipt,identity,training['corpus_identity'])
        immutable_json(root/'geometry.json',identity);immutable_json(root/'isolation.json',receipt)
        preparation={'declaration_file_sha256':sha256_file(root/'declaration.json'),
            'elapsed_seconds':time.monotonic()-STARTED,'peak_process_bytes':process_peak_bytes(),
            'wall_seconds':3600,'process_commit_bytes':1024**3,'teacher_calls':0,'model_calls':0}
        if preparation['elapsed_seconds']>=3590 or preparation['peak_process_bytes']>1024**3:
            raise ValueError('preparation allocation exhausted before publication')
        immutable_json(root/'preparation.json',preparation)
    except BaseException as exc:
        immutable_json(root/'preparation-failure.json',{'type':type(exc).__name__,'message':str(exc),'retry_or_resume':False});raise
    return {'declaration_file_sha256':sha256_file(root/'declaration.json'),
            'geometry_file_sha256':sha256_file(root/'geometry.json'),'isolation_file_sha256':sha256_file(root/'isolation.json')}


def load(root):
    root=Path(root);d=_read(root/'declaration.json');config=FrozenFinalConfig();identity=_read(root/'geometry.json')
    source_binding(d['source'],_source(Path(__file__).resolve().parents[3]))
    validate_isolation(_read(root/'isolation.json'),identity,d.get('training_corpus_identity'))
    preparation=_read(root/'preparation.json')
    for key in ('ordered_task_ids_sha256','ordered_geometry_sha256'):digest(identity.get(key),key)
    if (d.get('schema')!='nca-ik-frozen-seal-declaration/v1'
        or d.get('runtime')!=runtime_expectation() or d.get('config')!=config.to_dict()
        or d.get('allocation')!=config.budget.to_dict() or d.get('attempts')!=1 or d.get('retry_or_resume') is not False
        or d.get('sha256')!=sha256_json({k:v for k,v in d.items() if k!='sha256'})
        or d.get('candidate_conditioned_filtering') is not False or identity.get('config_sha256')!=config.sha256
        or identity.get('task_count')!=2940
        or d.get('training_declaration_file_sha256')!=TRAIN_DECLARATION
        or d.get('preparation_allocation')!={'wall_seconds':3600,'process_commit_bytes':1024**3}
        or preparation.get('declaration_file_sha256')!=sha256_file(root/'declaration.json')
        or not 0<=preparation.get('elapsed_seconds',3600)<3590
        or not 0<preparation.get('peak_process_bytes',float('inf'))<=1024**3
        or preparation.get('wall_seconds')!=3600 or preparation.get('process_commit_bytes')!=1024**3
        or preparation.get('teacher_calls')!=0 or preparation.get('model_calls')!=0
        or (root/'preparation-failure.json').exists()):raise ValueError('new suite declaration/isolation/source differs')
    return d


def worker(root):
    memory_method=memory_limit(512)
    root=Path(root);d=load(root);config,tasks,identity=geometry()
    if identity!=_read(root/'geometry.json'):raise ValueError('frozen new geometry differs')
    if not (root/'started.json').is_file():raise ValueError('supervisor claim required')
    rows=[];iterations=0;peak=0
    with (root/'attempts.jsonl').open('xb') as journal:
        for task in tasks:
            if time.monotonic()-STARTED>780:raise TimeoutError('generation stop reserve reached')
            journal.write((canonical_json({'task_id':task.id,'teacher_attempts':1,'reserved_iterations':64})+'\n').encode());journal.flush();os.fsync(journal.fileno())
            sample=generate_sample(task,config);validate_sample(sample,config)
            iterations+=sample.label.teacher.iterations;rows.append((canonical_json(sample.to_dict())+'\n').encode())
            peak=max(peak,process_peak_bytes())
            if iterations>188160 or peak>512*1024**2:raise ValueError('generation resource cap exceeded')
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for index in range(0,len(rows),64):
            raw=b''.join(rows[index:index+64]);info=tarfile.TarInfo(f'final-{index//64:05d}.jsonl');info.size=len(raw)
            archive.addfile(info,io.BytesIO(raw))
    raw=stream.getvalue();digest=immutable_bytes(root/'sealed-final.tar.gz',raw);peak=max(peak,process_peak_bytes())
    if peak>512*1024**2:raise ValueError('generation publication memory cap exceeded')
    receipt={'schema':'nca-ik-frozen-final-seal/v1','declaration_sha256':d['sha256'],'registry_id':d['id'],
        'geometry':identity,'isolation_file_sha256':sha256_file(root/'isolation.json'),
        'archive_sha256':digest,'archive_bytes':len(raw),'rows':len(rows),'teacher_attempts':len(rows),
        'teacher_iterations':iterations,'reserved_teacher_iterations':188160,
        'attempt_journal_sha256':sha256_file(root/'attempts.jsonl'),'peak_process_bytes':peak,
        'hard_memory_limit_bytes':512*1024**2,'memory_limit_method':memory_method,
        'worker_seconds':time.monotonic()-STARTED,'final_state':'sealed','candidate_conditioned_filtering':False}
    receipt['sha256']=sha256_json(receipt);immutable_json(root/'seal.json',receipt);return receipt


def supervise(root):
    root=Path(root);d=load(root)
    immutable_json(root/'started.json',{'declaration_sha256':d['sha256'],'attempt':1,'retry_or_resume':False})
    with (root/'worker.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-m',__spec__.name,'_worker','--root',str(root)],stdout=log,stderr=log)
    timed_out=False
    try:p.wait(timeout=max(.001,870-(time.monotonic()-STARTED)))
    except subprocess.TimeoutExpired:
        timed_out=True;p.kill();p.wait(timeout=10)
    seal=_read(root/'seal.json') if (root/'seal.json').exists() else None
    complete=(not timed_out and p.returncode==0 and seal is not None and time.monotonic()-STARTED<895
        and seal.get('sha256')==sha256_json({k:v for k,v in seal.items() if k!='sha256'})
        and seal.get('rows')==2940 and seal.get('teacher_attempts')==2940 and seal.get('reserved_teacher_iterations')==188160
        and 0<=seal.get('teacher_iterations',-1)<=188160 and seal.get('peak_process_bytes',float('inf'))<=512*1024**2
        and seal.get('hard_memory_limit_bytes')==512*1024**2 and seal.get('candidate_conditioned_filtering') is False
        and seal.get('declaration_sha256')==d['sha256'] and seal.get('geometry')==_read(root/'geometry.json')
        and seal.get('isolation_file_sha256')==sha256_file(root/'isolation.json')
        and sha256_file(root/'attempts.jsonl')==seal['attempt_journal_sha256']
        and (root/'sealed-final.tar.gz').stat().st_size==seal['archive_bytes']
        and sha256_file(root/'sealed-final.tar.gz')==seal['archive_sha256'])
    receipt={'schema':'nca-ik-frozen-seal-supervisor/v1','declaration_sha256':d['sha256'],'returncode':p.returncode,
        'timed_out':timed_out,'whole_process_seconds':time.monotonic()-STARTED,'complete':complete,
        'seal_sha256':None if seal is None else seal['sha256'],'retry_or_resume':False}
    receipt['archive_file_sha256']=None if seal is None else seal['archive_sha256']
    immutable_json(root/'supervisor.json',receipt)
    if not complete or time.monotonic()-STARTED>=900:
        immutable_json(root/'supervisor-failure.json',{'complete':False,'retry_or_resume':False});raise ValueError('generation failed; preserve consumed allocation')
    return receipt


def inspect(root):
    """New suite metadata/stat only. Only the custodian hashes its publication."""
    root=Path(root);d=load(root);seal=_read(root/'seal.json');supervisor=_read(root/'supervisor.json')
    observed_exit=_read(root/'process-exit.json')
    if (seal.get('schema')!='nca-ik-frozen-final-seal/v1'
        or seal.get('sha256')!=sha256_json({k:v for k,v in seal.items() if k!='sha256'})
        or seal.get('declaration_sha256')!=d['sha256'] or seal.get('registry_id')!=d['id']
        or seal.get('geometry')!=_read(root/'geometry.json') or seal.get('isolation_file_sha256')!=sha256_file(root/'isolation.json')
        or seal.get('rows')!=2940 or seal.get('teacher_attempts')!=2940 or seal.get('reserved_teacher_iterations')!=188160
        or not 0<=seal.get('teacher_iterations',-1)<=188160 or seal.get('peak_process_bytes',float('inf'))>512*1024**2
        or seal.get('hard_memory_limit_bytes')!=512*1024**2 or seal.get('final_state')!='sealed'
        or seal.get('memory_limit_method') not in ('RLIMIT_AS','Windows Job Object process committed-memory limit')
        or seal.get('candidate_conditioned_filtering') is not False or supervisor.get('complete') is not True
        or supervisor.get('schema')!='nca-ik-frozen-seal-supervisor/v1' or supervisor.get('retry_or_resume') is not False
        or supervisor.get('returncode')!=0 or supervisor.get('timed_out') is not False
        or supervisor.get('whole_process_seconds',900)>=900 or supervisor.get('declaration_sha256')!=d['sha256']
        or supervisor.get('seal_sha256')!=seal['sha256'] or (root/'supervisor-failure.json').exists()
        or supervisor.get('archive_file_sha256')!=seal['archive_sha256']
        or observed_exit!={'exit_code':0,'seal_file_sha256':sha256_file(root/'seal.json'),
            'supervisor_file_sha256':sha256_file(root/'supervisor.json'),'failure_marker_absent':True}
        or (root/'opened.json').exists() or sha256_file(root/'attempts.jsonl')!=seal['attempt_journal_sha256']
        or (root/'sealed-final.tar.gz').stat().st_size!=seal['archive_bytes']):
        raise ValueError('new suite custody or finite generation proof differs')
    return seal


def main():
    global STARTED
    from nca.train.ik_qualification_v4 import invocation_start,hard_wall_watchdog
    from threading import Thread,Event
    STARTED=invocation_start()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or 'torch' in sys.modules:
        raise ValueError('custodian must be model-free with CUDA visibility empty')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('declare','generate','inspect','_worker'))
    parser.add_argument('--root',required=True);parser.add_argument('--training-declaration')
    a=parser.parse_args()
    if time.monotonic()-STARTED>30:raise ValueError('custodian startup allowance exceeded')
    limit=3600 if a.command=='declare' else 900
    if a.command!='_worker':memory_limit(1024 if a.command=='declare' else 512,kill_children_on_exit=True)
    Thread(target=hard_wall_watchdog,args=(Event(),STARTED,limit-5),daemon=True).start()
    result=declare(a.root,a.training_declaration) if a.command=='declare' else {'generate':supervise,'inspect':inspect,'_worker':worker}[a.command](a.root)
    if 'torch' in sys.modules:raise ValueError('custodian imported Torch')
    print(canonical_json(result))


if __name__=='__main__':main()

# Copyright 2026 techrote. All rights reserved.
