"""Single-use qualification of frozen CPU inference on a new sealed suite.

No consumed final directory is an input. Historical custody helpers authenticate
training/tuning only; candidate freezes are rebuilt for the new suite identity.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import os
from pathlib import Path
import shutil
import tarfile
import time
import uuid
from nca.contracts import canonical_json,sha256_json,strict_json_loads
from nca.runs import immutable_json,sha256_file,utc_now
from nca.train.ik_experiment import _source,_read,_titan_visibility
from nca.train.ik_matched_prepare import TITAN_FINGERPRINT,GIB
from nca.ik.frozen_final_v1 import FrozenFinalConfig
from nca.evaluation.ik_training import freeze_candidates
from nca.evaluation import ik54_final_v1 as historical
from nca.train import ik_frozen_seal_v1 as custodian
from nca.train.ik_qualification_v4 import SEEDS
from nca.train.ik_run_state_v2 import _verify_checkpoint

WALL_SECONDS=86400
EXIT_MARGIN_SECONDS=30
FINALIZATION_RESERVE_SECONDS=7200
EVALUATIONS_PER_SEED=78240
ACCEPTANCE_SHA=historical.ACCEPTANCE_SHA
COMPARATOR_NAMES=historical.COMPARATOR_NAMES
CHECKS=historical.CHECKS+('frozen_graph_parity_and_identity','new_geometry_isolation')
_bounded_json=historical._bounded_json
_write_report=historical._write_report
_external=historical._external
_counter_predictors=historical._counter_predictors
load_models=historical.load_models
REPORT_MAX_BYTES=historical.REPORT_MAX_BYTES


def inference_route():
    from nca.ik.inference_v1 import INFERENCE_ID
    import importlib.util
    spec=importlib.util.find_spec('torch')
    trace_path=None if spec is None else Path(spec.origin).parent/'jit/_trace.py'
    return {'id':INFERENCE_ID,'source_file_sha256':sha256_file(Path(__file__).resolve().parents[1]/'ik/inference_v1.py'),
            'construction':'torch.jit.trace(check_trace=True) then torch.jit.freeze',
            'torch_trace_source_sha256':None if trace_path is None else sha256_file(trace_path),
            'trained_and_matched_initial':'frozen','archived_control':'eager','batch_size':1,'dtype':'float32'}


def allocation():
    return {**historical.allocation(),'graph_construction_calls_per_seed':16,
            'proof_predictor_calls_per_seed':8080,'proof_graph_construction_calls_per_seed':16,
            'proof_wall_seconds':3600,'proof_memory_bytes':8*GIB}


def training_choices(training,polar,archived,seal,*,full=True):
    d,_,candidates,prior=historical.training_candidates(training,polar,archived,full=full)
    choices={e['seed']:{**{k:v for k,v in e.items() if k!='seed'},'config_sha256':FrozenFinalConfig().sha256} for e in prior['candidates']}
    frozen=freeze_candidates(choices,experiment='ik54-frozen-final-v1',dataset_manifest_sha256=seal['sha256'],
        final_suite_identity_sha256=seal['archive_sha256'],registry_id=seal['registry_id'])
    return d,candidates,frozen


def construct(seed,declaration,corpus):
    from nca.ik.device import configure_device
    from nca.ik.inference_v1 import freeze_batch_one
    context=configure_device('cpu',seed,source_sha256=declaration['source']['source_sha256'])
    context.torch.set_num_threads(1)
    trained,initial,old=load_models(seed,declaration,corpus,context)
    construction=[0];began=time.monotonic()
    def charge(module,args):
        construction[0]+=1
        if construction[0]>6:raise ValueError('synthetic eager-hook forward ceiling exceeded')
    result=[]
    for model in (trained,initial):
        hook=model.register_forward_pre_hook(charge)
        try:result.append(freeze_batch_one(context,model))
        finally:hook.remove()
    if context.torch.cuda.is_initialized():raise ValueError('CPU-only graph path initialized CUDA')
    if construction[0]!=6:raise ValueError('pinned tracing protocol eager-hook count differs')
    return (context,*result,old),{'seed':seed,'eager_hook_forward_calls':construction[0],
        'reserved_total_graph_forward_calls':16,'reviewed_total_forward_upper_bound':10,
        'construction_seconds':time.monotonic()-began,'route':inference_route()}


def corpus_for(training):
    from nca.train.ik_corpus import read_polar_followup_corpus
    from nca.ik.polar import fit_training_normalization
    corpus=read_polar_followup_corpus(training['corpus'],training['registry'],source_sha256=training['source']['source_sha256'])
    if corpus.identity!=training['corpus_identity'] or fit_training_normalization(corpus).to_dict()!=training['normalization']:
        raise ValueError('retained train/tuning corpus or normalizer changed')
    return corpus


def prove(args):
    from nca.train.ik_qualification_v4 import invocation_start,hard_wall_watchdog
    from nca.ik.process_limits import memory_limit
    from threading import Event,Thread
    began=invocation_start();memory_limit(8192,kill_children_on_exit=True)
    Thread(target=hard_wall_watchdog,args=(Event(),began,3570),daemon=True).start()
    root=_external(args.root,args.training,args.archived,Path(args.polar_declaration).parent)
    source=_source(Path(__file__).resolve().parents[3])
    root.mkdir(parents=True,exist_ok=False)
    from nca.ik.declaration import runtime_expectation
    proof_d={'schema':'nca-ik-frozen-proof-declaration/v1','source':source,'runtime':runtime_expectation(),
        'allocation':allocation(),'route':inference_route(),'training_declaration_file_sha256':historical.TRAIN_DECLARATION,
        'partition':'tuning','optimizer_calls':0,'final_calls':0}
    immutable_json(root/'declaration.json',proof_d)
    try:
        training,_,candidates,_=historical.training_candidates(args.training,args.polar_declaration,args.archived)
        corpus=corpus_for(training)
        d={'source':source,'candidates':candidates,'training_root':str(Path(args.training).resolve()),
           'archived_root':str(Path(args.archived).resolve()),'normalization':training['normalization']}
        from nca.evaluation.ik_training import model_predictor,collect_predictions
        from nca.ik.device import configure_device
        context=configure_device('cpu',41011,source_sha256=source['source_sha256']);context.torch.set_num_interop_threads(1)
        results={}
        for seed in SEEDS:
            ready,construction=construct(seed,d,corpus);context=ready[0]
            eager=load_models(seed,d,corpus,context)
            maximum=0.;calls=[0]
            def counted(predictor):
                def call(inputs):
                    calls[0]+=1
                    if calls[0]>8080:raise ValueError('proof predictor ceiling exceeded')
                    return predictor(inputs)
                return call
            for ordinary,frozen in zip(eager[:2],ready[1:3]):
                a=collect_predictions(corpus.tuning,counted(model_predictor(context,ordinary)))
                b=collect_predictions(corpus.tuning,counted(model_predictor(context,frozen)))
                for key,x in a.items():
                    y=b[key]
                    if x.status!=y.status or (x.raw_reference is None)!=(y.raw_reference is None):raise ValueError('frozen refusal parity differs')
                    if x.raw_reference is not None:maximum=max(maximum,*(abs(i-j) for i,j in zip(x.raw_reference,y.raw_reference)))
            if maximum>2e-5 or calls[0]>8080:raise ValueError('frozen proof parity/call limit failed')
            results[str(seed)]={**construction,'max_joint_difference_radians':maximum,'rows_per_model':len(corpus.tuning),
                'actual_predictor_calls':calls[0],'selected_sha256':candidates[str(seed)]['selected']['reference']['sha256'],
                'initial_sha256':candidates[str(seed)]['initial']['reference']['sha256'],'pass':True}
        immutable_json(root/'proof.json',{'schema':'nca-ik-frozen-proof/v1','source':source,'runtime':runtime_expectation(),
            'route':inference_route(),'declaration_file_sha256':sha256_file(root/'declaration.json'),'results':results,
            'optimizer_calls':0,'final_calls':0,'elapsed_seconds':time.monotonic()-began,
            'peak_process_bytes':custodian.process_peak_bytes()})
    except BaseException as exc:
        immutable_json(root/'failure.json',{'type':type(exc).__name__,'message':str(exc)});raise


def verify_proof(root,source,candidates):
    from nca.ik.declaration import runtime_expectation
    root=Path(root);p=_bounded_json(root/'proof.json',16*1024**2);pd=_bounded_json(root/'declaration.json',16*1024**2)
    exit_receipt=_bounded_json(root/'process-exit.json')
    if ((root/'failure.json').exists() or p.get('source')!=source or p.get('runtime')!=runtime_expectation()
        or p.get('route')!=inference_route() or p.get('declaration_file_sha256')!=sha256_file(root/'declaration.json')
        or pd.get('source')!=source or canonical_json(pd.get('allocation'))!=canonical_json(allocation())
        or pd.get('runtime')!=runtime_expectation() or pd.get('route')!=inference_route()
        or pd.get('training_declaration_file_sha256')!=historical.TRAIN_DECLARATION
        or pd.get('schema')!='nca-ik-frozen-proof-declaration/v1' or pd.get('partition')!='tuning'
        or p.get('schema')!='nca-ik-frozen-proof/v1'
        or exit_receipt!={'exit_code':0,'proof_file_sha256':sha256_file(root/'proof.json'),'failure_marker_absent':True}
        or p.get('optimizer_calls')!=0 or p.get('final_calls')!=0 or not 0<=p.get('elapsed_seconds',3600)<3600
        or not 0<p.get('peak_process_bytes',float('inf'))<=8*GIB
        or set(p.get('results',{}))!={str(s) for s in SEEDS}):raise ValueError('exact-source frozen proof required')
    for seed in SEEDS:
        row=p['results'][str(seed)];entry=candidates[str(seed)]
        if (row.get('pass') is not True or row.get('rows_per_model')!=2020 or not 0<=row.get('max_joint_difference_radians',1)<=2e-5
            or row.get('eager_hook_forward_calls')!=6 or row.get('reserved_total_graph_forward_calls')!=16
            or row.get('reviewed_total_forward_upper_bound')!=10 or row.get('actual_predictor_calls')!=7840
            or row.get('selected_sha256')!=entry['selected']['reference']['sha256']
            or row.get('initial_sha256')!=entry['initial']['reference']['sha256']):raise ValueError('frozen proof candidates/counts differ')
    return p


def prepare(args):
    root=_external(args.root,args.training,args.suite,args.archived,args.proof,Path(args.polar_declaration).parent)
    _titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    source=_source(Path(__file__).resolve().parents[3]);seal=custodian.inspect(args.suite)
    training,candidates,frozen=training_choices(args.training,args.polar_declaration,args.archived,seal)
    corpus_for(training);verify_proof(args.proof,source,candidates)
    from nca.ik.declaration import runtime_expectation
    if runtime_expectation()!=training['runtime']:raise ValueError('original runtime changed')
    declaration={'schema':'nca-ik54-frozen-final-declaration/v1','id':'ik54-frozen-final-v1','source':source,
        'training_root':str(Path(args.training).resolve()),'polar_declaration':str(Path(args.polar_declaration).resolve()),
        'archived_root':str(Path(args.archived).resolve()),'suite_root':str(Path(args.suite).resolve()),
        'proof_root':str(Path(args.proof).resolve()),'proof_file_sha256':sha256_file(Path(args.proof)/'proof.json'),
        'proof_process_exit_file_sha256':sha256_file(Path(args.proof)/'process-exit.json'),
        'suite_declaration_file_sha256':sha256_file(Path(args.suite)/'declaration.json'),
        'suite_geometry_file_sha256':sha256_file(Path(args.suite)/'geometry.json'),
        'suite_process_exit_file_sha256':sha256_file(Path(args.suite)/'process-exit.json'),
        'suite_seal':seal,'candidates':candidates,'frozen_candidates':frozen,'normalization':training['normalization'],
        'runtime':training['runtime'],'inference_route':inference_route(),'config':FrozenFinalConfig().to_dict(),
        'gpu_visibility_sha256':TITAN_FINGERPRINT,'acceptance_file_sha256':ACCEPTANCE_SHA,'allocation':allocation(),
        'comparators':list(COMPARATOR_NAMES),'final_state':'new_sealed_suite','selection':'original_tuning_only_no_reselection',
        'retry_or_resume':False,'execution_authorized':False,'created_utc':utc_now()}
    root.mkdir(parents=True,exist_ok=False);immutable_json(root/'declaration.json',declaration);immutable_json(root/'frozen-candidates.json',frozen)
    immutable_json(root/'admission-template.json',{'schema':'nca-ik54-frozen-final-admission/v1','head':source['revision'],
        'declaration_file_sha256':sha256_file(root/'declaration.json'),'reviewer':None,'independent_non_author':False,
        'checks':{k:False for k in CHECKS},'verdict':'pending'})
    print(sha256_file(root/'declaration.json'))


def check_admissions(root,reviews,owner):
    root=Path(root);d=_bounded_json(root/'declaration.json',16*1024**2);source=_source(Path(__file__).resolve().parents[3]);digest=sha256_file(root/'declaration.json')
    from nca.ik.declaration import runtime_expectation
    if (d.get('schema')!='nca-ik54-frozen-final-declaration/v1' or d.get('id')!='ik54-frozen-final-v1' or d.get('source')!=source
        or canonical_json(d.get('allocation'))!=canonical_json(allocation()) or d.get('comparators')!=list(COMPARATOR_NAMES)
        or d.get('gpu_visibility_sha256')!=TITAN_FINGERPRINT or d.get('acceptance_file_sha256')!=ACCEPTANCE_SHA
        or d.get('final_state')!='new_sealed_suite' or d.get('selection')!='original_tuning_only_no_reselection'
        or d.get('execution_authorized') is not False or d.get('retry_or_resume') is not False
        or d.get('inference_route')!=inference_route() or d.get('config')!=FrozenFinalConfig().to_dict()
        or d.get('runtime')!=runtime_expectation() or _bounded_json(root/'frozen-candidates.json')!=d.get('frozen_candidates')):
        raise ValueError('exact-source frozen final declaration differs')
    seen=set()
    if len(reviews)!=2:raise ValueError('two independent admissions required')
    for path in reviews:
        v=_bounded_json(path)
        if (v.get('schema')!='nca-ik54-frozen-final-admission/v1' or v.get('head')!=source['revision']
            or v.get('declaration_file_sha256')!=digest or v.get('independent_non_author') is not True
            or set(v.get('checks',{}))!=set(CHECKS) or any(v['checks'][k] is not True for k in CHECKS)
            or v.get('verdict')!='admit_frozen_final_qualification' or not isinstance(v.get('reviewer'),str)
            or not v['reviewer'] or v['reviewer'] in seen):raise ValueError('fresh exact-packet independent admission required')
        seen.add(v['reviewer'])
    if _bounded_json(owner)!={'schema':'nca-ik54-frozen-final-owner/v1','declaration_file_sha256':digest,'execute_final':True}:
        raise ValueError('exact-packet owner execution record required')
    return d


def verify_seal_metadata(root,expected,*,evaluation_root,reviews,owner):
    d=check_admissions(evaluation_root,reviews,owner);root=Path(root)
    if root.resolve()!=Path(d['suite_root']).resolve() or expected!=d['suite_seal']:raise ValueError('new suite locator/seal differs')
    evaluation=Path(evaluation_root).resolve();suite=root.resolve()
    if evaluation==suite or evaluation in suite.parents or suite in evaluation.parents:raise ValueError('suite/output overlap')
    names=('declaration.json','geometry.json','seal.json','supervisor.json','isolation.json','attempts.jsonl','sealed-final.tar.gz')
    if root.is_symlink() or any((root/n).is_symlink() for n in names):raise ValueError('new suite links prohibited')
    if (sha256_file(root/'declaration.json')!=d['suite_declaration_file_sha256'] or sha256_file(root/'geometry.json')!=d['suite_geometry_file_sha256']
        or sha256_file(root/'process-exit.json')!=d['suite_process_exit_file_sha256']
        or custodian.inspect(root)!=expected):raise ValueError('new sealed suite custody differs')
    return expected


def open_suite(root, evaluation_root, reviews, owner, seal):
    """Require live admission and reservations, then mark before archive access."""
    evaluation_root=Path(evaluation_root)
    declaration=check_admissions(evaluation_root,reviews,owner)
    started=_bounded_json(evaluation_root/'started.json')
    digest=sha256_file(evaluation_root/'declaration.json')
    if started.get('declaration_file_sha256')!=digest:
        raise ValueError('exact single-use start claim required')
    for seed in SEEDS:
        expected={'seed':seed,'declaration_file_sha256':digest,'evaluation_calls':EVALUATIONS_PER_SEED,
                  'optimizer_calls':0,'wall_seconds':WALL_SECONDS,'attempt':1,'graph_construction_calls':16}
        if _bounded_json(evaluation_root/f'reservation-{seed}.json')!=expected:
            raise ValueError('all three full write-ahead reservations required')
    root=Path(root)
    if root.resolve()!=Path(declaration['suite_root']).resolve() or seal!=declaration['suite_seal']:
        raise ValueError('suite root or seal differs from admitted declaration')
    verify_seal_metadata(root,seal,evaluation_root=evaluation_root,reviews=reviews,owner=owner)
    frozen=declaration['frozen_candidates']
    opening={'schema':'nca-ik-final-opening/v1','experiment':frozen['experiment'],
        'frozen_candidates_sha256':[c['checkpoint_sha256'] for c in frozen['candidates']],
        'suite_identity_sha256':seal['archive_sha256'],'dataset_manifest_sha256':seal['sha256'],
        'registry_id':seal['registry_id'],'future_experiments_status':'development_only'}
    opening['sha256']=sha256_json(opening)
    receipt={'declaration_file_sha256':digest,
             'frozen':frozen,'opening':opening,'created_utc':utc_now()}
    immutable_json(root/'opened.json',receipt)
    if _bounded_json(root/'opened.json')!=receipt:raise ValueError('opening readback differs')
    raw=(root/'sealed-final.tar.gz').read_bytes()
    if len(raw)!=seal['archive_bytes'] or hashlib.sha256(raw).hexdigest()!=seal['archive_sha256']:
        raise ValueError('opened suite archive bytes differ; retain consumed marker')
    from nca.ik.data import GeometricSample, iter_tasks, validate_sample, validate_dataset
    config=FrozenFinalConfig();samples=[]
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
        members=archive.getmembers()
        if [m.name for m in members]!=[f'final-{i:05d}.jsonl' for i in range(46)]:
            raise ValueError('final archive member schedule differs')
        for member in members:
            if not member.isfile() or member.size>16*1024**2:raise ValueError('final member exceeds bound')
            for line in archive.extractfile(member):
                sample=GeometricSample.from_dict(strict_json_loads(line,max_bytes=256*1024))
                validate_sample(sample,config);samples.append(sample)
    if len(samples)!=2940 or [x.task.to_dict() for x in samples]!=[x.to_dict() for x in iter_tasks(config,'final')]:
        raise ValueError('final task ordering differs')
    validate_dataset(samples,config,require_complete=True)
    return tuple(samples),config,opening


def evaluate_seed(seed, declaration, samples, config, opening, ready):
    """Load only model state, score six declared controls and matched CPU timing."""
    from nca.evaluation.ik_training import (COMPARATORS, model_predictor, reference_predictors,
        collect_predictions, score_raw, measure_cpu_latency, apply_cpu_utility, _accuracy, _refusal)
    context,trained,initial,old=ready
    if context.torch.get_num_threads()!=1:raise ValueError('one-thread CPU timing required')
    entry=declaration['candidates'][str(seed)]
    predictors={'trained_raw_reference':model_predictor(context,trained),
                'matched_untrained_initialization':model_predictor(context,initial),
                **reference_predictors(config),'archived_trained_v1':model_predictor(context,old)}
    wrapped,counts=_counter_predictors(predictors,EVALUATIONS_PER_SEED)
    ids=[x.task.id for x in samples];reports={}
    for name in COMPARATORS:
        predictions=collect_predictions(samples,wrapped[name])
        if counts['total']>EVALUATIONS_PER_SEED:raise ValueError('final invocation cap exceeded')
        reports[name]=score_raw(samples,predictions,config,partition='final',expected_task_ids=ids,
            final_opening=opening,frozen_candidates=declaration['frozen_candidates'],seed=seed,comparator=name)
    archived=score_raw(samples,collect_predictions(samples,wrapped['archived_trained_v1']),config,
        partition='final',expected_task_ids=ids,final_opening=opening,
        frozen_candidates=declaration['frozen_candidates'],seed=seed)
    archived=dict(archived,comparator='archived_trained_v1',diagnostic_only=True,
                  frozen_checkpoint_sha256=entry['archived']['sha256'],suite_opening_sha256=opening['sha256'])
    first,control=reports[COMPARATORS[0]],reports[COMPARATORS[1]]
    a,b=first['selection_metric'],control['selection_metric']
    contribution='inconclusive' if a is None or b is None else ('pass' if a<=.5*b else 'no_go')
    comparison={'schema':'nca-ik-direct-comparison/v1','seed':seed,'partition':'final',
        'config_sha256':config.sha256,'schedule_sha256':first['schedule_sha256'],'comparators':reports,
        'accuracy':_accuracy(first),'learned_contribution':contribution,'practical_utility':'inconclusive',
        'practical_utility_reason':'matched_complete_decision_cpu_latency_is_unmeasured','qualified':False}
    counts['phase']='timing'
    nominal=tuple(x.task.inputs for x in samples if x.task.state.kind!='stress')
    timing=measure_cpu_latency(nominal,{k:wrapped[k] for k in COMPARATORS},context=context,
                               environment={**context.report,'inference_route':inference_route(),'trained_and_initial':'frozen_graph_v1','archived':'eager_v1'},archived_predictor=wrapped['archived_trained_v1'])
    archived_timing=timing['comparators'].pop('archived_trained_v1')
    comparison=apply_cpu_utility(comparison,timing)
    comparison['archived_trained_v1']={'checkpoint_sha256':entry['archived']['sha256'],
                                        'score':archived,'cpu_latency':archived_timing}
    guarded=sum(_refusal(x.task) is not None for x in samples)
    if (counts['total']>EVALUATIONS_PER_SEED or sum(counts['timing'].values())!=60600
            or set(counts['final'])!=set(predictors) or set(counts['timing'])!=set(predictors)
            or any(counts['final'][k]!=len(samples)-guarded for k in predictors)
            or any(counts['timing'][k]!=10100 for k in predictors)):
        raise ValueError('actual decision and timing counts differ from frozen schedule')
    return comparison,{'scheduled_final_decisions':17640,'guarded_final_refusals':6*guarded,
        'actual_final_invocations':sum(counts['final'].values()),'timing_invocations':60600,
        'actual_predictor_invocations':counts['total'],'counts':counts,
        'optimizer_calls':0,'reservation_calls':EVALUATIONS_PER_SEED}


def run(args):
    from threading import Event,Thread
    from nca.ik.process_limits import memory_limit
    from nca.train.ik_qualification_v4 import invocation_start, hard_wall_watchdog
    from nca.train.ik_matched_supervisor import board_sample
    from nca.train.ik_dev_sweep import _host_free_commit, append
    from nca.train.ik_gpu_ownership import acquire,release
    started=invocation_start()
    memory_limit(8192,kill_children_on_exit=True)
    hard_stop=Event();timer=Thread(target=hard_wall_watchdog,args=(hard_stop,started,WALL_SECONDS-EXIT_MARGIN_SECONDS),daemon=True);timer.start()
    root=Path(args.root);d=check_admissions(root,args.review,args.authorization)
    from nca.ik.declaration import runtime_expectation
    if runtime_expectation()!=d['runtime']:
        raise ValueError('admitted runtime changed before final access')
    if (root/'started.json').exists():raise ValueError('single final attempt already claimed')
    board=board_sample()
    if board['temperature']>80 or _host_free_commit()<32*GIB or shutil.disk_usage(root).free<16*GIB:
        raise ValueError('final startup resource headroom insufficient')
    # After genuine admission but before spending the single attempt, confirm
    # all train/tuning identities and sealed-suite metadata. No archive bytes
    # or final rows are read here. Opening is separately rechecked atomically.
    seal=d['suite_seal']
    training,candidates,frozen=training_choices(d['training_root'],d['polar_declaration'],d['archived_root'],seal,full=False)
    if candidates!=d['candidates'] or frozen!=d['frozen_candidates']:
        raise ValueError('training choices or suite provenance changed after admission')
    if (sha256_file(Path(d['proof_root'])/'proof.json')!=d['proof_file_sha256']
        or sha256_file(Path(d['proof_root'])/'process-exit.json')!=d['proof_process_exit_file_sha256']):raise ValueError('frozen proof bytes changed')
    verify_proof(d['proof_root'],d['source'],candidates)
    from nca.train.ik_corpus import read_polar_followup_corpus
    from nca.ik.polar import fit_training_normalization
    corpus=read_polar_followup_corpus(training['corpus'],training['registry'],source_sha256=training['source']['source_sha256'])
    if corpus.identity!=training['corpus_identity'] or fit_training_normalization(corpus).to_dict()!=d['normalization']:
        raise ValueError('training-only corpus identity changed')
    # Model-format or environment incompatibility must fail before consuming
    # final custody. Loading these exact weights performs no inference/optimizer.
    from nca.ik.device import configure_device
    import torch
    torch.set_num_interop_threads(1)
    for seed in SEEDS:
        if runtime_expectation()!=d['runtime']:raise ValueError('runtime changed during model preflight')
        context=configure_device('cpu',seed,source_sha256=d['source']['source_sha256'])
        models=load_models(seed,d,corpus,context)
        if torch.cuda.is_initialized():raise ValueError('CPU-only final path unexpectedly initialized CUDA')
        del models,context
    verify_seal_metadata(d['suite_root'],seal,evaluation_root=root,reviews=args.review,owner=args.authorization)
    board=board_sample()
    if board['temperature']>80 or _host_free_commit()<32*GIB or shutil.disk_usage(root).free<16*GIB:
        raise ValueError('final post-preflight resource headroom insufficient')
    owner={'kind':'ik54_frozen_final_v1','claim':uuid.uuid4().hex,'declaration_file_sha256':sha256_file(root/'declaration.json')}
    acquire(owner)
    immutable_json(root/'ownership.json',owner)
    digest=sha256_file(root/'declaration.json')
    immutable_json(root/'started.json',{'declaration_file_sha256':digest,'attempt':1,'owner':owner,'started_utc':utc_now()})
    stop=Event()
    def watch():
        try:
            with (root/'board.jsonl').open('x',encoding='utf-8') as log:
                while not stop.wait(2):
                    sample=board_sample();sample['host_free_commit_bytes']=_host_free_commit();sample['disk_free_bytes']=shutil.disk_usage(root).free
                    if sample['temperature']>80 or sample['host_free_commit_bytes']<4*GIB or sample['disk_free_bytes']<4*GIB:
                        raise ValueError('final resource envelope breached')
                    append(log,sample)
        except BaseException as exc:
            try:immutable_json(root/'resource-failure.json',{'reason':str(exc),'elapsed':time.monotonic()-started})
            finally:os._exit(2)
    watcher=Thread(target=watch,daemon=True);watcher.start();success=False
    try:
        for seed in SEEDS:
            immutable_json(root/f'reservation-{seed}.json',{'seed':seed,'declaration_file_sha256':digest,
                'evaluation_calls':EVALUATIONS_PER_SEED,'optimizer_calls':0,'wall_seconds':WALL_SECONDS,'attempt':1,'graph_construction_calls':16})
        _,current_candidates,current_frozen=training_choices(d['training_root'],d['polar_declaration'],d['archived_root'],seal,full=False)
        if current_candidates!=candidates or current_frozen!=frozen:
            raise ValueError('frozen training choices changed before opening')
        ready={}
        for seed in SEEDS:
            ready[seed],construction=construct(seed,d,corpus)
            immutable_json(root/f'construction-{seed}.json',construction)
        if runtime_expectation()!=d['runtime']:raise ValueError('runtime changed before final opening')
        observed=verify_seal_metadata(d['suite_root'],seal,evaluation_root=root,reviews=args.review,owner=args.authorization)
        if time.monotonic()-started>=WALL_SECONDS-FINALIZATION_RESERVE_SECONDS:
            raise TimeoutError('insufficient paid time to open final suite')
        samples,config,opening=open_suite(d['suite_root'],root,args.review,args.authorization,observed)
        reports={}
        for seed in SEEDS:
            if runtime_expectation()!=d['runtime']:raise ValueError('runtime changed during final evaluation')
            if time.monotonic()-started>=WALL_SECONDS-FINALIZATION_RESERVE_SECONDS:
                raise TimeoutError('insufficient paid time for next final seed')
            comparison,receipt=evaluate_seed(seed,d,samples,config,opening,ready[seed])
            out=root/f'final-comparison-{seed}.json';file_hash=_write_report(out,comparison)
            receipt.update(seed=seed,report_file_sha256=file_hash,report_semantic_sha256=sha256_json(comparison),
                           accuracy=comparison['accuracy'],learned_contribution=comparison['learned_contribution'],
                           practical_utility=comparison['practical_utility'])
            _write_report(root/f'completion-{seed}.json',receipt);reports[seed]=comparison
        from nca.evaluation.ik_training import aggregate_comparisons
        aggregate=aggregate_comparisons(reports)
        _write_report(root/'aggregate.json',aggregate)
        # Re-authenticate selected checkpoint bytes after final scoring; no model calls.
        for seed in SEEDS:
            entry=d['candidates'][str(seed)];seed_root=Path(d['training_root'])/f'seed-{seed}'
            for key in ('selected','initial'):_verify_checkpoint(seed_root,entry[key],entry['training_identity'])
        _,post_candidates,post_frozen=training_choices(d['training_root'],d['polar_declaration'],d['archived_root'],seal,full=False)
        if post_candidates!=candidates or post_frozen!=frozen:
            raise ValueError('training custody or archived diagnostic changed during final scoring')
        if time.monotonic()-started>=WALL_SECONDS-EXIT_MARGIN_SECONDS:raise TimeoutError('final wall allocation exhausted')
        result={'schema':'nca-ik54-frozen-final-terminal/v1','declaration_file_sha256':digest,'opening_sha256':opening['sha256'],
                'aggregate_file_sha256':sha256_file(root/'aggregate.json'),
                'receipts':{str(s):sha256_file(root/f'completion-{s}.json') for s in SEEDS},
                'prepublication_elapsed_seconds':time.monotonic()-started,'optimizer_calls':0,
                'reserved_evaluation_calls_per_seed':EVALUATIONS_PER_SEED,'graph_construction_calls_per_seed':16,'inference_route':inference_route(),'attempts':1,'status':'completed'}
        terminal_hash=immutable_json(root/'terminal.json',result)
        if _bounded_json(root/'terminal.json')!=result:raise ValueError('terminal readback differs')
        immutable_json(root/'sealed.json',{'terminal_file_sha256':terminal_hash})
        if _bounded_json(root/'sealed.json')!={'terminal_file_sha256':terminal_hash}:
            raise ValueError('terminal seal readback differs')
        stop.set();watcher.join(timeout=15)
        if watcher.is_alive():raise TimeoutError('resource watcher failed to stop')
        release(owner)
        completion={'terminal_file_sha256':terminal_hash,'ownership_released':True}
        immutable_json(root/'complete.json',completion)
        if _bounded_json(root/'complete.json')!=completion:raise ValueError('completion readback differs')
        success=True
    except BaseException as exc:
        immutable_json(root/'failure.json',{'error_type':type(exc).__name__,'message':str(exc),
                       'elapsed_seconds':time.monotonic()-started,'retry_or_resume':False})
        raise
    finally:
        stop.set();watcher.join(timeout=1)
        # Failure retains any still-live ownership; an opened suite cannot be reset.


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for verb in ('prove','prepare'):
        p=sub.add_parser(verb);p.add_argument('--root',required=True);p.add_argument('--training',required=True)
        p.add_argument('--polar-declaration',required=True);p.add_argument('--archived',required=True)
        if verb=='prepare':p.add_argument('--suite',required=True);p.add_argument('--proof',required=True)
    for verb in ('check-admissions','run'):
        p=sub.add_parser(verb);p.add_argument('--root',required=True);p.add_argument('--review',action='append',required=True);p.add_argument('--authorization',required=True)
    a=parser.parse_args()
    if a.command=='prove':prove(a)
    elif a.command=='prepare':prepare(a)
    elif a.command=='check-admissions':check_admissions(a.root,a.review,a.authorization)
    else:run(a)


if __name__=='__main__':main()

# Copyright 2026 techrote. All rights reserved.
