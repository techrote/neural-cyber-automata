"""Single-use IK54 final qualification, bound to the completed v4 candidates.

Preparation never touches the sealed suite. Actual final access requires the
frozen declaration, two independent admissions and the owner's execution record.
This path cannot train, resume or modify the prior qualification ledgers.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import math
import os
from pathlib import Path
import shutil
import tarfile
import time
import uuid

from nca.contracts import canonical_json, sha256_json, strict_json_loads
from nca.runs import immutable_json, sha256_file, utc_now
from nca.train.ik_experiment import _source, _read, _titan_visibility
from nca.train.ik_matched_prepare import TITAN_FINGERPRINT, GIB
from nca.train.ik_qualification_v4 import SEEDS, identity as training_identity, validate_terminal, verify_selected
from nca.train.ik_run_state_v2 import IKRunLedger, _record, _verify_checkpoint
from nca.train.ik_followup import archived_paths
from nca.ik.fresh_final import FreshFinalConfig
from nca.evaluation.ik_training import freeze_candidates

TRAIN_DECLARATION = '5e0ce7c4e36bbcbebe7543f8d230c5df166cf33d291a01d847efc6ed9ad2c4f8'
TRAIN_RESULT = 'bff8553856aa7c9b98fe5fbbd3a51399e18396ab1612eb559b4078baa7f6d7f7'
POLAR_DECLARATION = 'f36137d484ec54fb50cbf466cc7421785e2d73136d2f30940e35a02bc064a3c0'
TRAIN_SUMMARY = '514e86e65d4d756df03f42b3ae70e7c76cf6e8b0db6ce9451350275958c04983'
ACCEPTANCE_SHA = 'cea8642ed41313c9c6fc8f27f288e90aa8743146e5f61a2f0ab2ea86b874aa8b'
SEAL_SHA = '505271bf3b96a25cde964b1e8cb7f3ed9616f8328118715a6bceec4b272a9ece'
ARCHIVE_SHA = '89e118be3cc9f1ed358198cd6448fe813a138281c42e34b2add419b1e1b7bfff'
ISOLATION_SHA = '08d1b4402cea8c2dd09cd1acafa9f2cca29abbe1ba05ae18050750082356d3db'
WALL_SECONDS = 86400
EXIT_MARGIN_SECONDS = 30
EVALUATIONS_PER_SEED = 78240
REPORT_MAX_BYTES = 128*1024**2
FINALIZATION_RESERVE_SECONDS = 7200
CHECKS = ('exact_head_ci','source_and_cpu_checks','candidate_custody_and_tuning',
          'sealed_suite_provenance','accounting_and_no_retry','scoring_and_resource_feasibility')

COMPARATOR_NAMES = ('trained_raw_reference','matched_untrained_initialization',
                    'learned_disabled_current_reference','independent_numerical_reference',
                    'independent_analytic_reference','archived_trained_v1')


def allocation():
    return {'attempts':1,'optimizer_calls':0,'evaluation_calls_per_seed':EVALUATIONS_PER_SEED,
            'seeds':list(SEEDS),'wall_seconds':WALL_SECONDS,'exit_margin_seconds':EXIT_MARGIN_SECONDS,
            'cpu_threads':1,'process_commit_bytes':8*GIB,'board_cap_bytes':20*GIB,
            'startup_host_free_bytes':32*GIB,'host_free_floor_bytes':4*GIB,
            'startup_disk_free_bytes':16*GIB,'disk_free_floor_bytes':4*GIB,
            'report_max_bytes':REPORT_MAX_BYTES,'work_stop_reserve_seconds':FINALIZATION_RESERVE_SECONDS}


def _bounded_json(path, maximum=REPORT_MAX_BYTES):
    value = _read(path, maximum)
    if not isinstance(value, dict):
        raise ValueError('bounded JSON object required')
    return value


def _write_report(path, value):
    if len((canonical_json(value) + '\n').encode()) > REPORT_MAX_BYTES:
        raise ValueError('bounded final report exceeded')
    digest = immutable_json(path, value)
    if sha256_file(path) != digest or _bounded_json(path) != value:
        raise ValueError('final report readback differs')
    return digest


def _pinned(path, expected, maximum=REPORT_MAX_BYTES):
    if sha256_file(path) != expected:
        raise ValueError('pinned artifact bytes differ')
    return _bounded_json(path, maximum)


def _external(root, *inputs):
    root = Path(os.path.abspath(os.path.normpath(root)))
    for path in inputs:
        path = Path(os.path.abspath(os.path.normpath(path)))
        if root == path or root in path.parents or path in root.parents:
            raise ValueError('evaluation output overlaps an input')
    root = root.resolve()
    checkout = Path(__file__).resolve().parents[3]
    if root.exists() or root == checkout or checkout in root.parents:
        raise ValueError('new external evaluation allocation required')
    return root


def training_candidates(training, polar_declaration, archived, *, full=True):
    """Authenticate retained tuning-only choices without reading any final data."""
    training = Path(training)
    declaration = _pinned(training/'declaration.json', TRAIN_DECLARATION, 16*1024**2)
    group = _pinned(training/'group-result.json', TRAIN_RESULT, 16*1024**2)
    summary=_pinned(training.parent/'PART_B_SUMMARY.json',TRAIN_SUMMARY,16*1024**2)
    exit_receipt=summary['process_exit_observation']
    if (summary.get('group_result_sha256')!=TRAIN_RESULT or exit_receipt.get('exit_code')!=0
            or exit_receipt.get('group_failure_exists') is not False
            or exit_receipt.get('resource_cap_failure_exists') is not False
            or exit_receipt.get('ownership_still_held') is not False):
        raise ValueError('verified original process exit and released ownership required')
    if (declaration['source']['revision'] != '7c65e5920d577695b3012b2b3c4391574329c6fc'
            or declaration['final_calls'] != 0 or group.get('final_calls') != 0
            or group.get('all_tuning_gates_pass') is not True
            or [x['seed'] for x in group['outcomes']] != list(SEEDS)):
        raise ValueError('completed exact training/tuning result required')
    polar = _pinned(polar_declaration, POLAR_DECLARATION, 16*1024**2)
    seal = polar['fresh_final']
    if (polar['sha256'] != sha256_json({k:v for k,v in polar.items() if k!='sha256'})
            or seal['sha256'] != SEAL_SHA or seal['archive_sha256'] != ARCHIVE_SHA
            or seal['final_state'] != 'sealed' or seal['rows'] != 2940):
        raise ValueError('published historical suite provenance differs')
    archived_refs = archived_paths(archived)
    if archived_refs != polar['archived_candidates']:
        raise ValueError('historical diagnostic controls differ')
    choices, candidates = {}, {}
    for outcome in group['outcomes']:
        seed=outcome['seed']; state=outcome['ledger']
        ledger=IKRunLedger.open(training/f'seed-{seed}', expected_identity=training_identity(declaration,seed),
                                expected_allocation=declaration['allocations'])
        if state['identity'] != ledger.identity or state['allocation'] != ledger.allocation:
            raise ValueError('terminal group ledger identity differs')
        validate_terminal(state, declaration['allocations'])
        if full and ledger.inspect()!=state:
            raise ValueError('current sealed training ledger differs from supervisor result')
        selected=verify_selected(ledger,state,declaration)
        if selected != outcome['selected'] or selected['gate'] != 'pass':
            raise ValueError('selected tuning gate or provenance differs')
        attempt=ledger.root/'attempts/attempt-0001'
        finish=_record(attempt/'finish.json')
        if (finish['sha256'] != state['attempts'][0]['record_sha256']
                or _bounded_json(attempt/'sealed.json') != {'finish_sha256':finish['sha256']}
                or finish['record']['status'] != 'completed'):
            raise ValueError('original terminal finish or seal differs')
        event=attempt/'event-000002.json'
        item=finish['record']['inventory']['event-000002.json']
        if event.stat().st_size != item['bytes'] or sha256_file(event) != item['sha256']:
            raise ValueError('epoch-zero receipt differs from sealed inventory')
        event_record=_record(event)['record']
        if event_record['sequence'] != 2 or event_record['kind'] != 'complete':
            raise ValueError('initial checkpoint event differs')
        initial=event_record['payload']['receipt']
        _verify_checkpoint(ledger.root,initial,ledger.identity)
        if (initial['state']['completed_optimizer_steps'] != 0
                or initial['state']['extra_state'].get('completed_epochs') != 0):
            raise ValueError('actual epoch-zero initial model required')
        history=state['latest_checkpoint']['state']['extra_state']['tuning_history']
        if full:
            identities=set()
            for row in history:
                report_check=_pinned(ledger.root/row['report'],row['report_file_sha256'],8*1024**2)
                metric=report_check.get('selection_metric')
                if (report_check.get('partition')!='tuning' or report_check.get('comparator')!='trained_raw_reference'
                    or report_check.get('complete') is not True or not report_check['nominal']['complete_raw_outputs']
                    or type(metric) not in (int,float) or not math.isfinite(metric) or metric<0
                    or metric!=report_check['nominal']['p95_error_cells'] or metric!=row['metric']):
                    raise ValueError('complete authenticated tuning metric required at every checkpoint')
                identities.add((report_check['config_sha256'],report_check['schedule_sha256']))
            if len(identities)!=1:raise ValueError('tuning configuration or schedule changed')
        step=selected['checkpoint']['state']['completed_optimizer_steps']
        event_name=f"event-{2+2*step+4*(selected['epoch']//5):06d}.json"
        selected_event=attempt/event_name;event_item=finish['record']['inventory'][event_name]
        selected_record=_record(selected_event)['record']
        if (selected_event.stat().st_size!=event_item['bytes'] or sha256_file(selected_event)!=event_item['sha256']
                or selected_record['sequence']!=int(event_name[6:12]) or selected_record['kind']!='complete'
                or selected_record['payload']['receipt']!=selected['checkpoint']):
            raise ValueError('selected checkpoint is not its sealed completion event')
        report=_pinned(ledger.root/selected['report'],selected['report_file_sha256'],8*1024**2)
        if step != selected['epoch']*43 or report['selection_metric'] != selected['metric']:
            raise ValueError('selected checkpoint step/report differs')
        choices[seed]={'step':step,'checkpoint_sha256':selected['checkpoint']['reference']['sha256'],
                       'tuning_report_sha256':sha256_json(report),'selection_metric':selected['metric'],
                       'config_sha256':FreshFinalConfig().sha256,
                       'tuning_schedule_sha256':report['schedule_sha256']}
        candidates[str(seed)]={'selected':selected['checkpoint'], 'initial':initial,
                               'training_identity':ledger.identity, 'selected_epoch':selected['epoch'],
                               'selected_report_file_sha256':selected['report_file_sha256'],
                               'archived':archived_refs[str(seed)]}
    frozen=freeze_candidates(choices,experiment='ik54-final-v1',dataset_manifest_sha256=seal['sha256'],
                             final_suite_identity_sha256=seal['archive_sha256'],registry_id=seal['registry_id'])
    return declaration, seal, candidates, frozen


def prepare(args):
    root=_external(args.root,args.training,args.suite,args.archived,Path(args.polar_declaration).parent)
    _titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    source=_source(Path(__file__).resolve().parents[3])
    training,seal,candidates,frozen=training_candidates(args.training,args.polar_declaration,args.archived)
    from nca.ik.declaration import runtime_expectation
    from nca.train.ik_corpus import read_polar_followup_corpus
    from nca.ik.polar import fit_training_normalization
    corpus=read_polar_followup_corpus(training['corpus'],training['registry'],
                                      source_sha256=training['source']['source_sha256'])
    if (corpus.identity!=training['corpus_identity'] or fit_training_normalization(corpus).to_dict()!=training['normalization']
            or runtime_expectation()!=training['runtime']):
        raise ValueError('retained corpus, normalization or environment changed')
    declaration={'schema':'nca-ik54-final-declaration/v1','id':'ik54-final-v1',
        'source':source,'training_root':str(Path(args.training).resolve()),
        'polar_declaration':str(Path(args.polar_declaration).resolve()),
        'archived_root':str(Path(args.archived).resolve()),'suite_root':os.path.abspath(os.path.normpath(args.suite)),
        'training_declaration_file_sha256':TRAIN_DECLARATION,'training_result_file_sha256':TRAIN_RESULT,
        'polar_declaration_file_sha256':POLAR_DECLARATION,'suite_seal':seal,'candidates':candidates,
        'frozen_candidates':frozen,'normalization':training['normalization'],'runtime':training['runtime'],
        'gpu_visibility_sha256':TITAN_FINGERPRINT,'acceptance_file_sha256':training['acceptance_file_sha256'],
        'allocation':allocation(),'comparators':list(COMPARATOR_NAMES),
        'final_state':'sealed_unverified_until_admitted_execution','retry_or_resume':False,
        'selection':'original_tuning_only_no_reselection','execution_authorized':False,
        'created_utc':utc_now()}
    root.mkdir(parents=True,exist_ok=False)
    immutable_json(root/'declaration.json',declaration)
    immutable_json(root/'frozen-candidates.json',frozen)
    immutable_json(root/'admission-template.json',{'schema':'nca-ik54-final-admission/v1',
        'head':source['revision'],'declaration_file_sha256':sha256_file(root/'declaration.json'),
        'reviewer':None,'independent_non_author':False,'checks':{k:False for k in CHECKS},'verdict':'pending'})
    print(sha256_file(root/'declaration.json'))


def check_admissions(root, reviews, owner):
    root=Path(root);d=_bounded_json(root/'declaration.json',16*1024**2);source=_source(Path(__file__).resolve().parents[3])
    digest=sha256_file(root/'declaration.json')
    if (d.get('schema')!='nca-ik54-final-declaration/v1' or d.get('id')!='ik54-final-v1'
            or d.get('source')!=source or canonical_json(d.get('allocation'))!=canonical_json(allocation())
            or d.get('comparators')!=list(COMPARATOR_NAMES)
            or d.get('gpu_visibility_sha256')!=TITAN_FINGERPRINT
            or d.get('acceptance_file_sha256')!=ACCEPTANCE_SHA
            or d.get('final_state')!='sealed_unverified_until_admitted_execution'
            or d.get('selection')!='original_tuning_only_no_reselection'
            or d.get('execution_authorized') is not False or d.get('retry_or_resume') is not False
            or _bounded_json(root/'frozen-candidates.json')!=d.get('frozen_candidates')):
        raise ValueError('final declaration/source/freeze differs')
    seen=set()
    if len(reviews)!=2:raise ValueError('two fresh independent admissions required')
    for p in reviews:
        v=_bounded_json(p)
        if (v.get('schema')!='nca-ik54-final-admission/v1' or v.get('head')!=source['revision']
            or v.get('declaration_file_sha256')!=digest or v.get('independent_non_author') is not True
            or set(v.get('checks',{}))!=set(CHECKS) or any(v['checks'][k] is not True for k in CHECKS)
            or v.get('verdict')!='admit_final_qualification' or not isinstance(v.get('reviewer'),str)
            or not v['reviewer'] or v['reviewer'] in seen):
            raise ValueError('fresh exact-packet independent admission required')
        seen.add(v['reviewer'])
    expected={'schema':'nca-ik54-final-owner/v1','declaration_file_sha256':digest,'execute_final':True}
    if _bounded_json(owner)!=expected:raise ValueError('exact-packet owner execution record required')
    from nca.ik.declaration import runtime_expectation
    if runtime_expectation()!=d.get('runtime'):raise ValueError('admitted runtime changed')
    return d


def verify_seal_metadata(root, expected, *, evaluation_root, reviews, owner):
    """Post-admission metadata only; opening and archive bytes remain untouched."""
    declaration=check_admissions(evaluation_root,reviews,owner)
    root=Path(root)
    if root.resolve()!=Path(declaration['suite_root']).resolve() or expected!=declaration['suite_seal']:
        raise ValueError('sealed suite identity differs from admission')
    evaluation=Path(evaluation_root).resolve();suite=root.resolve()
    if evaluation==suite or evaluation in suite.parents or suite in evaluation.parents:
        raise ValueError('resolved suite and evaluation allocation overlap')
    if root.is_symlink() or any((root/name).is_symlink() for name in
        ('declaration.json','seal.json','supervisor.json','isolation.json','attempts.jsonl','sealed-final.tar.gz')):
        raise ValueError('sealed suite metadata and archive cannot be symbolic links')
    declaration=_bounded_json(root/'declaration.json');seal=_bounded_json(root/'seal.json');supervisor=_bounded_json(root/'supervisor.json')
    if (seal!=expected or seal.get('sha256')!=sha256_json({k:v for k,v in seal.items() if k!='sha256'})
            or declaration.get('sha256')!=sha256_json({k:v for k,v in declaration.items() if k!='sha256'})
            or declaration['sha256']!=seal['declaration_sha256'] or declaration.get('geometry')!=seal['geometry']
            or seal['sha256']!=SEAL_SHA or seal['archive_sha256']!=ARCHIVE_SHA
            or seal['independent_isolation_file_sha256']!=ISOLATION_SHA
            or sha256_file(root/'isolation.json')!=ISOLATION_SHA
            or sha256_file(root/'attempts.jsonl')!=seal['attempt_journal_sha256']
            or (root/'supervisor-overrun.json').exists() or (root/'opened.json').exists()
            or supervisor.get('complete') is not True or supervisor.get('returncode')!=0
            or supervisor.get('timed_out') is not False or supervisor.get('whole_process_seconds',900)>=900
            or supervisor.get('declaration_sha256')!=declaration['sha256']
            or supervisor.get('seal_sha256')!=seal['sha256'] or seal.get('final_state')!='sealed'
            or seal.get('teacher_attempts')!=2940 or seal.get('reserved_teacher_iterations')!=188160
            or not 0<=seal.get('teacher_iterations',-1)<=188160 or seal.get('candidate_conditioned_filtering') is not False
            or seal.get('peak_process_bytes',float('inf'))>512*1024**2
            or seal['rows']!=2940 or seal['archive_bytes']!=832639
            or (root/'sealed-final.tar.gz').stat().st_size!=seal['archive_bytes']):
        raise ValueError('historical sealed suite metadata differs or is already opened')
    return seal


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
                  'optimizer_calls':0,'wall_seconds':WALL_SECONDS,'attempt':1}
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
    config=FreshFinalConfig();samples=[]
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


def _counter_predictors(predictors, limit):
    """Count actual permitted predictor invocations before each underlying call."""
    counts={'phase':'final','final':{},'timing':{},'total':0}
    wrapped={}
    for name,predictor in predictors.items():
        def call(inputs, name=name, predictor=predictor):
            counts['total']+=1
            if counts['total']>limit:raise ValueError('predictor invocation cap exceeded')
            phase=counts['phase'];counts[phase][name]=counts[phase].get(name,0)+1
            return predictor(inputs)
        wrapped[name]=call
    return wrapped,counts


def load_models(seed, declaration, corpus, context):
    """Authenticate and load all three model states without optimizer restoration."""
    from nca.ik.polar import PolarNormalization
    from nca.ik.models_v3 import build_model as build_v3
    from nca.ik.models import build_model as build_v1
    from nca.ik.checkpoint import load_checkpoint
    entry=declaration['candidates'][str(seed)]
    training_root=Path(declaration['training_root'])/f'seed-{seed}'
    normalization=PolarNormalization.from_dict(declaration['normalization'])
    def v3(receipt):
        p=training_root/receipt['path'];_verify_checkpoint(training_root,receipt,entry['training_identity'])
        payload=load_checkpoint(p,expected_sha256=receipt['reference']['sha256'],expected_metadata=entry['training_identity'])
        model=build_v3(context,normalization,head='tanh',hidden_width=64)
        model.load_state_dict(payload['model'],strict=True);model.eval();return model
    trained,initial=v3(entry['selected']),v3(entry['initial'])
    archived_root=Path(declaration['archived_root'])/f'seed-{seed}'
    archived_path=archived_root/entry['archived']['path']
    marker=_bounded_json(archived_path.with_name(archived_path.name+'.json'))['checkpoint']
    payload=load_checkpoint(archived_path,expected_sha256=entry['archived']['sha256'],expected_metadata=marker['metadata'])
    old=build_v1(context,corpus.normalization);old.load_state_dict(payload['model'],strict=True);old.eval()
    return trained,initial,old


def evaluate_seed(seed, declaration, samples, config, opening, corpus):
    """Load only model state, score six declared controls and matched CPU timing."""
    from nca.ik.device import configure_device
    from nca.evaluation.ik_training import (COMPARATORS, model_predictor, reference_predictors,
        collect_predictions, score_raw, measure_cpu_latency, apply_cpu_utility, _accuracy, _refusal)
    import torch
    context=configure_device('cpu',seed,source_sha256=declaration['source']['source_sha256'])
    torch.set_num_threads(1)
    if torch.get_num_threads()!=1:raise ValueError('one-thread CPU timing required')
    entry=declaration['candidates'][str(seed)]
    trained,initial,old=load_models(seed,declaration,corpus,context)
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
                               environment=context.report,archived_predictor=wrapped['archived_trained_v1'])
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
    training,seal,candidates,frozen=training_candidates(d['training_root'],d['polar_declaration'],d['archived_root'],full=False)
    if seal!=d['suite_seal'] or candidates!=d['candidates'] or frozen!=d['frozen_candidates']:
        raise ValueError('training choices or suite provenance changed after admission')
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
    owner={'kind':'ik54_final_v1','claim':uuid.uuid4().hex,'declaration_file_sha256':sha256_file(root/'declaration.json')}
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
                'evaluation_calls':EVALUATIONS_PER_SEED,'optimizer_calls':0,'wall_seconds':WALL_SECONDS,'attempt':1})
        _,current_seal,current_candidates,current_frozen=training_candidates(d['training_root'],d['polar_declaration'],d['archived_root'],full=False)
        if current_seal!=seal or current_candidates!=candidates or current_frozen!=frozen:
            raise ValueError('frozen training choices changed before opening')
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
            comparison,receipt=evaluate_seed(seed,d,samples,config,opening,corpus)
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
        _,post_seal,post_candidates,post_frozen=training_candidates(d['training_root'],d['polar_declaration'],d['archived_root'],full=False)
        if post_seal!=seal or post_candidates!=candidates or post_frozen!=frozen:
            raise ValueError('training custody or archived diagnostic changed during final scoring')
        if time.monotonic()-started>=WALL_SECONDS-EXIT_MARGIN_SECONDS:raise TimeoutError('final wall allocation exhausted')
        result={'schema':'nca-ik54-final-terminal/v1','declaration_file_sha256':digest,'opening_sha256':opening['sha256'],
                'aggregate_file_sha256':sha256_file(root/'aggregate.json'),
                'receipts':{str(s):sha256_file(root/f'completion-{s}.json') for s in SEEDS},
                'prepublication_elapsed_seconds':time.monotonic()-started,'optimizer_calls':0,
                'reserved_evaluation_calls_per_seed':EVALUATIONS_PER_SEED,'attempts':1,'status':'completed'}
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
    p=sub.add_parser('prepare');p.add_argument('--root',required=True);p.add_argument('--training',required=True)
    p.add_argument('--suite',required=True);p.add_argument('--polar-declaration',required=True);p.add_argument('--archived',required=True)
    for name in ('check-admissions','run'):
        p=sub.add_parser(name);p.add_argument('--root',required=True);p.add_argument('--review',action='append',required=True);p.add_argument('--authorization',required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    elif args.command=='check-admissions':check_admissions(args.root,args.review,args.authorization)
    else:run(args)


if __name__=='__main__':main()

# Copyright 2026 techrote. All rights reserved.
