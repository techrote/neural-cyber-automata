"""Finite CPU inference investigation on authenticated training/tuning only.

No final-suite argument, reader, archive or scoring path is provided. All results
are development evidence. Existing training declarations and weights are read-only.
"""
from __future__ import annotations
import argparse
import cProfile
import io
import os
from pathlib import Path
import pstats
import time

from nca.runs import immutable_json, sha256_file
from nca.train.ik_experiment import _read, _source

SEEDS = (41011, 42013, 43019)
TRAIN_DECLARATION = '5e0ce7c4e36bbcbebe7543f8d230c5df166cf33d291a01d847efc6ed9ad2c4f8'
TRAIN_RESULT = 'bff8553856aa7c9b98fe5fbbd3a51399e18396ab1612eb559b4078baa7f6d7f7'


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU development requires CUDA_VISIBLE_DEVICES empty')
    root=Path(args.root).resolve(); training=Path(args.training).resolve()
    checkout=Path(__file__).resolve().parents[3]
    if root.exists() or checkout in root.parents or root==checkout or training in root.parents or root in training.parents:
        raise ValueError('new external development output required')
    source=_source(checkout)
    if (sha256_file(training/'declaration.json')!=TRAIN_DECLARATION
            or sha256_file(training/'group-result.json')!=TRAIN_RESULT):
        raise ValueError('original training identity differs')
    declared=_read(training/'declaration.json',16*1024**2)
    group=_read(training/'group-result.json',16*1024**2)
    from nca.ik.declaration import runtime_expectation
    root.mkdir(parents=True,exist_ok=False)
    manifest={'schema':'nca-ik-cpu-development/v1','source':source,'training_declaration_file_sha256':TRAIN_DECLARATION,
        'training_result_file_sha256':TRAIN_RESULT,'runtime':runtime_expectation(),'seeds':list(SEEDS),
        'partition':'tuning','variants':['eager','frozen_graph_v1'],'rounds':2,'warmups':100,'measurements':10000,
        'seed_order':['41011,42013,43019','43019,42013,41011'],'variant_order':['eager,frozen','frozen,eager'],
        'max_predictor_calls':1000000,'optimizer_calls':0,'final_calls':0,'wall_seconds':7200,
        'process_commit_bytes':6*1024**3,'cpu_threads':1,'parity_max_joint_radians':2e-5,
        'utility_ratio_limit':0.8,'evidence':'development_only_no_qualification_claim'}
    immutable_json(root/'declaration.json',manifest)
    started=time.monotonic()
    if os.name=='nt':
        from nca.ik.process_limits import memory_limit
        memory_limit(6144,kill_children_on_exit=True)
    from threading import Event, Thread
    from nca.train.ik_qualification_v4 import hard_wall_watchdog
    Thread(target=hard_wall_watchdog,args=(Event(),started,7170),daemon=True).start()
    count=[0]
    def counted(predictor):
        def call(inputs):
            count[0]+=1
            if count[0]>manifest['max_predictor_calls']: raise ValueError('CPU call ceiling exceeded')
            return predictor(inputs)
        return call
    try:
        from nca.train.ik_corpus import read_polar_followup_corpus
        from nca.train.ik_qualification_v4 import identity, verify_selected
        from nca.train.ik_run_state_v2 import IKRunLedger
        from nca.ik.device import configure_device
        from nca.ik.polar import PolarNormalization, fit_training_normalization
        from nca.ik.models_v3 import build_model
        from nca.ik.checkpoint import load_checkpoint
        from nca.ik.inference_v1 import freeze_batch_one
        from nca.evaluation.ik_training import model_predictor, reference_predictors, collect_predictions, score_raw, measure_cpu_latency, _accuracy
        corpus=read_polar_followup_corpus(declared['corpus'],declared['registry'],source_sha256=declared['source']['source_sha256'])
        if corpus.identity!=declared['corpus_identity'] or fit_training_normalization(corpus).to_dict()!=declared['normalization']:
            raise ValueError('training-only normalization or corpus changed')
        normalization=PolarNormalization.from_dict(declared['normalization'])
        nominal=tuple(s.task.inputs for s in corpus.tuning if s.task.state.kind!='stress')
        contexts={}; predictors={}; summaries=[]; checkpoints={}
        for seed in SEEDS:
            context=configure_device('cpu',seed,source_sha256=source['source_sha256']);t=context.torch
            t.set_num_threads(1)
            if seed==SEEDS[0]:t.set_num_interop_threads(1)
            contexts[seed]=context
            ledger=IKRunLedger.open(training/f'seed-{seed}',expected_identity=identity(declared,seed))
            state=next(x['ledger'] for x in group['outcomes'] if x['seed']==seed)
            selected=verify_selected(ledger,state,declared)
            checkpoint=selected['checkpoint'];checkpoints[str(seed)]=checkpoint['reference']['sha256']
            payload=load_checkpoint(ledger.root/checkpoint['path'],expected_sha256=checkpoint['reference']['sha256'],expected_metadata=ledger.identity)
            model=build_model(context,normalization,head='tanh',hidden_width=64)
            model.load_state_dict(payload['model'],strict=True);model.eval()
            initial=build_model(context,normalization,head='tanh',hidden_width=64).eval()
            # Matched untrained weights are one deterministic fresh initialization,
            # identical between runtime variants; no qualification-control claim.
            eager=model_predictor(context,model);frozen=model_predictor(context,freeze_batch_one(context,model))
            first=collect_predictions(corpus.tuning,counted(eager));second=collect_predictions(corpus.tuning,counted(frozen))
            maximum=0.
            for key,a in first.items():
                b=second[key]
                if a.status!=b.status or (a.raw_reference is None)!=(b.raw_reference is None):
                    raise ValueError('refusal parity differs')
                if a.raw_reference is not None:maximum=max(maximum,*(abs(x-y) for x,y in zip(a.raw_reference,b.raw_reference)))
            if maximum>2e-5:raise ValueError('joint parity failed')
            scores=[score_raw(corpus.tuning,p,corpus.config,authenticated_corpus=corpus,seed=seed) for p in (first,second)]
            immutable_json(root/f'parity-{seed}.json',{'seed':seed,'rows':len(corpus.tuning),'max_joint_difference_radians':maximum,
                'checkpoint_sha256':checkpoints[str(seed)],'eager_accuracy':_accuracy(scores[0]),'frozen_accuracy':_accuracy(scores[1]),
                'eager_nominal':scores[0]['nominal'],'frozen_nominal':scores[1]['nominal']})
            predictors[seed]={}
            for name,trained,untrained in (('eager',eager,model_predictor(context,initial)),
                    ('frozen',frozen,model_predictor(context,freeze_batch_one(context,initial)))):
                predictors[seed][name]={k:counted(p) for k,p in {'trained_raw_reference':trained,
                    'matched_untrained_initialization':untrained,**reference_predictors(corpus.config)}.items()}
                profiler=cProfile.Profile();profiler.enable()
                for i in range(300):predictors[seed][name]['trained_raw_reference'](nominal[i%len(nominal)])
                profiler.disable();stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(35)
                (root/f'profile-{seed}-{name}.txt').write_text(stream.getvalue(),encoding='utf-8')
        for round_number in range(2):
            for seed in (SEEDS if round_number==0 else tuple(reversed(SEEDS))):
                for name in (('eager','frozen') if round_number==0 else ('frozen','eager')):
                    if time.monotonic()-started>6500:raise TimeoutError('CPU campaign work reserve reached')
                    timing=measure_cpu_latency(nominal,predictors[seed][name],context=contexts[seed],environment=contexts[seed].report)
                    immutable_json(root/f'timing-{round_number+1}-{seed}-{name}.json',timing)
                    values=timing['comparators']; ratio=values['trained_raw_reference']['p95_ms']/values['independent_numerical_reference']['p95_ms']
                    summary={'round':round_number+1,'seed':seed,'variant':name,'p95_ms':{k:v['p95_ms'] for k,v in values.items()},
                             'p95_ratio_to_numerical':ratio,'development_cost_check':'pass' if ratio<=.8 else 'no_go'}
                    summaries.append(summary);print(summary,flush=True)
        if contexts[SEEDS[0]].torch.cuda.is_initialized():raise ValueError('unexpected CUDA initialization')
        immutable_json(root/'summary.json',{'schema':'nca-ik-cpu-development-result/v1','declaration_file_sha256':sha256_file(root/'declaration.json'),
            'checkpoints':checkpoints,'results':summaries,'predictor_calls':count[0],'optimizer_calls':0,'final_calls':0,
            'elapsed_seconds':time.monotonic()-started,'qualified':False})
    except BaseException as exc:
        immutable_json(root/'failure.json',{'type':type(exc).__name__,'message':str(exc),'calls':count[0]})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--training',required=True)
    run(parser.parse_args())


if __name__=='__main__':main()

# Copyright 2026 techrote. All rights reserved.
