"""Strict IK v4 qualification: preparation and explicitly admitted execution.

No final access path exists. A development result is never execution admission.
"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import uuid

from nca.contracts import sha256_json
from nca.runs import immutable_json, sha256_file
from nca.train.ik_dev import DevConfig, SEEDS, sampler, epoch_batches, batched_predictions
from nca.train.ik_dev_sweep import cuda_context, freeze
from nca.train.ik_experiment import _source, _read, _titan_visibility, ACCEPTANCE_PATH
from nca.train.ik_matched_prepare import TITAN_FINGERPRINT, GIB
from nca.train.ik_run_state_v2 import IKRunLedger

RECIPE = "configs/ik/experiments/runner-qualification-v4.json"
REPORT_MAX_BYTES = 8*1024**2
GROUP_WALL_SECONDS = 24*60*60
EXIT_MARGIN_SECONDS = 10
PUBLICATION_RESERVE_SECONDS = 7200
WORK_RESERVE_SECONDS = 10800
CHECKS = ("exact_head_ci", "source_and_cpu_checks", "corpus_and_splits",
          "allocation_and_no_retry", "scoring_and_selection", "environment_and_resource_feasibility")


def invocation_start():
    """Map Windows process/venv-launcher creation time onto the monotonic clock.

    This includes interpreter imports and admission preflight in the ceiling.
    An earlier Python parent is conservative; execution requires a fresh CLI.
    """
    import ctypes
    from ctypes import wintypes
    if os.name!="nt":raise ValueError("qualified Windows process lifetime required")
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=(wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4))
    kernel.QueryFullProcessImageNameW.argtypes=(wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
    kernel.GetSystemTimeAsFileTime.argtypes=(ctypes.POINTER(wintypes.FILETIME),)
    def ticks(value):return (value.dwHighDateTime<<32)|value.dwLowDateTime
    candidates=[]
    for pid in (os.getpid(),os.getppid()):
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:raise ValueError("process lifetime metadata unavailable")
        try:
            values=[wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle,*[ctypes.byref(v) for v in values]):
                raise ValueError("process creation timestamp unavailable")
            image=ctypes.create_unicode_buffer(32768);size=wintypes.DWORD(len(image))
            if not kernel.QueryFullProcessImageNameW(handle,0,image,ctypes.byref(size)):
                raise ValueError("process image identity unavailable")
            if pid==os.getpid() or Path(image.value).name.lower()=="python.exe":candidates.append(ticks(values[0]))
        finally:kernel.CloseHandle(handle)
    now=wintypes.FILETIME();monotonic=time.monotonic();kernel.GetSystemTimeAsFileTime(ctypes.byref(now))
    elapsed=(ticks(now)-min(candidates))/10_000_000
    if elapsed<0 or elapsed>monotonic:raise ValueError("process clock identity invalid")
    return monotonic-elapsed


def ordered_sampler(samples, seed, config):
    plan = sampler(samples, seed, config)
    if config.batch_order == "shuffled":
        batches = tuple(b for e in range(config.epochs) for b in epoch_batches(plan, config, e))
        plan.schedule_sha256 = sha256_json(["ik-qualification-v3-order/v1", plan.schedule_sha256,
                                           config.to_dict(), batches])
        plan._batches = batches
        plan._prefix = [0]
        for b in batches:
            plan._prefix.append(plan._prefix[-1]+len(b))
    return plan


def ceilings(config, samples):
    plan = ordered_sampler(samples, SEEDS[0], config)
    return dict(max_optimizer_steps=plan.planned_batches,
                max_sample_presentations=plan.total_presentations,
                max_tuning_calls=2020*len(config.evaluation_epochs()),
                max_evaluation_calls=0, max_checkpoints=1+len(config.evaluation_epochs()),
                max_attempts=1, max_wall_seconds=GROUP_WALL_SECONDS, max_memory_bytes=GIB)



def resource_policy():
    return {"workers":3,"board_cap_bytes":20*GIB,"host_commit_per_process_bytes":8*GIB,
            "allocator_per_process_bytes":GIB,"cpu_threads":1,"group_wall_seconds":GROUP_WALL_SECONDS,
            "startup_admission_limit_seconds":60,"active_deadline_seconds":GROUP_WALL_SECONDS-EXIT_MARGIN_SECONDS,
            "worker_publication_reserve_seconds":PUBLICATION_RESERVE_SECONDS,
            "work_admission_reserve_seconds":WORK_RESERVE_SECONDS,"exit_margin_seconds":EXIT_MARGIN_SECONDS,
            "hard_timer_through_interpreter_exit":True,
            "startup_free_disk_bytes":16*GIB,"continuous_free_disk_floor_bytes":4*GIB}


def source():
    root = Path(__file__).resolve().parents[3]
    record = _source(root)
    record["recipe_file_sha256"] = sha256_file(root/RECIPE)
    return record


def recipe_record():
    return _read(Path(__file__).resolve().parents[3]/RECIPE)


def validate_declaration(declaration, expected_source, recipe):
    if (declaration.get("schema")!="nca-ik-qualification-v4/v1"
            or declaration.get("source")!=expected_source or declaration.get("recipe")!=recipe
            or declaration.get("final_calls")!=0 or recipe.get("final_calls")!=0
            or recipe.get("execution_authorized") is not False
            or recipe.get("executable_source_sha256")!=expected_source["source_sha256"]
            or declaration.get("allocations")!=recipe.get("ceilings_per_seed")
            or declaration.get("schedules")!=recipe.get("schedules")
            or declaration.get("corpus_identity")!=recipe.get("corpus_identity")
            or any(declaration.get(k)!=recipe.get(k) for k in
                   ('resource_policy','tuning_epochs','selection','runtime','profile_sha256',
                    'gpu_visibility_sha256','acceptance_file_sha256'))
            or sha256_json(declaration.get("normalization"))!=recipe.get("normalization_sha256")):
        raise ValueError("strict declaration differs from committed source/recipe/data/ceilings")


def corpus_for(declaration):
    from nca.train.ik_corpus import read_polar_followup_corpus
    return read_polar_followup_corpus(declaration["corpus"], declaration["registry"],
                                      source_sha256=declaration["source"]["source_sha256"])


def identity(declaration, seed):
    return dict(declaration_sha256=sha256_json(declaration), config_sha256=sha256_json(declaration["recipe"]),
                source_sha256=declaration["source"]["source_sha256"],
                profile_sha256=declaration["profile_sha256"],
                dataset_sha256=sha256_json(declaration["corpus_identity"]),
                normalization_sha256=sha256_json(declaration["normalization"]),
                runtime_sha256=sha256_json(declaration["runtime"]), seed=seed)


def prepare(args):
    root = Path(args.root).resolve()
    checkout = Path(__file__).resolve().parents[3]
    inputs=(checkout,Path(args.corpus).resolve(),Path(args.registry).resolve())
    if root.exists() or any(root==p or root.is_relative_to(p) or p.is_relative_to(root) for p in inputs):
        raise ValueError("new external qualification packet required")
    _titan_visibility(expected_sha256=TITAN_FINGERPRINT)
    recipe = recipe_record()
    if recipe["final_calls"] != 0 or recipe["execution_authorized"] is not False:
        raise ValueError("preparation never authorizes execution or finals")
    config = DevConfig(**recipe["config"])
    declaration = {"schema":"nca-ik-qualification-v4/v1", "source":source(), "recipe":recipe,
                   "corpus":str(Path(args.corpus).resolve()), "registry":str(Path(args.registry).resolve()),
                   "final_calls":0, "execution_authorized":False, "gpu_visibility_sha256":TITAN_FINGERPRINT}
    corpus = corpus_for(declaration)
    from nca.ik.polar import fit_training_normalization
    from nca.ik.profiles import runner_right_sole_profile
    from nca.ik.declaration import runtime_expectation
    declaration.update(corpus_identity=corpus.identity, normalization=fit_training_normalization(corpus).to_dict(),
                       profile_sha256=runner_right_sole_profile().sha256, runtime=runtime_expectation(),
                       acceptance_file_sha256=sha256_file(checkout/ACCEPTANCE_PATH),
                       allocations=ceilings(config, corpus.training),
                       schedules={str(s):ordered_sampler(corpus.training,s,config).schedule_sha256 for s in SEEDS},
                       tuning_epochs=list(config.evaluation_epochs()),
                       selection="lowest_complete_raw_nominal_p95_earliest_ties",
                       independent_admissions_required=2, owner_execution_required=True,
                       resource_policy=resource_policy())
    validate_declaration(declaration,source(),recipe)
    root.mkdir(parents=True)
    freeze(root/"declaration.json",declaration)
    for seed in SEEDS:
        IKRunLedger.create(root/f"seed-{seed}",identity=identity(declaration,seed),allocation=declaration["allocations"])
    freeze(root/"admission-template.json", {"schema":"nca-ik-v4-admission/v1", "head":declaration["source"]["revision"],
           "declaration_file_sha256":sha256_file(root/"declaration.json"), "reviewer":None,
           "independent_non_author":False,"checks":{k:False for k in CHECKS},"verdict":"pending"})
    print(sha256_file(root/"declaration.json"))


def check_reviews(root, reviews):
    root = Path(root)
    declaration = _read(root/"declaration.json",16*1024**2)
    validate_declaration(declaration,source(),recipe_record())
    digest = sha256_file(root/"declaration.json")
    reviewers=set()
    if len(reviews)!=2:
        raise ValueError("two independent admissions required")
    for path in reviews:
        review=_read(path)
        if (review.get("schema")!="nca-ik-v4-admission/v1" or review.get("head")!=declaration["source"]["revision"]
                or review.get("declaration_file_sha256")!=digest or review.get("independent_non_author") is not True
                or set(review.get("checks",{}))!=set(CHECKS)
                or any(review["checks"][k] is not True for k in CHECKS)
                or review.get("verdict")!="admit_strict_training_only"
                or not isinstance(review.get("reviewer"),str) or not review["reviewer"]
                or review["reviewer"] in reviewers):
            raise ValueError("fresh exact-head declaration-bound independent admission required")
        reviewers.add(review["reviewer"])
    return declaration


def check_admission(root, reviews, authorization):
    declaration=check_reviews(root,reviews)
    expected={"schema":"nca-ik-v4-owner-execution/v1",
              "declaration_file_sha256":sha256_file(Path(root)/'declaration.json'),
              "execute_full_training":True,"final_calls":0}
    if _read(authorization)!=expected:raise ValueError('new exact-packet owner go instruction required')
    return declaration


def strict_train(ledger, attempt, corpus, context, config, normalization):
    """Fixed full-training recipe, original durable reservations, zero retries."""
    from nca.train.ik_dev_trainer import DevTrainer, PreparedRows
    from nca.evaluation.ik_training import score_raw, _accuracy
    from nca.train.ik_experiment import independent_fk_guard
    plan=ordered_sampler(corpus.training,ledger.identity["seed"],config)
    trainer=DevTrainer(context,normalization,corpus.config,config)
    prepared=PreparedRows(context,corpus,corpus.training)
    history=[]
    attempt.checkpoint(trainer,plan.state_dict(),extra_state={"completed_epochs":0,"tuning_history":[]})
    ends={};cursor=0
    for e,count in enumerate(plan.batches_per_epoch,1):
        cursor+=count;ends[cursor]=e
    best=None
    while not plan.exhausted:
        if attempt.remaining_wall_seconds<WORK_RESERVE_SECONDS:
            raise TimeoutError("qualification allocation cannot extend")
        indexes=plan._batches[plan.cursor]
        plan.next_batch()
        batch=prepared.batch(indexes)
        token=attempt.reserve("optimizer",samples=len(indexes),call_id=f"batch-{plan.cursor:08d}")
        trainer.optimizer.param_groups[0]["lr"]=config.lr(trainer.committed_steps,plan.planned_batches)
        if trainer.committed_steps==0:
            before=[p.detach().clone() for p in trainer.model.parameters()]
            guard=independent_fk_guard(trainer,batch)
        receipt=trainer.step(batch)
        receipt["ordered_sample_ids_sha256"]=sha256_json(list(batch.sample_ids))
        receipt.update(lr=trainer.optimizer.param_groups[0]["lr"],stage=corpus.training[indexes[0]].task.stage)
        if trainer.committed_steps==1:
            receipt["independent_fk_guard"]=guard
            receipt["weights_changed"]=any(not context.torch.equal(a,b.detach()) for a,b in zip(before,trainer.model.parameters()))
            if not receipt["weights_changed"]:raise ValueError("first strict update changed no weights")
        attempt.complete(token,receipt)
        epoch=ends.get(plan.cursor)
        if epoch not in config.evaluation_epochs():continue
        token=attempt.reserve("tuning",calls=len(corpus.tuning),call_id=f"tuning-{epoch:04d}")
        predictions=batched_predictions(context,trainer.model,corpus.tuning)
        score=score_raw(corpus.tuning,predictions,corpus.config,authenticated_corpus=corpus,
                        seed=ledger.identity["seed"])
        path=ledger.root/f"tuning-epoch-{epoch:04d}.json"
        freeze(path,score)
        if path.stat().st_size>REPORT_MAX_BYTES:raise ValueError("bounded tuning report storage exceeded")
        attempt.complete(token,{"epoch":epoch,"report_file_sha256":sha256_file(path),"complete":score["complete"]})
        if not score["complete"]:raise ValueError("incomplete tuning report")
        item={"epoch":epoch,"metric":score["selection_metric"],"report":path.name,
              "report_file_sha256":sha256_file(path)}
        history.append(item)
        saved=attempt.checkpoint(trainer,plan.state_dict(),extra_state={"completed_epochs":epoch,"tuning_history":history})
        if best is None or item["metric"]<best["metric"]:
            best={**item,"checkpoint":saved,"gate":_accuracy(score)}
    freeze(ledger.root/"selected.json",best)
    attempt.finish("completed",reason="fixed_full_training_tuning_only",publication_reserve_seconds=PUBLICATION_RESERVE_SECONDS)
    return best


def worker(args):
    declaration=check_admission(args.root,args.review,args.authorization)
    from nca.train.ik_gpu_ownership import verify
    verify(_read(Path(args.root)/"ownership.json"))
    ledger=IKRunLedger.open(Path(args.root)/f"seed-{args.seed}",expected_identity=identity(declaration,args.seed),
                             expected_allocation=declaration["allocations"])
    if ledger.inspect()["attempt_count"]!=0:raise ValueError("one attempt; no resume")
    attempt=ledger.begin_attempt(started=args.started)
    try:
        corpus=corpus_for(declaration)
        from nca.ik.declaration import runtime_expectation
        from nca.ik.polar import fit_training_normalization
        if (runtime_expectation()!=declaration["runtime"]
                or corpus.identity!=declaration["corpus_identity"]
                or fit_training_normalization(corpus).to_dict()!=declaration["normalization"]):
            raise ValueError("strict corpus/normalization/runtime binding changed")
        context=cuda_context(args.seed)
        from nca.ik.polar import PolarNormalization
        strict_train(ledger,attempt,corpus,context,DevConfig(**declaration["recipe"]["config"]),
                     PolarNormalization.from_dict(declaration["normalization"]))
    except BaseException:
        if not attempt.closed:attempt.finish("failed",reason="strict_qualification_failed",publication_reserve_seconds=PUBLICATION_RESERVE_SECONDS)
        raise


def verify_selected(ledger, state, declaration):
    """Authenticate tuning files and earliest selection without another model call."""
    from nca.evaluation.ik_training import _accuracy
    from nca.train.ik_run_state_v2 import _verify_checkpoint
    history=state["latest_checkpoint"]["state"]["extra_state"]["tuning_history"]
    if [x["epoch"] for x in history]!=declaration["tuning_epochs"]:
        raise ValueError("complete declared tuning cadence required")
    for item in history:
        path=ledger.root/f"tuning-epoch-{item['epoch']:04d}.json"
        if item["report"]!=path.name or sha256_file(path)!=item["report_file_sha256"]:
            raise ValueError("tuning report bytes changed")
    best=min(history,key=lambda x:(x["metric"],x["epoch"]))
    selected=_read(ledger.root/"selected.json",16*1024**2)
    if any(selected[k]!=v for k,v in best.items()):
        raise ValueError("selection differs from earliest complete minimum")
    _verify_checkpoint(ledger.root,selected["checkpoint"],ledger.identity)
    if selected["checkpoint"]["state"]["extra_state"]["completed_epochs"]!=selected["epoch"]:
        raise ValueError("selected checkpoint epoch differs")
    report=_read(ledger.root/best["report"],8*1024**2)
    if not report["complete"] or report["selection_metric"]!=best["metric"] or _accuracy(report)!=selected["gate"]:
        raise ValueError("selected report metric or unchanged gate differs")
    return selected


def validate_terminal(state, allocation):
    expected={name:allocation['max_'+name] for name in
              ('optimizer_steps','sample_presentations','tuning_calls','evaluation_calls','checkpoints')}
    if (state['active'] or state['attempt_count']!=1 or state['unresolved_work']
            or state['last_status']!='completed' or state['charged']!=expected
            or state['completed_optimizer_steps']!=allocation['max_optimizer_steps']):
        raise ValueError('strict ledger did not complete the exact fixed recipe')


def hard_wall_watchdog(stop, began, deadline_seconds=GROUP_WALL_SECONDS-EXIT_MARGIN_SECONDS):
    """Never block the whole-process timer on filesystem or metadata queries."""
    while not stop.wait(.25):
        if time.monotonic()-began>=(deadline_seconds() if callable(deadline_seconds) else deadline_seconds):
            os._exit(2)  # Inherited Job kills children; durable start/ownership retain the failure hold.


def validate_cleanup_budget(elapsed, states):
    if elapsed+EXIT_MARGIN_SECONDS>=min(x['charged_wall_seconds'] for x in states):
        raise ValueError('supervisor publication/cleanup lacks paid per-seed wall reserve')


def run(args):
    began=invocation_start()
    from threading import Thread,Event
    from nca.ik.process_limits import memory_limit
    memory_limit(8192,kill_children_on_exit=True)
    hard_stop=Event();finish_deadline=[float(GROUP_WALL_SECONDS-EXIT_MARGIN_SECONDS)]
    timer=Thread(target=hard_wall_watchdog,args=(hard_stop,began,lambda:finish_deadline[0]),daemon=True)
    timer.start()  # Protect source/admission preflight too; never stop before interpreter exit.
    root=Path(args.root)
    declaration=check_admission(root,args.review,args.authorization)
    if time.monotonic()-began>60:raise ValueError("fresh qualification CLI startup exceeded its admission allowance")
    if (root/"group-start.json").exists():raise ValueError("strict group cannot repeat")
    from nca.train.ik_gpu_ownership import acquire,release
    from nca.train.ik_matched_supervisor import board_sample,stop_tree
    from nca.train.ik_dev_sweep import _host_free_commit
    if _host_free_commit()<32*GIB:raise ValueError("strict three-worker host commitment headroom insufficient")
    if shutil.disk_usage(root).free<16*GIB:raise ValueError("strict storage headroom insufficient")
    board_sample()
    ownership={"kind":"ik_v4_strict_qualification","claim":uuid.uuid4().hex,
               "declaration_file_sha256":sha256_file(root/"declaration.json")}
    acquire(ownership);freeze(root/"ownership.json",ownership)
    freeze(root/"group-start.json",{"monotonic":began,"ownership":ownership})
    active=[];successful=False
    stop=Event()
    def resource_watchdog():
        try:
            from nca.train.ik_dev_sweep import append
            sampled=0.
            with (root/"board.jsonl").open("x",encoding="utf-8") as log:
                while not stop.wait(.25):
                    now=time.monotonic()
                    if now-sampled>=2:
                        sample=board_sample();sample["host_free_commit_bytes"]=_host_free_commit()
                        sample["disk_free_bytes"]=shutil.disk_usage(root).free
                        if sample["temperature"]>80 or sample["host_free_commit_bytes"]<4*GIB or sample["disk_free_bytes"]<4*GIB:
                            raise ValueError("whole-invocation thermal/host/storage headroom cap")
                        append(log,sample);sampled=now
        except BaseException as exc:
            try:freeze(root/"resource-cap-failure.json",{"wall_seconds":time.monotonic()-began,"reason":str(exc)})
            finally:os._exit(2)  # The inherited Windows Job kills all workers; ownership remains held.
    watcher=Thread(target=resource_watchdog,daemon=True)
    watcher.start()
    try:
        for seed in SEEDS:
            command=[sys.executable,"-m","nca.train.ik_qualification_v4","_worker","--root",str(root),
                     "--seed",str(seed),"--started",repr(began),"--authorization",args.authorization]
            for review in args.review:command.extend(["--review",review])
            env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
            with (root/f"worker-{seed}.log").open("xb") as log:
                active.append(subprocess.Popen(command,stdout=log,stderr=log,env=env))
        while any(p.poll() is None for p in active):
            if any(p.poll() not in (None,0) for p in active):raise ValueError("strict worker failed; no retry")
            time.sleep(2)
        outcomes=[]
        # Tighten the timer from authenticated terminal envelopes before the expensive replay.
        from nca.train.ik_run_state_v2 import _record
        for seed in SEEDS:
            paid=_record(root/f'seed-{seed}/attempts/attempt-0001/finish.json')['record']['elapsed_seconds']
            finish_deadline[0]=min(finish_deadline[0],paid-EXIT_MARGIN_SECONDS)
        for seed,process in zip(SEEDS,active):
            if process.returncode!=0:raise ValueError("strict worker failed")
            ledger=IKRunLedger.open(root/f"seed-{seed}",expected_identity=identity(declaration,seed),
                                   expected_allocation=declaration["allocations"])
            state=ledger.inspect()
            validate_terminal(state,declaration['allocations'])
            finish_deadline[0]=min(finish_deadline[0],state['charged_wall_seconds']-EXIT_MARGIN_SECONDS)
            outcomes.append({"seed":seed,"selected":verify_selected(ledger,state,declaration),"ledger":state})
        if time.monotonic()-began>=GROUP_WALL_SECONDS-EXIT_MARGIN_SECONDS:raise ValueError("strict finalization exceeded active deadline")
        freeze(root/"group-result.json",{"wall_seconds":time.monotonic()-began,"outcomes":outcomes,
                                          "final_calls":0,"all_tuning_gates_pass":all(x["selected"]["gate"]=="pass" for x in outcomes)})
        if any(x["selected"]["gate"]!="pass" for x in outcomes):
            freeze(root/"terminal-no-go.json",{"reason":"unchanged_tuning_gate_failed","final_calls":0})
        validate_cleanup_budget(time.monotonic()-began,[x['ledger'] for x in outcomes])
        successful=True
    except BaseException as exc:
        for process in active:stop_tree(process)
        freeze(root/"group-failure.json",{"message":str(exc),"wall_seconds":time.monotonic()-began})
        raise
    finally:
        stop.set();watcher.join(timeout=1)
        # hard_stop remains unset: the daemon timer protects cleanup and interpreter exit.
        if successful and all(p.poll() is not None for p in active):release(ownership)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for verb in ("prepare","check-reviews","check-admission","run","_worker"):
        p=sub.add_parser(verb);p.add_argument("--root",required=True)
        if verb=="prepare":p.add_argument("--corpus",required=True);p.add_argument("--registry",required=True)
        else:
            p.add_argument("--review",action="append",required=True)
            if verb!='check-reviews':p.add_argument("--authorization",required=True)
        if verb=="_worker":p.add_argument("--seed",type=int,choices=SEEDS,required=True);p.add_argument("--started",type=float,required=True)
    args=parser.parse_args()
    if args.command=="check-admission":check_admission(args.root,args.review,args.authorization)
    elif args.command=='check-reviews':check_reviews(args.root,args.review)
    else:{"prepare":prepare,"run":run,"_worker":worker}[args.command](args)


if __name__=="__main__":main()

# Copyright 2026 techrote. All rights reserved.
