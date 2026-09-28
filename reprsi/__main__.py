from pathlib import Path
import argparse
import json
import importlib.metadata
from .diagnostics import make_math_pool,read_jsonl,write_jsonl,validate_pools,sample_batch
from .metrics import seed_for
from .storage import atomic_json,digest,file_digest,link_or_copy


from .interfaces import load_config,make_policy,resolve_factory


def make_domain(args,cfg):
    if cfg.get("domain_factory"):
        return resolve_factory(cfg["domain_factory"])(cfg,args)
    domain=cfg.get("domain","math")
    if domain=="manufactoria":
        from .manufactoria import Domain
        if not args.delta_root:raise ValueError("--delta-root is required")
        return Domain(args.delta_root)
    if domain=="twohop":
        from .mechanism import TwoHopDomain
        if not args.twohop_data:raise ValueError("--twohop-data is required")
        return TwoHopDomain(args.twohop_data)
    if domain!="math":raise ValueError("Unknown domain")
    return None


def load_tasks(path,args,cfg,domain=None,training=False,allow_empty=False):
    if cfg.get("task_loader"):
        return resolve_factory(cfg["task_loader"])(path,args,cfg,domain,training,allow_empty)
    from .benchmarks import validate_targets
    rows=read_jsonl(path)
    if cfg.get("domain")=="twohop":
        from .mechanism import load_entities
        if training and any(r.get("split")!="train" for r in rows):raise ValueError("Target training split required")
        tasks=load_entities(rows)
    elif domain:
        if training and any(r.get("split")!="train" for r in rows):raise ValueError("Target training split required")
        tasks=domain.load_released_tasks(rows)
    else:
        from .benchmarks import load_targets,HARPChecker
        checker=resolve_factory(cfg["checker_factory"])(cfg,args) if cfg.get("checker_factory") else HARPChecker(args.harp_root)
        tasks=load_targets(path,checker,"train" if training else None,training,
                           cfg["model"] if training else None,cfg["revision"] if training else None,allow_empty)
    if not tasks and not allow_empty:raise ValueError("Empty task set")
    if len({t.identity for t in tasks})!=len(tasks):raise ValueError("Duplicate task identity")
    if not training and len({r.get("split") for r in rows})>1:raise ValueError("Mixed partitions")
    return rows,tasks


def common(sub,name):
    p=sub.add_parser(name);p.add_argument("--config",default="configs/math.json");p.add_argument("--output",required=True)
    p.add_argument("--delta-root");p.add_argument("--harp-root");p.add_argument("--twohop-data")
    p.add_argument("--seed",type=int);p.add_argument("--resume",action="store_true")
    return p


def main():
    parser=argparse.ArgumentParser(description="RepRSI workflows")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("experiment");p.add_argument("--spec",required=True);p.add_argument("--resume",action="store_true");p.add_argument("--plan",action="store_true")
    p=sub.add_parser("smoke");p.add_argument("--output",default="runs/smoke");p.add_argument("--resume",action="store_true")
    p=sub.add_parser("prepare-diagnostics");p.add_argument("--output",default="data/diagnostics")
    p.add_argument("--domain",choices=["math","manufactoria"],default="math");p.add_argument("--seed",type=int,default=2026)
    for command in ("prepare-math","prepare-harp"):
        p=sub.add_parser(command);p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("prepare-manufactoria");p.add_argument("--source",required=True);p.add_argument("--output",required=True)
    p.add_argument("--delta-root",required=True);p.add_argument("--split",choices=["train","dev","test"],required=True)
    p.add_argument("--family",choices=["START","APPEND","EXACT","REGEX","COMPR","HAS"],required=True)
    p=sub.add_parser("prepare-twohop");p.add_argument("--output",required=True)
    p=sub.add_parser("aggregate");p.add_argument("--inputs",nargs="+",required=True);p.add_argument("--baseline",nargs="+");p.add_argument("--output",required=True)
    p=sub.add_parser("analyze-prediction");p.add_argument("--data",required=True);p.add_argument("--likelihood",action="store_true");p.add_argument("--output",required=True)
    p=sub.add_parser("summarize-fresh");p.add_argument("--runs",nargs="+",required=True);p.add_argument("--plot",action="store_true");p.add_argument("--output",required=True)
    p=sub.add_parser("efficiency");p.add_argument("--runs",nargs="+",required=True);p.add_argument("--reference-method",default="target");p.add_argument("--output",required=True)
    p=sub.add_parser("doctor");p.add_argument("--config",default="configs/math.json");p.add_argument("--output")
    p=common(sub,"probe");p.add_argument("--diagnostics");p.add_argument("--layer",type=int,default=0)
    p=common(sub,"train");p.add_argument("--diagnostics",required=True);p.add_argument("--calibration",required=True)
    p.add_argument("--initial");p.add_argument("--monitor");p.add_argument("--target-train");p.add_argument("--eval-data");p.add_argument("--budget-from")
    p.add_argument("--method",choices=["reprsi","matched-search","target","uncertainty","shuffled","prompted","direct","human"])
    p.add_argument("--rounds",type=int);p.add_argument("--preparation-costs",nargs="*",default=[])
    p=common(sub,"calibration-trajectory");p.add_argument("--batches",type=int,default=2);p.add_argument("--initial")
    p=common(sub,"calibrate");p.add_argument("--diagnostics",required=True);p.add_argument("--checkpoints",nargs="+",required=True)
    p.add_argument("--layers",nargs="+",type=int);p.add_argument("--specs-per-family",type=int,default=1)
    p.add_argument("--initial")
    for command in ("evaluate","screen"):
        p=common(sub,command);p.add_argument("--data",required=True);p.add_argument("--checkpoint");p.add_argument("--samples",type=int,default=32)
        p.add_argument("--greedy",action="store_true");p.add_argument("--save-responses",action="store_true");p.add_argument("--initial")
        p.add_argument("--method");p.add_argument("--allow-legacy-checkpoint",action="store_true")
    p=common(sub,"export-curricula");p.add_argument("--initial");p.add_argument("--checkpoint",required=True);p.add_argument("--count",type=int,default=32)
    p.add_argument("--sampling-seed",type=int,default=2027)
    p=common(sub,"export-reference");p.add_argument("--method",choices=["direct","human"],required=True)
    p.add_argument("--count",type=int,default=1024);p.add_argument("--target-train");p.add_argument("--sampling-seed",type=int,default=2027)
    p=common(sub,"fresh-student");p.add_argument("--curricula",required=True);p.add_argument("--initial")
    p.add_argument("--steps",type=int,default=800);p.add_argument("--eval-every",type=int,default=40);p.add_argument("--eval-data")
    p.add_argument("--teacher-id");p.add_argument("--supervised",action="store_true")
    p=common(sub,"intervene");p.add_argument("--initial",required=True);p.add_argument("--layer",type=int);p.add_argument("--calibration")
    p.add_argument("--seeds",nargs="+",type=int,default=list(range(5)));p.add_argument("--steps",type=int,default=500)
    p.add_argument("--continuation-steps",type=int,default=400)
    p=common(sub,"prediction-trials");p.add_argument("--groups",required=True);p.add_argument("--initial")
    p.add_argument("--diagnostics");p.add_argument("--target-train");p.add_argument("--eval-data")
    p.add_argument("--layer",type=int);p.add_argument("--calibration");p.add_argument("--trials",type=int,default=5)
    p.add_argument("--curriculum-steps",type=int,default=150);p.add_argument("--continuation-steps",type=int,default=400)
    p=common(sub,"prediction-origins");p.add_argument("--initial");p.add_argument("--target-train");p.add_argument("--runs",type=int,default=200)
    p.add_argument("--steps",type=int,default=400)
    p=common(sub,"prediction-groups");p.add_argument("--teacher-output-limit",type=int);p.add_argument("--origins",required=True);p.add_argument("--groups-per-run",type=int,default=10)
    p.add_argument("--items",type=int,default=100)
    args=parser.parse_args()
    if args.command=="experiment":
        from .workflow import execute
        print(json.dumps(execute(args.spec,args.resume,args.plan),indent=2));return
    if args.command=="smoke":
        from .smoke import TinyPolicy,smoke_config
        from .loop import run
        cfg=smoke_config();print(run(cfg,TinyPolicy(cfg),make_math_pool("reward",1),args.output,0,resume=args.resume));return
    if args.command=="prepare-diagnostics":
        maker=make_math_pool;counts=[64,16,4]
        if args.domain=="manufactoria":
            from .manufactoria import make_dfa_pool
            maker=make_dfa_pool;counts=[128,32,8]
        pools=[maker(split,count,args.seed) for split,count in zip(["reward","monitor","calibration"],counts)]
        validate_pools(pools)
        for split,pool in zip(["reward","monitor","calibration"],pools):write_jsonl(Path(args.output)/(split+".jsonl"),pool)
        atomic_json(Path(args.output)/"manifest.json",{"domain":args.domain,"seed":args.seed,"counts":list(map(len,pools)),"digests":list(map(digest,pools))})
        print({"counts":list(map(len,pools)),"domain":args.domain});return
    if args.command in ("prepare-math","prepare-harp"):
        from .benchmarks import prepare_math,prepare_harp
        (prepare_math if args.command=="prepare-math" else prepare_harp)(args.source,args.output);return
    if args.command=="prepare-manufactoria":
        from .manufactoria import Domain,FAMILIES
        domain=Domain(args.delta_root);records=read_jsonl(args.source);tasks=domain.load_released_tasks(records)
        for row in records:
            family=row.get("problem_family") or row.get("problem_type") or row.get("pattern_type") or row.get("family")
            if family not in (args.family,FAMILIES[args.family]):raise ValueError("Released family metadata does not match --family")
        from .curricula import task_record
        rows=[{**task_record(t),"split":args.split,"benchmark":"Manufactoria-"+args.family,"family":args.family} for t in tasks]
        if len({r["id"] for r in rows})!=len(rows):raise ValueError("Duplicate released Manufactoria identity")
        write_jsonl(args.output,rows);print({"examples":len(rows)});return
    if args.command=="prepare-twohop":
        from .mechanism import prepare_twohop,exposure_report
        print(prepare_twohop(args.output));print(exposure_report(args.output));return
    if args.command=="efficiency":
        from .analysis import efficiency_report
        report=efficiency_report(args.runs,args.reference_method);atomic_json(args.output,report);print(report);return
    if args.command in ("aggregate","analyze-prediction","summarize-fresh"):
        from .analysis import aggregate,prediction_analysis,fresh_curves,plot_fresh
        if args.command=="aggregate":report=aggregate([load_config(x) for x in args.inputs],[load_config(x) for x in args.baseline] if args.baseline else None)
        elif args.command=="analyze-prediction":report=prediction_analysis(read_jsonl(args.data),args.likelihood)
        else:report=fresh_curves(args.runs)
        atomic_json(args.output,report)
        if args.command=="summarize-fresh" and args.plot:plot_fresh(report,Path(args.output).with_suffix(".pdf"))
        print(report);return
    cfg=load_config(args.config)
    if args.command=="doctor":
        import torch,shutil
        report={"python_libraries":{},"cuda_available":torch.cuda.is_available(),"cuda_devices":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "model":cfg["model"],"revision":cfg["revision"],"disk_free_bytes":shutil.disk_usage(Path.cwd()).free}
        for name in ("torch","transformers","numpy","sympy","lm-format-enforcer","automata-lib"):
            try:report["python_libraries"][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:report["python_libraries"][name]="not installed"
        if args.output:atomic_json(args.output,report)
        print(json.dumps(report,indent=2));return
    if args.seed is not None:cfg["seed"]=args.seed
    if getattr(args,"teacher_output_limit",None) is not None:cfg["teacher_output_limit"]=args.teacher_output_limit
    if args.command=="screen" and (args.checkpoint or args.initial):raise ValueError("Screening must use the initial instruction-tuned model")
    if args.command=="train":
        if args.method:cfg["method"]=args.method
        if args.rounds:cfg["rounds"]=args.rounds
        if args.budget_from:
            reference=load_config(args.budget_from)
            if any(reference[k]!=cfg[k] for k in ("model","revision","seed")):raise ValueError("Reference budget belongs to a different backbone/seed")
            if reference["method"]!="reprsi" or reference["stop_reason"]!="round_limit":raise ValueError("Matched budget requires a completed RepRSI reference")
            cfg["max_gpu_seconds"]=reference["gpu_seconds"]
            cfg["rounds"]=args.rounds or 100000
    domain=make_domain(args,cfg)
    if args.command=="export-reference":
        from .curricula import human_curriculum,task_record
        import numpy as np
        from .policy import seed_all
        if args.count<3:raise ValueError("Reference set needs at least three examples")
        seed_all(args.sampling_seed);rows=[]
        if args.method=="direct":
            if not args.target_train:raise ValueError("Direct reference requires --target-train")
            _,tasks=load_tasks(args.target_train,args,cfg,domain,True)
            rng=np.random.default_rng(args.sampling_seed)
            tasks=[tasks[int(i)] for i in rng.choice(len(tasks),args.count,replace=len(tasks)<args.count)]
            for task in tasks:
                row=task_record(task)
                if domain is None:row["kind"]="benchmark_math"
                rows.append(row)
        else:
            for stage,indices in enumerate(np.array_split(np.arange(args.count),3)):
                tasks=human_curriculum(len(indices),seed_for(args.sampling_seed,"human",stage),stage,domain)
                rows.extend({**task_record(t),"stage":stage} for t in tasks)
        write_jsonl(args.output,rows);print({"examples":len(rows),"method":args.method});return
    if args.command=="probe":
        from .probe import probe
        print(probe(cfg,args.output,domain,read_jsonl(args.diagnostics) if args.diagnostics else None,args.layer));return
    policy=make_policy(cfg)
    if args.command=="train":
        from .loop import run
        if args.initial:
            policy.load(args.initial);cfg["initial_id"]=file_digest(args.initial);policy.set_reference(cfg["initial_id"])
        calibration=load_config(args.calibration)
        if any(calibration[k]!=cfg[k] for k in ("model","revision","domain")):raise ValueError("Calibration backbone/domain mismatch")
        cfg["preparation_gpu_seconds"]=calibration["calibration_gpu_seconds"]
        cfg["preparation_wall_seconds"]=calibration.get("calibration_wall_seconds",0.)
        seen=set()
        for filename in args.preparation_costs:
            sha=file_digest(filename)
            if sha in seen:raise ValueError("Duplicate preparation charge")
            seen.add(sha);cost=load_config(filename)
            if cost["cost_role"]!="shared_training_preparation":raise ValueError("Only TRAIN/DEV screening is charged to training")
            if any(cost[k]!=cfg[k] for k in ("model","revision")):raise ValueError("Preparation backbone mismatch")
            cfg["preparation_gpu_seconds"]+=cost["gpu_seconds"]
            cfg["preparation_wall_seconds"]+=cost.get("wall_seconds",0.)
        reward=read_jsonl(args.diagnostics);monitor=read_jsonl(args.monitor) if args.monitor else None
        for pool in (reward,monitor or []):
            if {r["id"] for r in pool}&set(calibration["diagnostic_ids"]):raise ValueError("Calibration overlaps training/monitoring")
            if {digest(r["prompt"]) for r in pool}&set(calibration.get("diagnostic_prompt_hashes",[])):raise ValueError("Calibration prompt overlap")
        targets=load_tasks(args.target_train,args,cfg,domain,True)[1] if args.target_train else None
        eval_tasks=None
        if args.eval_data:
            rows,eval_tasks=load_tasks(args.eval_data,args,cfg,domain)
            if any(r.get("split") not in ("test","dev") for r in rows):raise ValueError("Periodic evaluation requires held-out records")
        if args.budget_from:
            from .loop import hardware_signature
            if reference["gpu_count"]!=policy.gpu_count:raise ValueError("Matched GPU count differs")
            if reference.get("hardware")!=hardware_signature(policy):raise ValueError("Matched hardware/precision differs")
            if reference["preparation_gpu_seconds"]!=cfg["preparation_gpu_seconds"]:raise ValueError("Shared preparation costs differ")
        print(run(cfg,policy,reward,args.output,calibration["layer"],domain,targets,args.resume,monitor,eval_tasks));return
    if args.command=="calibration-trajectory":
        from .curricula import math_items
        from .tasks import construct
        from .experiments import train_fixed
        from .policy import seed_all
        initial_id="pretrained"
        if args.initial:
            policy.load(args.initial);initial_id=file_digest(args.initial);policy.set_reference(initial_id)
        tasks=[]
        for batch in range(args.batches):
            seed=seed_for(2026,"independent_calibration_training",batch);seed_all(seed)
            current=domain.fixed_calibration_curriculum(cfg["items"],seed) if domain else [construct(x) for x in math_items(cfg["items"],seed,batch%3)]
            tasks.extend(current*2)
        cfg["seed"]=2026
        report=train_fixed(policy,cfg,tasks,args.output,args.batches*cfg["student_steps"],cfg["student_steps"],resume=args.resume,initial_id=initial_id)
        output=Path(args.output);link_or_copy(output/"initial.pt",output/"step_000.pt")
        for step in range(cfg["student_steps"],args.batches*cfg["student_steps"]+1,cfg["student_steps"]):
            link_or_copy(output/"states"/f"step_{step:06d}"/"student.pt",output/f"step_{step:03d}.pt")
        print(report);return
    if args.command=="calibrate":
        from .calibrate import select_layer
        from .loop import Ledger
        if args.initial:
            policy.load(args.initial);policy.set_reference(file_digest(args.initial))
        pool=read_jsonl(args.diagnostics);validate_pools([pool])
        batch=sample_batch(pool,args.specs_per_family,seed_for(2026,"calibration_sample"))
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        ledger=Ledger(policy,path.with_suffix(".compute.jsonl"))
        with ledger.charge("layer_calibration"):
            report=select_layer(policy,batch,args.checkpoints,args.layers or range(len(policy.blocks)),path.with_suffix(".progress.json"),args.resume)
        trajectory_cost=0.;trajectory_wall=0.
        for parent in {Path(x).parent for x in args.checkpoints}:
            cost_file=parent/"compute.jsonl"
            if not cost_file.exists():raise ValueError("Calibration checkpoints require a sibling compute.jsonl")
            trajectory_cost+=sum(x["gpu_seconds"] for x in read_jsonl(cost_file))
            trajectory_wall+=sum(x["wall_seconds"] for x in read_jsonl(cost_file))
        report.update(model=cfg["model"],revision=cfg["revision"],domain=cfg["domain"],
            diagnostic_ids=[r["id"] for r in pool],diagnostic_prompt_hashes=[digest(r["prompt"]) for r in pool],
            calibration_gpu_seconds=ledger.total+trajectory_cost,calibration_wall_seconds=ledger.wall+trajectory_wall)
        atomic_json(path,report);print({"selected_layer":report["layer"]});return
    if args.command in ("evaluate","screen"):
        from .evaluation import evaluate
        cfg["method"]=args.method or ("initial" if not args.checkpoint and not args.initial else cfg.get("method","reprsi"))
        rows,tasks=load_tasks(args.data,args,cfg,domain,allow_empty=True)
        if args.initial:
            policy.load(args.initial);policy.set_reference(file_digest(args.initial))
        if args.checkpoint:policy.load(args.checkpoint,allow_legacy=args.allow_legacy_checkpoint)
        checkpoint_id=file_digest(args.checkpoint) if args.checkpoint else (file_digest(args.initial) if args.initial else "initial")
        print(evaluate(policy,cfg,rows,tasks,args.output,args.samples,args.greedy,args.command=="screen",args.resume,checkpoint_id,args.save_responses));return
    if args.command=="export-curricula":
        from .experiments import export_teacher
        if args.initial:
            policy.load(args.initial);policy.set_reference(file_digest(args.initial))
        policy.load(args.checkpoint)
        print(export_teacher(policy,cfg,args.output,args.count,domain,args.resume,file_digest(args.checkpoint),args.sampling_seed));return
    if args.command=="fresh-student":
        from .experiments import train_fixed
        from .curricula import load_frozen
        records=read_jsonl(args.curricula)
        if args.supervised and (cfg.get("domain")!="twohop" or any(r.get("split")!="atomic" for r in records)):
            raise ValueError("Supervised preparation is reserved for the atomic fact partition")
        if cfg.get("domain")=="twohop":
            from .mechanism import load_entities
            tasks=load_entities(records)
        else:
            checker=None
            if any(r.get("kind")=="benchmark_math" for r in records):
                from .benchmarks import HARPChecker
                checker=resolve_factory(cfg["checker_factory"])(cfg,args) if cfg.get("checker_factory") else HARPChecker(args.harp_root)
            tasks=load_frozen(records,domain,checker)
        stages=None
        if any("stage" in r for r in records):
            if any(r.get("stage") not in (0,1,2) for r in records):raise ValueError("Invalid human stage labels")
            stages=[[t for t,r in zip(tasks,records) if r["stage"]==s] for s in range(3)]
            if any(not x for x in stages):raise ValueError("All three human stages are required")
        initial_id="pretrained"
        if args.initial:
            policy.load(args.initial);initial_id=file_digest(args.initial);policy.set_reference(initial_id)
        eval_tasks=None
        if args.eval_data:
            rows,eval_tasks=load_tasks(args.eval_data,args,cfg,domain)
            if any(r.get("split") not in ("test","dev","atomic") for r in rows):raise ValueError("Fresh evaluation partition is invalid")
        print(train_fixed(policy,cfg,tasks,args.output,args.steps,args.eval_every,eval_tasks,args.resume,
                          args.supervised,initial_id,args.teacher_id or digest(records),stages));return
    if args.command in ("intervene","prediction-trials"):
        if args.layer is None:
            if not args.calibration:raise ValueError("Supply --layer or --calibration")
            calibration=load_config(args.calibration)
            if any(calibration[k]!=cfg[k] for k in ("model","revision","domain")):raise ValueError("Calibration differs")
            args.layer=calibration["layer"]
    if args.command=="intervene":
        from .mechanism import run_interventions
        if cfg.get("domain")!="twohop":raise ValueError("Use the twohop configuration")
        print(run_interventions(policy,cfg,args.twohop_data,args.initial,args.output,args.layer,args.seeds,args.steps,args.continuation_steps,args.resume));return
    if args.command=="prediction-trials":
        from .prediction import collect
        if cfg.get("domain")=="twohop":
            root=Path(args.twohop_data)
            diagnostics=read_jsonl(root/"reward.jsonl")
            targets=load_tasks(root/"train.jsonl",args,cfg,domain,True)[1]
            evaluation=load_tasks(root/"test.jsonl",args,cfg,domain)[1]
        else:
            if not all((args.diagnostics,args.target_train,args.eval_data)):
                raise ValueError("Prediction requires --diagnostics, --target-train, and --eval-data")
            diagnostics=read_jsonl(args.diagnostics)
            targets=load_tasks(args.target_train,args,cfg,domain,True)[1]
            rows,evaluation=load_tasks(args.eval_data,args,cfg,domain)
            if any(r.get("split") not in ("dev","test") for r in rows):raise ValueError("Held-out evaluation required")
        result=collect(policy,cfg,diagnostics,targets,evaluation,read_jsonl(args.groups),args.initial,args.output,args.layer,
            domain,args.trials,args.curriculum_steps,args.continuation_steps,args.resume)
        atomic_json(Path(args.output)/"summary.json",result);print(result);return
    if args.command=="prediction-origins":
        from .prediction import origins
        source=Path(args.twohop_data)/"train.jsonl" if cfg.get("domain")=="twohop" else args.target_train
        if not source:raise ValueError("--target-train is required")
        targets=load_tasks(source,args,cfg,domain,True)[1]
        print(origins(policy,cfg,targets,args.initial,args.output,args.runs,args.steps,args.resume));return
    if args.command=="prediction-groups":
        from .prediction import generate_groups
        origins=read_jsonl(args.origins)
        for r in origins:
            r["origin"]=str((Path(args.origins).parent/r["origin"]).resolve())
        print(generate_groups(policy,cfg,origins,args.output,domain,args.groups_per_run,args.items,args.resume));return


if __name__=="__main__":main()
