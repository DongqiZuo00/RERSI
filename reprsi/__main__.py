"""Command line entry points; run python -m reprsi --help."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import random
import time
from .diagnostics import make_math_pool,read_jsonl,write_jsonl,validate_pools,sample_batch
from .metrics import seed_for,pass_at_k


def load_config(path):
    return json.loads(Path(path).read_text())


def make_policy(cfg):
    from .policy import Policy,seed_all
    seed_all(cfg["seed"])
    return Policy(cfg)


def make_domain(args,cfg):
    if cfg.get("domain","math")=="manufactoria":
        from .manufactoria import Domain
        if not args.delta_root: raise ValueError("--delta-root is required")
        return Domain(args.delta_root)
    return None


def main():
    parser=argparse.ArgumentParser(description="Minimal RepRSI core implementation")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("smoke",help="Real tiny Transformer/RLOO integration check on CPU")
    p.add_argument("--output",default="runs/smoke")
    p=sub.add_parser("prepare-diagnostics")
    p.add_argument("--output",default="data/diagnostics")
    p.add_argument("--domain",choices=["math","manufactoria"],default="math")
    p.add_argument("--seed",type=int,default=2026)
    p=sub.add_parser("prepare-math");p.add_argument("--source",required=True);p.add_argument("--output",default="data/math")
    p=sub.add_parser("prepare-harp");p.add_argument("--source",required=True);p.add_argument("--output",default="data/harp")
    for command in ["train","calibration-trajectory","calibrate","evaluate","screen"]:
        p=sub.add_parser(command)
        p.add_argument("--config",default="configs/math.json")
        p.add_argument("--output",required=True)
        p.add_argument("--delta-root")
        if command=="train":
            p.add_argument("--diagnostics",required=True);p.add_argument("--calibration",required=True)
            p.add_argument("--target-train");p.add_argument("--harp-root");p.add_argument("--seed",type=int)
            p.add_argument("--preparation-costs",nargs="*",default=[])
        elif command=="calibration-trajectory":
            p.add_argument("--batches",type=int,default=2)
        elif command=="calibrate":
            p.add_argument("--diagnostics",required=True)
            p.add_argument("--checkpoints",nargs="+",required=True)
            p.add_argument("--layers",nargs="+",type=int)
            p.add_argument("--specs-per-family",type=int,default=1)
        else:
            p.add_argument("--data",required=True);p.add_argument("--checkpoint");p.add_argument("--harp-root")
            p.add_argument("--samples",type=int,default=32)
            p.add_argument("--greedy",action="store_true")
    args=parser.parse_args()
    if args.command=="smoke":
        from .smoke import TinyPolicy,smoke_config
        from .loop import run
        cfg=smoke_config();pool=make_math_pool("reward",1)
        run(cfg,TinyPolicy(cfg),pool,args.output,layer=0)
        return
    if args.command=="prepare-diagnostics":
        maker=make_math_pool
        counts=[64,16,4]
        if args.domain=="manufactoria":
            from .manufactoria import make_dfa_pool
            maker=make_dfa_pool;counts=[128,32,8]
        pools=[maker(split,count,args.seed) for split,count in zip(["reward","monitor","calibration"],counts)]
        validate_pools(pools)
        for split,pool in zip(["reward","monitor","calibration"],pools):
            write_jsonl(Path(args.output)/(split+".jsonl"),pool)
        print({"counts":list(map(len,pools)),"domain":args.domain});return
    if args.command in {"prepare-math","prepare-harp"}:
        from .benchmarks import prepare_math,prepare_harp
        (prepare_math if args.command=="prepare-math" else prepare_harp)(args.source,args.output);return
    cfg=load_config(args.config)
    if args.command=="train" and args.seed is not None: cfg["seed"]=args.seed
    domain=make_domain(args,cfg)
    policy=make_policy(cfg)
    if args.command=="train":
        from .loop import run
        calibration=load_config(args.calibration)
        if calibration["model"]!=cfg["model"] or calibration["revision"]!=cfg["revision"]:
            raise ValueError("Calibration backbone does not match training backbone")
        if calibration["domain"]!=cfg["domain"]: raise ValueError("Calibration domain mismatch")
        cfg["preparation_gpu_seconds"]=calibration["calibration_gpu_seconds"]
        for cost_file in args.preparation_costs:
            cost=load_config(cost_file)
            if cost["cost_role"]!="shared_training_preparation":raise ValueError("Only training/development screening is charged to training")
            cfg["preparation_gpu_seconds"]+=cost["gpu_seconds"]
        reward=read_jsonl(args.diagnostics)
        if {r["id"] for r in reward}&set(calibration["diagnostic_ids"]): raise ValueError("Calibration/reward overlap")
        targets=None
        if args.target_train:
            from .benchmarks import load_targets,HARPChecker
            targets=load_targets(args.target_train,HARPChecker(args.harp_root),"train",True)
        run(cfg,policy,reward,args.output,calibration["layer"],domain,targets)
        return
    if args.command=="calibration-trajectory":
        from .loop import Ledger
        from .schema import EXAMPLES
        from .tasks import construct
        from .policy import seed_all
        output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
        ledger=Ledger(policy,output/"compute.jsonl")
        with ledger.charge("calibration_checkpoint_save"): policy.save(output/"step_000.pt")
        for batch in range(args.batches):
            seed=seed_for(2026,"independent_calibration_training",batch);seed_all(seed)
            if domain:
                tasks=domain.fixed_calibration_curriculum(cfg["items"],seed)
            else:
                rng=random.Random(seed);tasks=[]
                for i in range(cfg["items"]):
                    item=deepcopy(EXAMPLES[i%4])
                    if i%4==0: item["parameters"]["a"]=[rng.randint(-20,20),rng.randint(1,20)]
                    elif i%4==1: item["parameters"]["a"]["poly"]=[rng.randint(-9,9) for _ in range(5)]
                    elif i%4==2: item["parameters"]["a"]=rng.randint(-20,20)
                    else:
                        item["parameters"]["a"]["matrix"]=[[1,0,rng.randint(-9,9)],[0,1,rng.randint(-9,9)]]
                    tasks.append(construct(item))
            with ledger.charge("calibration_training",batch):
                policy.train_curriculum(tasks,seed)
                policy.save(output/f"step_{(batch+1)*cfg['student_steps']:03d}.pt")
        return
    if args.command=="calibrate":
        from .calibrate import select_layer
        from .loop import Ledger
        pool=read_jsonl(args.diagnostics)
        batch=sample_batch(pool,args.specs_per_family,seed_for(2026,"calibration_sample"))
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        ledger=Ledger(policy,path.with_suffix(".compute.jsonl"))
        with ledger.charge("layer_calibration"):
            report=select_layer(policy,batch,args.checkpoints,args.layers or range(len(policy.blocks)))
        report.update(model=cfg["model"],revision=cfg["revision"],domain=cfg["domain"],diagnostic_ids=[r["id"] for r in batch])
        trajectory_cost=0.0
        for parent in {Path(checkpoint).parent for checkpoint in args.checkpoints}:
            cost_file=parent/"compute.jsonl"
            if not cost_file.exists():raise ValueError("Calibration checkpoints require a sibling compute.jsonl ledger")
            trajectory_cost+=sum(x["gpu_seconds"] for x in read_jsonl(cost_file))
        report["calibration_gpu_seconds"]=ledger.total+trajectory_cost
        path.write_text(json.dumps(report,indent=2)+"\n");print({"selected_layer":report["layer"]});return
    from .policy import seed_all
    from .benchmarks import HARPChecker,load_targets
    records=read_jsonl(args.data)
    if domain: tasks=domain.load_released_tasks(records)
    else:
        if not args.harp_root: raise ValueError("Evaluation requires --harp-root at the paper revision")
        tasks=load_targets(args.data,HARPChecker(args.harp_root))
    if args.checkpoint: policy.load(args.checkpoint)
    if args.command=="screen" and args.checkpoint: raise ValueError("Screening must use the initial checkpoint")
    n=128 if args.command=="screen" else (1 if args.greedy else args.samples)
    outcomes=[];policy.synchronize();begin=time.perf_counter()
    for record,task in zip(records,tasks):
        overflow=False
        try: policy.encode(task.prompt,16384)
        except ValueError: overflow=True
        successes=[]
        for sample in range(n):
            purpose="screening" if args.command=="screen" else "final_evaluation"
            seed_all(seed_for(2026 if args.command=="screen" else cfg["seed"],purpose,cfg["model"],task.identity,sample))
            value=0.0 if overflow else task.verify(policy.sample(task.prompt,greedy=args.greedy and args.command!="screen",input_limit=16384).text)
            successes.append(int(value))
        row={**record,"input_overflow":overflow,"model":cfg["model"],"revision":cfg["revision"]}
        row.update({("screening_outcomes" if args.command=="screen" else "outcomes"):successes})
        if args.command=="screen": row["screening_successes"]=sum(successes)
        outcomes.append(row)
    write_jsonl(args.output,outcomes)
    policy.synchronize();elapsed=time.perf_counter()-begin
    if args.command=="screen":
        write_jsonl(Path(args.output).with_suffix(".hard.jsonl"),[x for x in outcomes if x["screening_successes"]==0])
        scores={"hard_count":sum(x["screening_successes"]==0 for x in outcomes)}
    else:
        ks=[1] if args.greedy else [k for k in [1,4,8,16,24,32,64,128] if k<=n]
        scores={f"pass@{k}":(sum(pass_at_k(n,sum(x["outcomes"]),k) for x in outcomes)/len(outcomes) if outcomes else None) for k in ks}
    summary={**scores,"examples":len(outcomes),"samples":n,"input_overflow":sum(x["input_overflow"] for x in outcomes),
        "wall_seconds":elapsed,"gpu_seconds":elapsed*policy.gpu_count,
        "cost_role":"shared_training_preparation" if args.command=="screen" and all(r.get("split") in {"train","dev"} for r in records) else "evaluation_only"}
    Path(args.output).with_suffix(".summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(summary)


if __name__=="__main__": main()
