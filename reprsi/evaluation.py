from pathlib import Path
import time
import numpy as np
from .diagnostics import write_jsonl
from .metrics import seed_for, pass_at_k
from .policy import seed_all
from .storage import digest, atomic_json, append_json, journal, exclusive


def task_fingerprint(tasks):
    return digest([{"id": t.identity, "prompt": t.prompt,
                    "answer": getattr(t, "answer", None), "tests": getattr(t, "tests", None)} for t in tasks])


def greedy_score(policy, tasks):
    if not tasks: raise ValueError("Evaluation set is empty")
    outcomes=[];overflow=0
    for task in tasks:
        try: policy.encode(task.prompt,16384)
        except ValueError:
            outcomes.append(0);overflow+=1;continue
        outcomes.append(int(task.verify(policy.sample(task.prompt,greedy=True,input_limit=16384).text)))
    return {"accuracy":float(np.mean(outcomes)),"examples":len(tasks),"input_overflow":overflow,
            "outcomes":outcomes,"ids":[t.identity for t in tasks]}


def evaluate(policy, config, records, tasks, output, samples=32, greedy=False,
             screening=False, resume=False, checkpoint_id="initial", save_responses=False):
    if samples<1 or len(records)!=len(tasks): raise ValueError("Invalid evaluation dimensions")
    if screening and checkpoint_id!="initial": raise ValueError("Screening requires the initial model")
    if screening: samples=128;greedy=False
    elif greedy: samples=1
    ids=[str(t.identity) for t in tasks]
    if len(set(ids))!=len(ids): raise ValueError("Duplicate evaluation identity")
    splits={r.get("split") for r in records}
    if len(splits)>1: raise ValueError("Evaluate each split independently")
    output=Path(output);state_dir=output.with_suffix(".state")
    protocol={"model":config["model"],"revision":config["revision"],"seed":2026 if screening else config["seed"],
        "checkpoint":checkpoint_id,"screening":screening,"greedy":greedy,"samples":samples,
        "input_limit":16384,"output_limit":config["output_limit"],"save_responses":save_responses,
        "data_digest":digest(records),"task_digest":task_fingerprint(tasks)}
    signature=digest(protocol)
    with exclusive(state_dir):
        metadata=state_dir/"protocol.json"
        if metadata.exists():
            if not resume: raise ValueError("Evaluation exists; use --resume")
            import json
            if json.loads(metadata.read_text())["signature"]!=signature:
                raise ValueError("Evaluation data, checkpoint, or decoding protocol changed")
        else:
            if resume: raise ValueError("No evaluation journal to resume")
            if output.exists(): raise ValueError("Output file already exists")
            atomic_json(metadata,{"signature":signature,**protocol})
        log=state_dir/"samples.jsonl"
        saved=journal(log,repair=True);seen={};cost=0.;wall=0.;tokens=0
        for row in saved:
            key=(row["example"],row["sample"])
            if key in seen or not 0<=key[0]<len(tasks) or not 0<=key[1]<samples:
                raise ValueError("Invalid or duplicate evaluation journal record")
            if row["id"]!=ids[key[0]] or row["success"] not in (0,1):
                raise ValueError("Evaluation journal content mismatch")
            seen[key]=row;cost+=row["gpu_seconds"];wall+=row["wall_seconds"];tokens+=row["generated_tokens"]
        for i,task in enumerate(tasks):
            try: policy.encode(task.prompt,16384);overflow=False
            except ValueError: overflow=True
            for sample in range(samples):
                key=(i,sample)
                if key in seen: continue
                purpose="screening" if screening else "final_evaluation"
                seed=seed_for(2026 if screening else config["seed"],purpose,config["model"],task.identity,sample)
                seed_all(seed,policy);policy.synchronize();begin=time.perf_counter();before_tokens=policy.rollout_tokens
                rollout=None
                if not overflow:
                    rollout=policy.sample(task.prompt,greedy=greedy,input_limit=16384)
                success=0 if overflow else int(task.verify(rollout.text))
                if success not in (0,1): raise ValueError("Verifier returned a non-binary reward")
                policy.synchronize();elapsed=time.perf_counter()-begin
                row={"example":i,"id":ids[i],"sample":sample,"seed":seed,"success":success,
                     "input_overflow":overflow,"truncated":False if rollout is None else rollout.truncated,
                     "wall_seconds":elapsed,"gpu_seconds":elapsed*policy.gpu_count,
                     "generated_tokens":policy.rollout_tokens-before_tokens}
                if save_responses: row["response"]="" if rollout is None else rollout.text
                append_json(log,row);seen[key]=row
                cost+=row["gpu_seconds"];wall+=elapsed;tokens+=row["generated_tokens"]
        outcomes=[]
        for i,record in enumerate(records):
            group=[seen[(i,j)] for j in range(samples)]
            success=[r["success"] for r in group]
            row={**record,"input_overflow":any(r["input_overflow"] for r in group),
                 "model":config["model"],"revision":config["revision"],
                 ("screening_outcomes" if screening else "outcomes"):success}
            if screening: row["screening_successes"]=sum(success)
            outcomes.append(row)
        write_jsonl(output,outcomes)
        if screening:
            hard=[r for r in outcomes if r["screening_successes"]==0 and not r["input_overflow"]]
            write_jsonl(output.with_suffix(".hard.jsonl"),hard)
            scores={"hard_count":len(hard)}
        else:
            ks=[1] if greedy else [k for k in (1,4,8,16,24,32,64,128) if k<=samples]
            scores={f"pass@{k}":float(np.mean([pass_at_k(samples,sum(r["outcomes"]),k) for r in outcomes]))
                    if outcomes else None for k in ks}
        summary={**scores,"model":config["model"],"revision":config["revision"],"seed":config["seed"],
            "method":config.get("method","reprsi"),"benchmark":records[0].get("benchmark","unknown") if records else "empty",
            "split":next(iter(splits),None),"data_digest":digest(records),"task_digest":task_fingerprint(tasks),
            "checkpoint":checkpoint_id,"greedy":greedy,"screening":screening,"examples":len(tasks),"samples":samples,
            "input_overflow":sum(r["input_overflow"] for r in outcomes),"wall_seconds":wall,"gpu_seconds":cost,
            "generated_tokens":tokens,"cost_role":"shared_training_preparation" if screening and splits and splits<={"train","dev"} else "evaluation_only"}
        atomic_json(output.with_suffix(".summary.json"),summary)
        return summary
