from pathlib import Path
import numpy as np
from .storage import exclusive,RunState,atomic_json,append_json,journal,digest
from .policy import seed_all
from .metrics import seed_for
from .schema import curriculum_schema,teacher_prompt
from .tasks import build_curriculum
from .curricula import task_record
from .diagnostics import write_jsonl
from .evaluation import greedy_score,task_fingerprint
from .loop import Ledger


def export_teacher(policy,config,output,count=32,domain=None,resume=False,checkpoint_id="initial",sampling_seed=2027):
    if count<1:raise ValueError("Positive curriculum count required")
    path=Path(output);directory=path.with_suffix(".state")
    signature=digest({"config":{k:config[k] for k in ("model","revision","domain","items","input_limit","output_limit")},
                      "checkpoint":checkpoint_id,"count":count,"seed":sampling_seed})
    with exclusive(directory):
        metadata=directory/"protocol.json"
        if metadata.exists():
            import json
            if not resume or json.loads(metadata.read_text())["signature"]!=signature:raise ValueError("Export already exists or protocol changed")
        else:
            if resume:raise ValueError("No export to resume")
            atomic_json(metadata,{"signature":signature})
        ledger=Ledger(policy,directory/"compute.jsonl");saved=journal(directory/"proposals.jsonl",repair=True)
        valid=[r for r in saved if "tasks" in r]
        builder=domain.build_curriculum if domain else build_curriculum
        prompt=domain.teacher_prompt(config["items"]) if domain else teacher_prompt(config["items"])
        schema=domain.curriculum_schema(config["items"]) if domain else curriculum_schema(config["items"])
        for attempt in range(len(saved),count*20):
            if len(valid)>=count:break
            seed_all(seed_for(sampling_seed,"curriculum_export",attempt))
            with ledger.charge("frozen_teacher_generation",attempt):
                proposal=policy.sample(prompt,schema=schema)
                row={"attempt":attempt,"specification":proposal.text,"truncated":proposal.truncated}
                try:
                    if proposal.truncated:raise ValueError("Truncated curriculum")
                    seed_all(seed_for(sampling_seed,"export_construction",attempt))
                    tasks=builder(proposal.text,config["items"])
                    for task in tasks:policy.encode(task.prompt)
                    row["tasks"]=[task_record(t) for t in tasks]
                except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,RecursionError) as exc:
                    row["invalid_reason"]=str(exc)
            append_json(directory/"proposals.jsonl",row)
            if "tasks" in row:valid.append(row)
        if len(valid)!=count:raise RuntimeError("Could not obtain enough valid curricula within the declared attempt cap")
        rows=[{**task,"curriculum":i,"position":j} for i,row in enumerate(valid) for j,task in enumerate(row["tasks"])]
        write_jsonl(path,rows)
        summary={"curricula":count,"examples":len(rows),"seed":sampling_seed,"teacher_checkpoint":checkpoint_id,
                 "teacher_seed":config["seed"],"task_digest":digest(rows),"gpu_seconds":ledger.total,"cost_role":"fresh_curriculum_preparation"}
        atomic_json(path.with_suffix(".summary.json"),summary)
        return summary


def train_fixed(policy,config,tasks,output,steps=800,eval_every=40,eval_tasks=None,resume=False,
                supervised=False,initial_id="pretrained",teacher_id=None,stages=None):
    if steps<1 or eval_every<1 or not tasks:raise ValueError("Invalid fresh-student schedule")
    cfg={**config,"rounds":steps,"steps":steps,"eval_every":eval_every,"supervised":supervised,
         "initial_id":initial_id,"teacher_id":teacher_id,"keep_every":eval_every}
    inputs={"curriculum":task_fingerprint(tasks),"evaluation":task_fingerprint(eval_tasks or []),
            "stages":[task_fingerprint(x) for x in stages] if stages else None}
    with exclusive(output):
        state=RunState(output,policy,cfg,inputs,resume)
        ledger=Ledger(policy,Path(output)/"compute.jsonl");eval_ledger=Ledger(policy,Path(output)/"evaluation_compute.jsonl")
        if state.completed==0 and eval_tasks and not (Path(output)/"initial_metrics.json").exists():
            with eval_ledger.charge("heldout_greedy",0):metric=greedy_score(policy,eval_tasks)
            atomic_json(Path(output)/"initial_metrics.json",{"student_steps":0,"evaluation":metric})
        for step in range(state.completed,steps):
            seed_all(seed_for(config["seed"],"fresh_student_step",step))
            current=tasks;local_step=step
            if stages:
                stage=min(2,step*3//steps);current=stages[stage];local_step=step-(stage*steps+2)//3
            n=config["prompts_per_step"]
            batch=[current[(local_step*n+j)%len(current)] for j in range(n)]
            with ledger.charge("atomic_preparation" if supervised else "fresh_student_training",step):
                policy.load(state.student)
                stats=policy.supervised_batch(batch) if supervised else policy.train_batch(batch)
                destination=state.work/"student.pt";policy.save(destination)
            row={"student_steps":step+1,"training":stats,"cumulative_gpu_seconds":ledger.total}
            if eval_tasks and ((step+1)%eval_every==0 or step+1==steps):
                with eval_ledger.charge("heldout_greedy",step+1):row["evaluation"]=greedy_score(policy,eval_tasks)
            with ledger.charge("commit",step):state.commit(state.teacher,destination,row)
        policy.load(state.student)
        result={"steps":state.completed,"student_seed":config["seed"],"teacher_id":teacher_id,"inputs":inputs,
                "gpu_seconds":ledger.total,"evaluation_gpu_seconds":eval_ledger.total,"supervised":supervised,
                "model":config["model"],"revision":config["revision"]}
        atomic_json(Path(output)/"summary.json",result)
        return result
