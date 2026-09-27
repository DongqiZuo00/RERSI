from contextlib import contextmanager
from pathlib import Path
import json
import time
import numpy as np
from .diagnostics import sample_batch,validate_pools
from .metrics import cohesion,seed_for,teacher_rewards,rloo,choose_branch
from .policy import seed_all
from .schema import teacher_prompt,curriculum_schema
from .tasks import build_curriculum
from .storage import RunState,exclusive,digest,atomic_json,append_json,journal,link_or_copy
from .evaluation import task_fingerprint,greedy_score
from .curricula import human_curriculum

METHODS=("reprsi","matched-search","target","uncertainty","shuffled","prompted","direct","human")


def hardware_signature(policy):
    import torch
    devices={str(policy.device),str(getattr(policy,"reference_device",policy.device))}
    return {"torch":torch.__version__,"cuda":torch.version.cuda,
            "devices":[torch.cuda.get_device_name(torch.device(x)) if x.startswith("cuda") else x for x in sorted(devices)],
            "dtype":str(next(policy.model.parameters()).dtype) if hasattr(policy,"model") else "test_backend"}


class Ledger:
    def __init__(self,policy,path):
        self.policy=policy;self.path=Path(path)
        self.entries=journal(self.path,repair=True)
        self.total=sum(r["gpu_seconds"] for r in self.entries)

    def preparation(self,seconds):
        if seconds<0 or not np.isfinite(seconds):raise ValueError("Invalid preparation cost")
        if any(r["stage"]=="shared_preparation" for r in self.entries):return
        self.total+=seconds
        row={"stage":"shared_preparation","round":-1,"wall_seconds":0.,"gpu_seconds":seconds,
             "cumulative_gpu_seconds":self.total,"generated_tokens":0,"complete":True}
        append_json(self.path,row);self.entries.append(row)

    @contextmanager
    def charge(self,stage,round_index=-1):
        self.policy.synchronize();begin=time.perf_counter();tokens=self.policy.rollout_tokens;complete=False
        try:
            yield
            complete=True
        finally:
            self.policy.synchronize();elapsed=time.perf_counter()-begin
            gpu_seconds=elapsed*getattr(self.policy,"gpu_count",int(self.policy.device.type=="cuda"))
            self.total+=gpu_seconds
            row={"stage":stage,"round":round_index,"wall_seconds":elapsed,"gpu_seconds":gpu_seconds,
                 "cumulative_gpu_seconds":self.total,"generated_tokens":self.policy.rollout_tokens-tokens,"complete":complete}
            append_json(self.path,row);self.entries.append(row)


def method_for(config):
    method=config.get("method")
    if method is None:
        method="target" if config.get("feedback")=="target" else ("reprsi" if config.get("teacher_updates",True) else "matched-search")
    if method not in METHODS:raise ValueError("Unknown method")
    return method


def validate_config(config):
    for key in ("rounds","items","replicates","student_steps","prompts_per_step","input_limit","output_limit"):
        if type(config[key]) is not int or config[key]<1:raise ValueError(f"{key} must be a positive integer")
    if config["candidates"]<2 or config["completions"]<2:raise ValueError("RLOO groups need at least two samples")
    if config["student_steps"]*config["prompts_per_step"]!=2*config["items"]:
        raise ValueError("Recursive trials must traverse the ordered curriculum exactly twice")
    if any(config[k]<=0 for k in ("teacher_lr","student_lr")) or config["kl_coef"]<0:raise ValueError("Invalid optimizer settings")
    if config.get("max_gpu_seconds") is not None and config["max_gpu_seconds"]<=config.get("preparation_gpu_seconds",0):
        raise ValueError("Training budget must exceed preparation cost")


def _observe(policy,student,pool,layer,t,config,ledger,eval_ledger,eval_tasks):
    row={}
    interval=int(config.get("monitor_every",5));evaluation_interval=int(config.get("eval_every",5))
    observe_monitor=pool and interval and t%interval==0
    observe_eval=eval_tasks and evaluation_interval and t%evaluation_interval==0
    if observe_monitor or observe_eval:policy.load(student)
    if observe_monitor:
        with ledger.charge("monitor_diagnostic",t):row["monitor"]=cohesion(policy.hidden(pool,layer),pool)
    if observe_eval:
        with eval_ledger.charge("heldout_greedy",t):row["evaluation"]=greedy_score(policy,eval_tasks)
    return row


def run(config,policy,pool,output,layer,domain=None,target_tasks=None,resume=False,monitor=None,eval_tasks=None):
    config=dict(config);validate_config(config);method=method_for(config);config["method"]=method
    config["readout_layer"]=layer
    config["hardware"]=hardware_signature(policy)
    if method in ("target","direct") and not target_tasks:raise ValueError("This method requires a nonempty target TRAIN split")
    if any(r["split"]!="reward" for r in pool):raise ValueError("Only reward diagnostics enter feedback")
    if monitor and any(r["split"]!="monitor" for r in monitor):raise ValueError("Monitoring split required")
    validate_pools([x for x in (pool,monitor) if x])
    if not pool and method in ("reprsi","matched-search","shuffled"):raise ValueError("Cohesion feedback needs reward diagnostics")
    if config.get("max_gpu_seconds") and not getattr(policy,"gpu_count",0):raise ValueError("GPU-time matching cannot run on a CPU-only backend")
    inputs={"reward":digest(pool),"monitor":digest(monitor or []),"targets":task_fingerprint(target_tasks or []),
            "evaluation":task_fingerprint(eval_tasks or [])}
    with exclusive(output):
        state=RunState(output,policy,config,inputs,resume)
        ledger=Ledger(policy,Path(output)/"compute.jsonl");ledger.preparation(config.get("preparation_gpu_seconds",0))
        evaluation_ledger=Ledger(policy,Path(output)/"evaluation_compute.jsonl")
        if state.completed==0 and not (Path(output)/"initial_metrics.json").exists():
            metrics=_observe(policy,state.student,monitor,layer,0,config,ledger,evaluation_ledger,eval_tasks)
            atomic_json(Path(output)/"initial_metrics.json",{"completed_rounds":0,"cumulative_gpu_seconds":ledger.total,**metrics})
        if method in ("direct","prompted","human"):
            result=_sequential(config,policy,state,domain,target_tasks,ledger,evaluation_ledger,monitor,eval_tasks,layer)
        else:
            result=_recursive(config,policy,state,pool,layer,domain,target_tasks,ledger,evaluation_ledger,monitor,eval_tasks)
        policy.load(state.student);budget=config.get("max_gpu_seconds")
        result.update(method=method,model=config["model"],revision=config["revision"],seed=config["seed"],
            gpu_seconds=ledger.total,evaluation_gpu_seconds=evaluation_ledger.total,preparation_gpu_seconds=config.get("preparation_gpu_seconds",0),
            max_gpu_seconds=budget,budget_overshoot_seconds=max(0,ledger.total-budget) if budget else 0,
            inputs=inputs,gpu_count=getattr(policy,"gpu_count",0),cost_role="training",
            hardware=config["hardware"],
            unit="round" if method not in ("direct","human") else "student_step")
        atomic_json(Path(output)/"summary.json",result)
        return result


def _budget_reached(config,ledger):
    return config.get("max_gpu_seconds") is not None and ledger.total>=config["max_gpu_seconds"]


def _recursive(config,policy,state,pool,layer,domain,targets,ledger,eval_ledger,monitor,eval_tasks):
    method=config["method"]
    prompt=domain.teacher_prompt(config["items"]) if domain else teacher_prompt(config["items"])
    schema=domain.curriculum_schema(config["items"]) if domain else curriculum_schema(config["items"])
    builder=domain.build_curriculum if domain else build_curriculum
    reserved={r["prompt"] for r in pool+(monitor or [])}
    previous=state.current/"record.json"
    student_steps=json.loads(previous.read_text()).get("student_steps",0) if previous.exists() else 0
    for t in range(state.completed,config["rounds"]):
        if _budget_reached(config,ledger):break
        seed=seed_for(config["seed"],"round",t);seed_all(seed)
        batch=sample_batch(pool,config["specs_per_family"],seed_for(seed,"diagnostic")) if pool else []
        reward_batch=[]
        if method=="target":
            rng=np.random.default_rng(seed_for(seed,"targets"));size=config.get("target_batch",256)
            reward_batch=[targets[i] for i in rng.choice(len(targets),size=size,replace=len(targets)<size)]
        def metric():
            if method=="target":return greedy_score(policy,reward_batch)["accuracy"]
            return cohesion(policy.hidden(batch,layer),batch)["phi"]
        with ledger.charge("teacher_generation",t):
            policy.load(state.teacher);proposals=[policy.sample(prompt,schema=schema) for _ in range(config["candidates"])]
        before=None
        if method!="uncertainty":
            with ledger.charge("baseline_diagnostic",t):policy.load(state.student);before=metric()
        raw=[];details=[];best=None
        selected=state.work/"selected.pt";first=state.work/"trial_first.pt"
        for k,proposal in enumerate(proposals):
            detail={"candidate":k,"specification":proposal.text,"truncated":proposal.truncated}
            with ledger.charge("construction",t):
                try:
                    if proposal.truncated:raise ValueError("Truncated curriculum")
                    seed_all(seed_for(config["seed"],"construction",t,k));tasks=builder(proposal.text,config["items"])
                    if any(x.prompt in reserved for x in tasks):raise ValueError("Diagnostic item reused for training")
                    for task in tasks:policy.encode(task.prompt)
                except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,RecursionError) as exc:
                    raw.append(None);detail["invalid_reason"]=str(exc);details.append(detail);continue
            uncertainty=None
            if method=="uncertainty":
                with ledger.charge("uncertainty_estimation",t):
                    policy.load(state.student);seed_all(seed_for(config["seed"],"uncertainty",t))
                    frequencies=[np.mean([task.verify(policy.sample(task.prompt).text) for _ in range(4)]) for task in tasks]
                    uncertainty=float(np.mean([4*p*(1-p) for p in frequencies]))
            changes=[];training=[]
            for replicate in range(config["replicates"]):
                trial_seed=seed_for(config["seed"],"trial",t,replicate)
                with ledger.charge("student_trial",t):
                    policy.load(state.student);training.append(policy.train_curriculum(tasks,trial_seed))
                    if replicate==0:policy.save(first)
                if method!="uncertainty":
                    with ledger.charge("endpoint_diagnostic",t):changes.append(metric()-before)
            reward=uncertainty if method=="uncertainty" else float(np.mean(changes))
            raw.append(reward);detail.update(progress=changes,raw_reward=reward,training_reward=training,uncertainty=uncertainty)
            details.append(detail)
            if best is None or reward>raw[best]:
                with ledger.charge("branch_save",t):link_or_copy(first,selected)
                best=k
        chosen=choose_branch(raw);assigned=list(raw)
        if method=="shuffled":
            valid=[i for i,v in enumerate(raw) if v is not None]
            permutation=np.random.default_rng(seed_for(2027,"teacher_reward_shuffle",t)).permutation(valid)
            for i,j in zip(valid,permutation):assigned[i]=raw[j]
        normalized=teacher_rewards(assigned);teacher_stats=None;teacher=state.teacher;student=state.student
        if normalized is not None:
            student=selected
            student_steps+=config["student_steps"]
            if method!="matched-search":
                with ledger.charge("teacher_update",t):
                    policy.load(state.teacher);teacher_stats=policy.update(proposals,rloo(normalized),config["teacher_lr"])
                    teacher=state.work/"teacher.pt";policy.save(teacher)
        observation=_observe(policy,student,monitor,layer,t+1,config,ledger,eval_ledger,eval_tasks)
        row={"round":t,"diagnostic_ids":[r["id"] for r in batch],"target_ids":[x.identity for x in reward_batch],
             "before":before,"raw_rewards":raw,"teacher_raw_rewards":assigned,
             "normalized_rewards":None if normalized is None else normalized.tolist(),"selected_candidate":chosen,
             "selected_replicate":0 if chosen is not None else None,"teacher_update":teacher_stats,"candidates":details,
             "cumulative_gpu_seconds":ledger.total,"student_steps":student_steps,
             "trial_steps_this_round":sum(v is not None for v in raw)*config["replicates"]*config["student_steps"],**observation}
        with ledger.charge("commit",t):state.commit(teacher,student,row)
        print(f"round={t} method={method} selected={chosen} rewards={raw}",flush=True)
    return {"completed_rounds":state.completed,"stop_reason":"budget" if _budget_reached(config,ledger) else "round_limit"}


def _sequential(config,policy,state,domain,targets,ledger,eval_ledger,monitor,eval_tasks,layer):
    method=config["method"]
    limit=config.get("max_units",config["rounds"] if method=="prompted" else config["rounds"]*config["student_steps"])
    builder=domain.build_curriculum if domain else build_curriculum
    previous=state.current/"record.json"
    student_steps=json.loads(previous.read_text()).get("student_steps",0) if previous.exists() else 0
    observation_config=dict(config)
    if method in ("direct","human"):
        observation_config["eval_every"]=config.get("eval_every",5)*config["student_steps"]
        observation_config["monitor_every"]=config.get("monitor_every",5)*config["student_steps"]
    for t in range(state.completed,limit):
        if _budget_reached(config,ledger):break
        seed=seed_for(config["seed"],"sequential",t);seed_all(seed);detail={}
        with ledger.charge("curriculum_preparation",t):
            if method=="prompted":
                policy.load(state.teacher)
                proposal=policy.sample(domain.teacher_prompt(config["items"]) if domain else teacher_prompt(config["items"]),
                    schema=domain.curriculum_schema(config["items"]) if domain else curriculum_schema(config["items"]))
                detail["specification"]=proposal.text
                try:
                    if proposal.truncated:raise ValueError("Truncated curriculum")
                    tasks=builder(proposal.text,config["items"])
                    for task in tasks:policy.encode(task.prompt)
                except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError,RecursionError) as exc:
                    tasks=[];detail["invalid_reason"]=str(exc)
            elif method=="direct":
                rng=np.random.default_rng(seed);tasks=[targets[i] for i in rng.choice(len(targets),size=config["prompts_per_step"],replace=True)]
            else:
                fraction=t/limit
                if config.get("max_gpu_seconds"):
                    prep=config.get("preparation_gpu_seconds",0);fraction=(ledger.total-prep)/(config["max_gpu_seconds"]-prep)
                stage=min(2,int(3*fraction));detail["stage"]=stage
                tasks=human_curriculum(config["prompts_per_step"],seed,stage,domain)
        student=state.student
        if tasks:
            with ledger.charge("student_training",t):
                policy.load(state.student)
                if method=="prompted":detail["training_reward"]=policy.train_curriculum(tasks,seed)
                else:detail["training_reward"]=policy.train_batch(tasks)["reward"]
                student=state.work/"student.pt";policy.save(student)
                student_steps+=config["student_steps"] if method=="prompted" else 1
        observation=_observe(policy,student,monitor,layer,t+1,observation_config,ledger,eval_ledger,eval_tasks)
        row={"round":t,"student_steps":student_steps,"cumulative_gpu_seconds":ledger.total,"method":method,"details":detail,**observation}
        with ledger.charge("commit",t):state.commit(state.teacher,student,row)
        print(f"unit={t} method={method} valid={bool(tasks)}",flush=True)
    return {"completed_rounds":state.completed,"stop_reason":"budget" if _budget_reached(config,ledger) else "unit_limit"}
