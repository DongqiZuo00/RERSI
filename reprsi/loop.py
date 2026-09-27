"""Algorithm: shared origin -> paired trials -> teacher RLOO -> fixed continuation."""
from contextlib import contextmanager
from pathlib import Path
import json
import shutil
import time
import numpy as np
from .diagnostics import sample_batch,write_jsonl
from .metrics import cohesion,seed_for,teacher_rewards,rloo,choose_branch
from .policy import seed_all
from .schema import teacher_prompt,curriculum_schema
from .tasks import build_curriculum


class Ledger:
    def __init__(self,policy,path):
        self.policy=policy;self.path=Path(path);self.total=0.0
        if self.path.exists():
            for line in self.path.read_text().splitlines(): self.total+=json.loads(line)["gpu_seconds"]

    def preparation(self,seconds):
        if seconds<0:raise ValueError("Negative preparation cost")
        self.total+=seconds
        with self.path.open("a") as stream:
            stream.write(json.dumps({"stage":"shared_preparation","round":-1,"wall_seconds":None,
                "gpu_seconds":seconds,"cumulative_gpu_seconds":self.total,"generated_tokens":0})+"\n")

    @contextmanager
    def charge(self,stage,round_index=-1):
        self.policy.synchronize();begin=time.perf_counter()
        tokens=self.policy.rollout_tokens
        try: yield
        finally:
            self.policy.synchronize();elapsed=time.perf_counter()-begin
            gpu_seconds=elapsed*getattr(self.policy,"gpu_count",int(self.policy.device.type=="cuda"))
            self.total+=gpu_seconds
            entry={"stage":stage,"round":round_index,"wall_seconds":elapsed,
                "gpu_seconds":gpu_seconds,"cumulative_gpu_seconds":self.total,
                "generated_tokens":self.policy.rollout_tokens-tokens}
            with self.path.open("a") as stream: stream.write(json.dumps(entry)+"\n")


def run(config,policy,pool,output,layer,domain=None,target_tasks=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/"progress.json").exists():
        raise ValueError("Output contains a run. Use a new directory to avoid overwriting checkpoints")
    if any(r["split"]!="reward" for r in pool): raise ValueError("Only reward split may enter training")
    if config["candidates"]<2 or config["replicates"]<1 or config["completions"]<2:
        raise ValueError("Invalid comparison-group sizes")
    if config.get("feedback","cohesion")=="target" and not target_tasks:
        raise ValueError("Target feedback requires the fixed hard TRAINING subset")
    origin=output/"initial.pt"
    policy.save(origin)
    teacher=output/"teacher.pt";student=output/"student.pt"
    shutil.copyfile(origin,teacher);shutil.copyfile(origin,student)
    ledger=Ledger(policy,output/"compute.jsonl")
    ledger.preparation(config.get("preparation_gpu_seconds",0.0))
    prompt=domain.teacher_prompt(config["items"]) if domain else teacher_prompt(config["items"])
    schema=domain.curriculum_schema(config["items"]) if domain else curriculum_schema(config["items"])
    builder=domain.build_curriculum if domain else build_curriculum
    # Save a portable configuration; user-specified paths are deliberately omitted.
    metadata={k:v for k,v in config.items() if k not in {"delta_root","harp_root"}}
    metadata["readout_layer"]=layer
    (output/"config.json").write_text(json.dumps(metadata,indent=2)+"\n")
    reserved_prompts={r["prompt"] for r in pool}
    for t in range(config["rounds"]):
        seed=seed_for(config["seed"],"round",t)
        seed_all(seed)
        batch=sample_batch(pool,config["specs_per_family"],seed_for(seed,"diagnostic"))
        reward_batch=None
        if target_tasks:
            rng=np.random.default_rng(seed_for(seed,"targets"))
            indices=rng.choice(len(target_tasks),size=config.get("target_batch",256),replace=len(target_tasks)<config.get("target_batch",256))
            reward_batch=[target_tasks[i] for i in indices]

        def metric():
            if config.get("feedback","cohesion")=="target":
                return float(np.mean([task.verify(policy.sample(task.prompt,greedy=True,input_limit=16384).text) for task in reward_batch]))
            return cohesion(policy.hidden(batch,layer),batch)["phi"]

        with ledger.charge("teacher_generation",t):
            policy.load(teacher)
            proposals=[policy.sample(prompt,schema=schema) for _ in range(config["candidates"])]
        with ledger.charge("baseline_diagnostic",t):
            policy.load(student);before=metric()
        raw=[];details=[];best=None
        selected_path=output/"selected.pt";first_path=output/"trial_first.pt"
        for k,proposal in enumerate(proposals):
            detail={"candidate":k,"specification":proposal.text,"truncated":proposal.truncated}
            with ledger.charge("construction",t):
                try:
                    if proposal.truncated: raise ValueError("Truncated curriculum")
                    seed_all(seed_for(config["seed"],"construction",t,k))
                    tasks=builder(proposal.text,config["items"])
                    if any(x.prompt in reserved_prompts for x in tasks): raise ValueError("Reward diagnostic reused as training item")
                    # Prompt-length failures invalidate the entire proposal before trials.
                    for task in tasks: policy.encode(task.prompt)
                except (ValueError,TypeError,KeyError,IndexError,ZeroDivisionError) as exc:
                    raw.append(None);detail["invalid_reason"]=str(exc);details.append(detail)
                    continue
            changes=[];train_rewards=[]
            for a in range(config["replicates"]):
                # Candidate index intentionally ABSENT: seeds are paired across curricula.
                trial_seed=seed_for(config["seed"],"trial",t,a)
                with ledger.charge("student_trial",t):
                    policy.load(student)
                    train_rewards.append(policy.train_curriculum(tasks,trial_seed))
                    if a==0: policy.save(first_path)
                with ledger.charge("endpoint_diagnostic",t): changes.append(metric()-before)
            reward=float(np.mean(changes));raw.append(reward)
            detail.update(progress=changes,raw_reward=reward,training_reward=train_rewards)
            details.append(detail)
            if best is None or reward>raw[best]:
                shutil.copyfile(first_path,selected_path);best=k
        chosen=choose_branch(raw)
        normalized=teacher_rewards(raw)
        teacher_stats=None
        if normalized is not None:
            if config.get("teacher_updates",True):
                with ledger.charge("teacher_update",t):
                    policy.load(teacher)
                    teacher_stats=policy.update(proposals,rloo(normalized),config["teacher_lr"])
                    policy.save(teacher)
            # No extra trial: continue the ALREADY TRAINED first replicate, with optimizer.
            selected_path.replace(student)
        row={"round":t,"diagnostic_ids":[r["id"] for r in batch],"before":before,
             "raw_rewards":raw,"normalized_rewards":None if normalized is None else normalized.tolist(),
             "selected_candidate":chosen,"selected_replicate":0 if chosen is not None else None,
             "teacher_update":teacher_stats,"candidates":details,"cumulative_gpu_seconds":ledger.total}
        write_jsonl(output/f"round_{t:03d}.jsonl",[row])
        progress={"completed_rounds":t+1,"student":"student.pt","teacher":"teacher.pt"}
        (output/"progress.json").write_text(json.dumps(progress,indent=2)+"\n")
        if first_path.exists(): first_path.unlink()
        print(f"round={t} selected={chosen} progress={raw}",flush=True)
    policy.load(student)
    return {"rounds":config["rounds"],"gpu_seconds":ledger.total,"output":str(output)}
