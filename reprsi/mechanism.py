from collections import Counter,defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import numpy as np
from .diagnostics import BOUNDARY,write_jsonl,read_jsonl,validate_pools
from .storage import atomic_json,digest,exclusive,append_json,journal,file_digest
from .metrics import seed_for,cohesion
from .policy import seed_all
from .evaluation import greedy_score
from .experiments import train_fixed
from .loop import Ledger


@dataclass
class EntityTask:
    identity:str
    prompt:str
    answer:str
    family:str="twohop"

    def verify(self,response):
        text=response.strip()
        boxed=re.fullmatch(r"\\boxed\{([ESA]\d+)\}",text)
        if boxed:text=boxed.group(1)
        return float(text==self.answer)


def load_entities(records):
    return [EntityTask(r["id"],r["prompt"],r["answer"]) for r in records]


def prepare_twohop(output,seed=2027):
    rng=np.random.default_rng(seed);states=[f"S{s}" for s in range(6)]
    prefixes={s:[{"entity":f"E{i*24+j:03d}","relation":f"R{j%4}"} for j in range(24)] for i,s in enumerate(states)}
    continuations=[f"C{i:02d}" for i in range(32)]
    table={c:{s:f"A{i*6+int(p):03d}" for s,p in zip(states,rng.permutation(6))} for i,c in enumerate(continuations)}
    queries=[];atomic=[];pools={split:[] for split in ("reward","monitor","calibration")};reserved=set()
    for s in states:
        for index,p in enumerate(prefixes[s]):
            local=f"Apply {p['relation']} to {p['entity']}. Answer with only the resulting entity."
            atomic.append({"id":f"atomic/{p['entity']}/{p['relation']}","prompt":local,"answer":s,"split":"atomic","kind":"entity"})
            for j,c in enumerate(continuations):
                prefix=f"Apply {p['relation']} to {p['entity']}"
                identity=f"{p['entity']}/{p['relation']}/{c}"
                row={"id":identity,"family":"twohop","prompt":prefix+BOUNDARY+f"Apply {c} to the resulting entity. Answer with only the final entity.",
                     "prefix":prefix,"answer":table[c][s],"reference_state":s,"state":s,"context":identity,
                     "continuation":{"kind":"lookup","table":table[c]},"answer_format":"entity","kind":"entity",
                     "local_prompt":local,"entity":p["entity"],"relation":p["relation"],"next_relation":c}
                role=None
                for split,start in (("calibration",8),("reward",12),("monitor",16)):
                    if start<=index<start+4 and j==index:
                        role=split;pools[split].append({**row,"split":split,"specification":f"{split}/twohop"});break
                if role is None:queries.append(row)
                if index<8 and j<8:reserved.add(identity)
        for c in continuations:
            atomic.append({"id":f"atomic/{s}/{c}","prompt":f"Apply {c} to {s}. Answer with only the resulting entity.",
                           "answer":table[c][s],"split":"atomic","kind":"entity"})
    candidates=[r for r in queries if r["id"] not in reserved];rng.shuffle(candidates)
    test_ids={r["id"] for r in candidates[:1000]}
    train=[{**r,"split":"train"} for r in queries if r["id"] not in test_ids]
    test=[{**r,"split":"test"} for r in candidates[:1000]]
    validate_pools(list(pools.values()))
    directory=Path(output)
    for name,rows in {"atomic":atomic,"train":train,"test":test,**pools}.items():write_jsonl(directory/f"{name}.jsonl",rows)
    metadata={"seed":seed,"states":states,"prefixes":prefixes,"continuations":continuations,"table":table,
              "counts":{k:len(v) for k,v in {"atomic":atomic,"train":train,"test":test,**pools}.items()}}
    atomic_json(directory/"definition.json",metadata)
    return metadata["counts"]


class TwoHopDomain:
    def __init__(self,root):
        self.rows=read_jsonl(Path(root)/"train.jsonl");self.lookup={r["id"]:r for r in self.rows}
        if len(self.lookup)!=len(self.rows):raise ValueError("Duplicate controlled task")

    def teacher_prompt(self,count):
        return (f"Generate an ordered curriculum of {count} two-hop queries. Return JSON with an items array of query IDs. "
                "IDs have form E000/R0/C00. Entity identifiers run E000 through E143. The first relation for entity number e is R(e mod 4), "
                "where R0,R1,R2,R3 are valid. Continuations are C00 through C31. Choose varied queries and their order to improve learning. "
                "The grammar restricts IDs to the training partition. No diagnostic or test IDs are permitted.")

    def curriculum_schema(self,count):
        return {"type":"object","properties":{"items":{"type":"array","items":{"type":"string","enum":sorted(self.lookup)},
                "minItems":count,"maxItems":count}},"required":["items"],"additionalProperties":False}

    def build_curriculum(self,text,count):
        obj=json.loads(text)
        if set(obj)!={"items"} or len(obj["items"])!=count:raise ValueError("Wrong two-hop curriculum size")
        if any(x not in self.lookup for x in obj["items"]):raise ValueError("Two-hop curriculum includes a reserved query")
        return load_entities([self.lookup[x] for x in obj["items"]])

    def human_curriculum(self,count,seed,stage):
        if stage not in (0,1,2) or count<1:raise ValueError("Invalid staged curriculum")
        limit=(8,16,32)[stage]
        rows=[r for r in self.rows if int(r["next_relation"][1:])<limit]
        rng=np.random.default_rng(seed)
        return load_entities([rows[int(i)] for i in rng.choice(len(rows),count,replace=count>len(rows))])

    def fixed_calibration_curriculum(self,count,seed):
        rng=np.random.default_rng(seed);indices=rng.choice(len(self.rows),count,replace=count>len(self.rows))
        return load_entities([self.rows[int(i)] for i in indices])


def intervention_batches(root,condition,seed,steps=500):
    if condition not in ("full","reduced","restored"):raise ValueError("Unknown exposure intervention")
    root=Path(root);definition=json.loads((root/"definition.json").read_text());lookup={r["id"]:r for r in read_jsonl(root/"train.jsonl")}
    states=sorted(definition["states"]);rng=np.random.default_rng(2027)
    prefixes={s:list(rng.permutation(sorted(definition["prefixes"][s][:8],key=lambda p:(p["entity"],p["relation"])))) for s in states}
    continuations=list(rng.permutation(sorted(definition["continuations"][:8])))
    blocks=[(s,a,b) for s in states for a in range(4) for b in range(4)]
    order=np.random.default_rng(2027).permutation(len(blocks))
    counts={"full":((2,2),(2,2)),"reduced":((4,0),(0,4)),"restored":((3,1),(1,3))}[condition]
    batches=[]
    for step in range(steps):
        s,a,b=blocks[int(order[step%len(order)])];slots=[]
        for i in range(2):
            prefix=prefixes[s][a+4*i]
            for j in range(2):
                c=continuations[b+4*j];identity=f"{prefix['entity']}/{prefix['relation']}/{c}"
                if identity not in lookup:raise ValueError("Intervention query is not in the training split")
                slots.extend([lookup[identity]]*counts[i][j])
        permutation=np.random.default_rng(seed_for(seed,"within_batch",step)).permutation(8)
        batches.append(load_entities([slots[int(i)] for i in permutation]))
    return batches


def exposure_report(root,steps=500):
    result={};marginals=[]
    for condition in ("full","reduced","restored"):
        tasks=[t for batch in intervention_batches(root,condition,0,steps) for t in batch]
        joint=Counter(t.identity for t in tasks);prefix=Counter();continuation=Counter();state=Counter()
        lookup={r["id"]:r for r in read_jsonl(Path(root)/"train.jsonl")}
        for identity,count in joint.items():
            e,r,c=identity.split("/");prefix[(e,r)]+=count;continuation[(lookup[identity]["state"],c)]+=count;state[lookup[identity]["state"]]+=count
        marginals.append((prefix,continuation,state))
        result[condition]={"presentations":len(tasks),"distinct_queries":len(joint)}
    result["equal_marginals"]=all(m==marginals[0] for m in marginals[1:])
    if not result["equal_marginals"]:raise AssertionError("Exposure marginals differ")
    return result


def measure_mechanism(policy,records,layer,patching=False):
    hidden=policy.hidden(records,layer);result=cohesion(hidden,records)
    z=hidden/np.maximum(np.linalg.norm(hidden,axis=1,keepdims=True),1e-8)
    states=defaultdict(list)
    for i,r in enumerate(records):states[r["state"]].append(i)
    result["per_state"]={}
    for s,ids in sorted(states.items()):
        same=z[ids];other=np.mean([z[j].mean(0) for t,j in states.items() if t!=s],axis=0)
        plus=((same.sum(0)@same.sum(0))-(same*same).sum())/(len(ids)*(len(ids)-1))
        result["per_state"][s]={"phi":float(plus-same.mean(0)@other)}
    if patching:
        labels=sorted(states)
        for index,s in enumerate(labels):
            successes=[];gains=[];recipients=states[labels[(index+1)%len(labels)]]
            for j,source in enumerate(states[s]):
                target=recipients[(j+1)%len(recipients)];record=records[target]
                answer=record["continuation"]["table"][s]
                response=policy.patched_sample(record,layer,hidden[source]).text
                successes.append(EntityTask("",record["prompt"],answer).verify(response))
                gains.append(policy.answer_probability(record,answer,layer,hidden[source])-policy.answer_probability(record,answer))
            result["per_state"][s].update(patch_success=float(np.mean(successes)),counterfactual_probability_gain=float(np.mean(gains)))
    return result


def run_interventions(policy,config,root,initial,output,layer,seeds=range(5),steps=500,continuation_steps=400,resume=False):
    root=Path(root);output=Path(output)
    train=load_entities(read_jsonl(root/"train.jsonl"));test=load_entities(read_jsonl(root/"test.jsonl"));monitor=read_jsonl(root/"monitor.jsonl")
    if config["prompts_per_step"]!=8:raise ValueError("Exposure interventions use batches of eight prompts")
    initial_hash=file_digest(initial);reports=[]
    for seed in seeds:
        for condition in ("full","reduced","restored"):
            policy.reference_tag="pretrained";policy.load(initial);policy.set_reference(initial_hash)
            cfg={**config,"seed":seed};directory=output/f"seed_{seed}"/condition
            tasks=[t for batch in intervention_batches(root,condition,seed,steps) for t in batch]
            train_fixed(policy,cfg,tasks,directory/"exposure",steps,steps,eval_tasks=test,
                        resume=resume and (directory/"exposure/progress.json").exists(),initial_id=initial_hash,teacher_id=condition)
            ledger=Ledger(policy,directory/"diagnostic_compute.jsonl")
            with ledger.charge("mechanism_measurement"):measurement=measure_mechanism(policy,monitor,layer,patching=True)
            policy.reset_optimizer(cfg["student_lr"])
            rng=np.random.default_rng(seed_for(seed,"target_continuation_order"));order=rng.permutation(len(train))
            continuation=[train[int(i)] for i in order]
            train_fixed(policy,cfg,continuation,directory/"continuation",continuation_steps,continuation_steps,test,
                        resume=resume and (directory/"continuation/progress.json").exists(),
                        initial_id=file_digest(directory/"exposure/student.pt"),teacher_id=condition)
            score=greedy_score(policy,test)
            row={"seed":seed,"condition":condition,"measurement":measurement,"future_accuracy":100*score["accuracy"]}
            atomic_json(directory/"result.json",row);reports.append(row)
    atomic_json(output/"results.json",{"exposure":exposure_report(root,steps),"runs":reports})
    return {"runs":len(reports),"output":"results.json"}


def collect_prediction_trials(policy,config,root,groups,initial,output,layer,trials=5,curriculum_steps=150,continuation_steps=400,resume=False):
    from .prediction import collect
    root=Path(root)
    return collect(policy,config,read_jsonl(root/"reward.jsonl"),load_entities(read_jsonl(root/"train.jsonl")),
        load_entities(read_jsonl(root/"test.jsonl")),groups,initial,output,layer,TwoHopDomain(root),
        trials,curriculum_steps,continuation_steps,resume)


def prediction_origins(policy,config,root,initial,output,runs=200,steps=400,resume=False):
    from .prediction import origins
    return origins(policy,config,load_entities(read_jsonl(Path(root)/"train.jsonl")),initial,output,runs,steps,resume)


def prediction_groups(policy,config,root,origins,output,groups_per_run=10,items=100,resume=False):
    from .prediction import generate_groups
    return generate_groups(policy,config,origins,output,TwoHopDomain(root),groups_per_run,items,resume)
