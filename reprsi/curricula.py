import json
import random
from .tasks import Task,construct
from .storage import digest


def math_items(count,seed,stage=0):
    if stage not in (0,1,2) or count<1: raise ValueError("Invalid staged curriculum")
    rng=random.Random(seed);items=[]
    families=("rational","polynomial","modular","linear")
    for index in range(count):
        family=families[index%4];depth=(1,2,rng.choice((3,4)))[stage];nodes=[]
        if family=="linear":
            a,b=rng.randint(-4,4),rng.randint(-4,4)
            params={"a":{"matrix":[[1,0,a],[0,1,b]]},"b":0,"c":1,"d":rng.choice((-2,-1,1,2))}
            for j in range(depth-1):
                nodes.append({"id":f"n{j}","op":"row_add","args":["a" if j==0 else f"n{j-1}","b","c","d"]})
            nodes.append({"id":f"n{depth-1}","op":"solve","args":["a" if depth==1 else f"n{depth-2}"]})
        else:
            if family=="rational":params={"a":[rng.randint(-20,20),rng.randint(1,20)],"b":[rng.randint(1,20),rng.randint(1,20)]}
            elif family=="polynomial":params={"a":{"poly":[rng.randint(-9,9) for _ in range(rng.randint(2,5))]},"b":{"poly":[rng.randint(-9,9) for _ in range(3)]}}
            else:params={"a":rng.randint(-20,20),"b":rng.randint(1,8),"modulus":rng.choice((2,3,5,7,11,13,17,19,23,29,31))}
            for j in range(depth):
                choices=("add","sub","mul","div") if family=="rational" else (("add","sub","pow") if family=="modular" else ("add","sub","diff"))
                op=rng.choice(choices);args=["a" if j==0 else f"n{j-1}"]
                if op!="diff":args.append("b")
                nodes.append({"id":f"n{j}","op":op,"args":args})
        item={"family":family,"parameters":params,"composition":nodes,"presentation":rng.choice(("symbolic","verbal"))}
        construct(item);items.append(item)
    return items


def human_curriculum(count,seed,stage,domain=None):
    if domain is None:return [construct(x) for x in math_items(count,seed,stage)]
    rng=random.Random(seed);examples={x["family"]:x for x in domain.example_items()}
    families=("START","APPEND","EXACT") if stage==0 else (("REGEX",) if stage==1 else ("HAS",))
    tasks=[]
    for i in range(count):
        item=json.loads(json.dumps(examples[families[i%len(families)]]));p=item["parameters"]
        if "sequence" in p:p["sequence"]="".join(rng.choices("RB",k=rng.randint(2,4)))
        elif "pattern_parts" in p:p["pattern_parts"]=[["".join(rng.choices("RB",k=2)),rng.choice(("+","*"))] for _ in range(2)]
        tasks.append(domain.construct(item))
    return tasks


def task_record(task):
    if hasattr(task,"tests"):
        return {"id":task.identity,"messages":[{"role":"user","content":task.prompt}],"ground_truth":task.tests,"kind":"manufactoria"}
    return {"id":task.identity or digest({"prompt":task.prompt,"answer":task.answer}),"prompt":task.prompt,
            "answer":task.answer,"family":getattr(task,"family","math"),"kind":"exact_math"}


def load_frozen(records,domain=None,checker=None):
    if not records:raise ValueError("Frozen curriculum is empty")
    if domain:
        if any(r.get("kind")!="manufactoria" for r in records):raise ValueError("Frozen curriculum domain mismatch")
        return domain.load_released_tasks(records)
    if all(r.get("kind")=="benchmark_math" for r in records):
        if checker is None:raise ValueError("Frozen benchmark tasks require the official checker")
        from .benchmarks import BenchmarkTask
        return [BenchmarkTask(r["id"],r["prompt"],r["answer"],checker) for r in records]
    if any(r.get("kind")!="exact_math" for r in records):raise ValueError("Expected exact constructor tasks")
    return [Task(r["prompt"],r["answer"],r["family"],r["id"]) for r in records]
