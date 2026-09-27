from dataclasses import dataclass
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from .tasks import final_box
from .diagnostics import read_jsonl,write_jsonl

HARP_REVISION="dac2734ff6443bcaf3bbdcb10f13cf21ae9729c2"


def prepare_math(root,output):
    root=Path(root)
    def load(split):
        rows=[]
        for path in sorted((root/split).rglob("*.json")):
            item=json.loads(path.read_text());answer=final_box(item["solution"])
            if answer is None: raise ValueError(f"Missing boxed reference: {path.name}")
            rows.append({"id":path.relative_to(root).as_posix(),"problem":item["problem"],"answer":answer})
        return rows
    train,test=load("train"),load("test")
    if len(train)!=7500 or len(test)!=5000: raise ValueError("Expected original MATH: 7500/5000")
    if len({r["id"] for r in train+test})!=len(train)+len(test):raise ValueError("Duplicate MATH identity")
    random.Random(2026).shuffle(train)
    for split,rows in [("train",train[:6750]),("dev",train[6750:]),("test",test)]:
        write_jsonl(Path(output)/f"{split}.jsonl",[{**r,"split":split,"benchmark":"MATH"} for r in rows])


def prepare_harp(path,output):
    rows=read_jsonl(path)
    if len(rows)!=4780: raise ValueError("Expected 4780 HARP short-answer problems")
    if len({(str(r["year"]),str(r["contest"]),str(r["number"])) for r in rows})!=4780:raise ValueError("Duplicate HARP identity")
    rows.sort(key=lambda r:(int(re.match(r"\d{4}",str(r["year"])).group()),str(r["contest"]),int(r["number"])))
    random.Random(2026).shuffle(rows)
    for split,part in [("train",rows[:3442]),("dev",rows[3442:3824]),("test",rows[3824:])]:
        converted=[{"id":f"{r['year']}/{r['contest']}/{r['number']}","problem":r["problem"],
             "answer":r["answer"],"split":split,"benchmark":"HARP"} for r in part]
        write_jsonl(Path(output)/f"{split}.jsonl",converted)


class HARPChecker:
    def __init__(self,root):
        if not root: raise ValueError("Supply the pinned HARP checkout with --harp-root")
        root=Path(root)
        revision=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        if revision!=HARP_REVISION: raise ValueError("HARP checkout does not match the paper revision")
        if subprocess.run(["git","-C",str(root),"diff","--quiet","HEAD","--","src"]).returncode:
            raise ValueError("Pinned HARP checker source has local modifications")
        sys.path.insert(0,str(root/"src"))
        from eval.latex_answer_check import check_one_latex_answer
        from eval.utils import run_with_timeout
        self.check=check_one_latex_answer;self.timeout=run_with_timeout

    def __call__(self,response,reference):
        answer=final_box(response)
        if answer is None: return 0.0
        result=self.timeout(self.check,10,{"is_correct":False},answer,reference,extract_policy="none")
        return float(result["is_correct"])


@dataclass
class BenchmarkTask:
    identity:str
    prompt:str
    answer:str
    checker:object

    def verify(self,response): return self.checker(response,self.answer)


def validate_targets(rows,required_split=None,require_hard=False,model=None,revision=None,allow_empty=False):
    if not rows and not allow_empty: raise ValueError("Empty target manifest")
    ids=[r["id"] for r in rows]
    if len(set(ids))!=len(ids): raise ValueError("Duplicate target identity")
    if len({r.get("benchmark") for r in rows})>1: raise ValueError("Mixed benchmark manifest")
    if len({r.get("split") for r in rows})>1: raise ValueError("Mixed data partitions")
    if required_split and any(r.get("split")!=required_split for r in rows):
        raise ValueError(f"Expected {required_split} records only")
    if require_hard:
        for r in rows:
            results=r.get("screening_outcomes",[])
            if r.get("screening_successes")!=0 or len(results)!=128 or any(type(x) is not int or x!=0 for x in results):
                raise ValueError("Target reward needs a fixed fail@128 training manifest")
            if r.get("input_overflow"):
                raise ValueError("A token-overflow item is not a verified fail@128 example")
            if model and r.get("model")!=model: raise ValueError("Screening backbone mismatch")
            if revision and r.get("revision")!=revision: raise ValueError("Screening revision mismatch")


def load_targets(path,checker,required_split=None,require_hard=False,model=None,revision=None,allow_empty=False):
    rows=read_jsonl(path)
    validate_targets(rows,required_split,require_hard,model,revision,allow_empty)
    return [BenchmarkTask(r["id"],"Solve the problem and put the final answer in a single boxed expression.\n"+r["problem"],r["answer"],checker) for r in rows]
