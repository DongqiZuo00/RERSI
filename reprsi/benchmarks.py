"""Released MATH/HARP partitions and the pinned official HARP answer checker."""
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
    random.Random(2026).shuffle(train)
    for split,rows in [("train",train[:6750]),("dev",train[6750:]),("test",test)]:
        write_jsonl(Path(output)/f"{split}.jsonl",[{**r,"split":split,"benchmark":"MATH"} for r in rows])


def prepare_harp(path,output):
    rows=read_jsonl(path)
    if len(rows)!=4780: raise ValueError("Expected 4780 HARP short-answer problems")
    # Release contains season-tagged years such as 2021_Fall. Sort by the
    # numeric year; Python's stable sort preserves release order on equal keys.
    rows.sort(key=lambda r:(int(re.match(r"\d{4}",str(r["year"])).group()),str(r["contest"]),int(r["number"])))
    random.Random(2026).shuffle(rows)
    for split,part in [("train",rows[:3442]),("dev",rows[3442:3824]),("test",rows[3824:])]:
        converted=[{"id":f"{r['year']}/{r['contest']}/{r['number']}","problem":r["problem"],
             "answer":r["answer"],"split":split,"benchmark":"HARP"} for r in part]
        write_jsonl(Path(output)/f"{split}.jsonl",converted)


class HARPChecker:
    def __init__(self,root):
        root=Path(root)
        revision=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
        if revision!=HARP_REVISION: raise ValueError("HARP checkout does not match the paper revision")
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


def load_targets(path,checker,required_split=None,require_hard=False):
    rows=read_jsonl(path)
    if required_split and any(r.get("split")!=required_split for r in rows):
        raise ValueError(f"Expected {required_split} records only")
    if require_hard and any(r.get("screening_successes")!=0 or len(r.get("screening_outcomes",[]))!=128 for r in rows):
        raise ValueError("Target reward needs a fixed fail@128 training manifest")
    return [BenchmarkTask(r["id"],"Solve the problem and put the final answer in a single boxed expression.\n"+r["problem"],r["answer"],checker) for r in rows]
