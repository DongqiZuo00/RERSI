import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from reprsi.smoke import TinyPolicy,smoke_config
from reprsi.loop import run
from reprsi.storage import atomic_json,RunState,digest
from reprsi.metrics import seed_for
from reprsi.diagnostics import make_math_pool,read_jsonl
from reprsi.policy import seed_all
from reprsi.tasks import Task,construct,safe_expression
from reprsi.curricula import math_items
from reprsi.evaluation import evaluate
from reprsi.benchmarks import validate_targets
from reprsi.mechanism import prepare_twohop,exposure_report,TwoHopDomain,intervention_batches
from reprsi.analysis import aggregate,ranking_accuracy,prediction_analysis,first_crossing
from reprsi.experiments import train_fixed,export_teacher


class RecoveryTests(unittest.TestCase):
    def test_interrupted_round_replays_identically(self):
        cfg=smoke_config();pool=make_math_pool("reward",1)
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);full=TinyPolicy(cfg)
            run(cfg,full,pool,root/"full",0)
            interrupted=TinyPolicy(cfg);real_save=interrupted.save;failed=False
            def fail_once(path):
                nonlocal failed
                if Path(path).parent.name=="working" and Path(path).name=="teacher.pt" and not failed:
                    failed=True;raise RuntimeError("simulated interruption before commit")
                real_save(path)
            interrupted.save=fail_once
            with self.assertRaisesRegex(RuntimeError,"simulated"):
                run(cfg,interrupted,pool,root/"resume",0)
            self.assertEqual(json.loads((root/"resume/progress.json").read_text())["completed_rounds"],0)
            resumed=TinyPolicy(cfg);run(cfg,resumed,pool,root/"resume",0,resume=True)
            for name,value in full.model.state_dict().items():self.assertTrue(torch.equal(value,resumed.model.state_dict()[name]),name)
            for key,value in full.optimizer.state_dict()["state"].items():
                for field,tensor in value.items():self.assertTrue(torch.equal(tensor,resumed.optimizer.state_dict()["state"][key][field]))
            cfg_changed={**cfg,"student_lr":.01}
            with self.assertRaisesRegex(ValueError,"fingerprint"):
                run(cfg_changed,TinyPolicy(cfg_changed),pool,root/"resume",0,resume=True)

    def test_commit_pointer_recovers_stale_aliases(self):
        cfg=smoke_config();p=TinyPolicy(cfg)
        with tempfile.TemporaryDirectory() as d:
            state=RunState(d,p,cfg,{})
            sample=[p.sample("1+1") for _ in range(4)];p.update(sample,[-1,0,1,2],.01)
            path=state.work/"trained.pt";p.save(path)
            with patch.object(state,"_aliases",side_effect=RuntimeError("alias interruption")):
                with self.assertRaises(RuntimeError):state.commit(state.teacher,path,{"round":0})
            recovered=RunState(d,p,cfg,{},True)
            self.assertEqual(recovered.completed,1)
            p.load(Path(d)/"student.pt")
            expected=torch.load(path if path.exists() else recovered.student,weights_only=True)
            for key,value in expected["model"].items():self.assertTrue(torch.equal(value,p.model.state_dict()[key]))

    def test_rng_restore_and_checkpoint_backbone_guard(self):
        cfg=smoke_config();p=TinyPolicy(cfg)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"state.pt";seed_all(6);p.save(path)
            first=torch.rand(4);p.load(path,restore_random=True);self.assertTrue(torch.equal(first,torch.rand(4)))
            wrong=TinyPolicy({**cfg,"revision":"different"})
            with self.assertRaisesRegex(ValueError,"mismatch"):wrong.load(path)

    def test_evaluation_resumes_without_repeating_completed_answers(self):
        class Evaluator:
            gpu_count=0;rollout_tokens=0
            def __init__(self,fail=None):self.calls=0;self.fail=fail
            def synchronize(self):pass
            def encode(self,*args):return [1]
            def sample(self,*args,**kwargs):
                self.calls+=1
                if self.calls==self.fail:raise RuntimeError("interrupted evaluation")
                self.rollout_tokens+=1;return SimpleNamespace(text="\\boxed{1}",truncated=False)
        rows=[{"id":"a","split":"test","benchmark":"fixture"}];tasks=[Task("one","1","rational","a")]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"scores.jsonl";cfg=smoke_config();p=Evaluator(3)
            with self.assertRaises(RuntimeError):evaluate(p,cfg,rows,tasks,path,samples=4)
            p=Evaluator();result=evaluate(p,cfg,rows,tasks,path,samples=4,resume=True)
            self.assertEqual(p.calls,2);self.assertEqual(result["pass@4"],1)
            self.assertEqual(result["generated_tokens"],4)
            with self.assertRaisesRegex(ValueError,"protocol changed"):
                evaluate(p,cfg,rows,tasks,path,samples=5,resume=True)


class ProtocolTests(unittest.TestCase):
    def test_invalid_hard_manifest_is_rejected(self):
        row={"id":"a","benchmark":"MATH","split":"train","screening_successes":0,"screening_outcomes":[0]*128,"model":"m","revision":"r"}
        validate_targets([row],"train",True,"m","r")
        bad=copy.deepcopy(row);bad["screening_outcomes"][60]=1
        with self.assertRaises(ValueError):validate_targets([bad],"train",True)
        with self.assertRaises(ValueError):validate_targets([{**row,"split":"test"}],"train",True)
        with self.assertRaises(ValueError):validate_targets([{**row,"input_overflow":True}],"train",True)
        with self.assertRaises(ValueError):validate_targets([row],"train",True,"other","r")

    def test_large_exact_answer_and_bounded_polynomial(self):
        self.assertEqual(Task("x",str(20**16),"rational").verify("\\boxed{"+str(20**16)+"}"),1)
        with self.assertRaises(ValueError):safe_expression("((z+1)**16)**16")
        for stage in range(3):
            items=math_items(40,17,stage)
            self.assertEqual({x["family"] for x in items},{"rational","polynomial","modular","linear"})
            self.assertTrue(all(len(x["composition"]) in ((1,),(2,),(3,4))[stage] for x in items))
            for item in items:
                task=construct(item);self.assertEqual(task.verify("\\boxed{"+task.answer+"}"),1)

    def test_twohop_splits_and_exact_exposure_marginals(self):
        with tempfile.TemporaryDirectory() as d:
            counts=prepare_twohop(d);self.assertEqual(counts["test"],1000)
            pools=[read_jsonl(Path(d)/(s+".jsonl")) for s in ("train","test","reward","monitor","calibration")]
            ids=[{r["id"] for r in x} for x in pools]
            self.assertEqual(len(set.union(*ids)),sum(map(len,ids)))
            report=exposure_report(d);self.assertTrue(report["equal_marginals"])
            self.assertEqual([report[c]["distinct_queries"] for c in ("full","reduced","restored")],[384,192,384])
            self.assertTrue(all(report[c]["presentations"]==4000 for c in ("full","reduced","restored")))
            domain=TwoHopDomain(d)
            with self.assertRaises(ValueError):domain.build_curriculum(json.dumps({"items":[pools[1][0]["id"]]}),1)

    def test_uncertainty_and_shuffled_selection_semantics(self):
        cfg={**smoke_config(),"rounds":1,"replicates":1,"student_steps":4}
        pool=make_math_pool("reward",1)
        for method in ("uncertainty","shuffled","matched-search","target","prompted","direct","human"):
            c={**cfg,"method":method,"max_units":1}
            tasks=[Task("Compute 1+1","2","rational","target")]
            with tempfile.TemporaryDirectory() as d:
                result=run(c,TinyPolicy(c),pool,d,0,target_tasks=tasks)
                self.assertEqual(result["completed_rounds"],1)
                row=json.loads((Path(d)/"round_000000.json").read_text())
                if method=="uncertainty":
                    self.assertTrue(all(x is None or 0<=x<=1 for x in row["raw_rewards"]))
                    self.assertEqual(row["selected_candidate"],int(np.argmax(row["raw_rewards"])))
                if method=="shuffled":
                    self.assertEqual(sorted(row["raw_rewards"]),sorted(row["teacher_raw_rewards"]))
                    self.assertEqual(row["selected_candidate"],int(np.argmax(row["raw_rewards"])))
                if method=="matched-search":self.assertIsNone(row["teacher_update"])

    def test_frozen_export_and_fresh_training(self):
        from reprsi.curricula import load_frozen
        cfg=smoke_config();p=TinyPolicy(cfg)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);result=export_teacher(p,cfg,d/"curricula.jsonl",count=2)
            self.assertEqual(result["examples"],4)
            tasks=load_frozen(read_jsonl(d/"curricula.jsonl"))
            fresh=TinyPolicy(cfg);result=train_fixed(fresh,cfg,tasks,d/"fresh",steps=2,eval_every=1,eval_tasks=tasks)
            self.assertEqual(result["steps"],2)
            result=train_fixed(TinyPolicy(cfg),cfg,tasks,d/"fresh",steps=2,eval_every=1,eval_tasks=tasks,resume=True)
            self.assertEqual(result["steps"],2)


class ReportingTests(unittest.TestCase):
    def test_paired_aggregation_and_ties(self):
        common={"model":"m","revision":"r","task_digest":"d","samples":32,"greedy":False,"benchmark":"b","split":"test"}
        a=[{**common,"seed":i,"method":"reprsi","pass@1":.1+i/100} for i in range(5)]
        b=[{**common,"seed":i,"method":"direct","pass@1":.08+i/100} for i in range(5)]
        report=aggregate(a,b);self.assertAlmostEqual(report["paired_differences"]["pass@1"]["mean"],2)
        self.assertAlmostEqual(report["paired_differences"]["pass@1"]["sd"],0)
        self.assertTrue(report["paper_five_seed_complete"])
        self.assertEqual(ranking_accuracy([1,2,2],[1,1,1],["g"]*3),{"accuracy":50.,"pairs":2})
        self.assertEqual(first_crossing([(0,10),(20,12),(40,11),(60,15)],13),60)
        self.assertIsNone(first_crossing([(0,10)],12))
        with self.assertRaises(ValueError):aggregate(a,b[:-1])

    def test_prediction_runs_never_leak_into_fitting(self):
        rows=[]
        for run_id in range(10):
            for group in range(2):
                for k in range(4):
                    delta=(k-1.5)/10
                    rows.append(dict(run=run_id,group=group,candidate=k,curriculum_digest=f"{run_id}/{group}/{k}",
                        current_accuracy=30+run_id,curriculum_accuracy=40+group,state_accuracy=50+k,
                        delta_phi=delta,future_accuracy=30+run_id+20*delta))
        result=prediction_analysis(rows,repetitions=2)
        self.assertFalse(set(result["train_runs"])&set(result["test_runs"]))
        self.assertEqual(len(result["train_runs"]),8)
        self.assertLess(result["behavioral_plus_cohesion"]["mae_pp"],.01)
        self.assertEqual(result["behavioral_plus_cohesion"]["ranking"]["accuracy"],100.)


if __name__=="__main__":unittest.main()
