from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from reprsi.benchmarks import prepare_math
from reprsi.curricula import math_items
from reprsi.datasets import download,fetch,source_url
from reprsi.diagnostics import read_jsonl,write_jsonl
from reprsi.loop import Ledger
from reprsi.manufactoria import prepare_released,FactoryTask
from reprsi.prediction import generate_groups
from reprsi.storage import atomic_json,journal
from reprsi.tasks import final_box
from reprsi.workflow import execute


class DatasetTests(unittest.TestCase):
    def test_tex_boxed_argument(self):
        self.assertEqual(final_box(r"Answer $\boxed 2$."),"2")
        self.assertEqual(final_box(r"\boxed 9"),"9")
        self.assertEqual(final_box(r"\boxed {\frac{1}{2}}"),r"\frac{1}{2}")
        self.assertEqual(final_box(r"\boxed{\{2\}}"),r"\{2\}")
        self.assertEqual(final_box(r"\boxed{1} then \boxed 2"),"2")
        self.assertIsNone(final_box(r"\boxed{1"))
        self.assertIsNone(final_box(r"\boxedtext{1}"))

    def test_math_original_splits_jsonl(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for split,count in (("train",7500),("test",5000)):
                write_jsonl(root/(split+".jsonl"),[{"id":f"{split}/{i}","problem":str(i),"solution":r"\boxed 2"} for i in range(count)])
            report=prepare_math(root,root/"prepared")
            self.assertEqual(report,{"train":6750,"dev":750,"test":5000})
            parts={s:read_jsonl(root/"prepared"/(s+".jsonl")) for s in report}
            self.assertTrue(all(r["answer"]=="2" for rows in parts.values() for r in rows))
            self.assertTrue(all(r["id"].startswith("test/") for r in parts["test"]))
            self.assertFalse({r["id"] for r in parts["train"]}&{r["id"] for r in parts["dev"]})
            write_jsonl(root/"ambiguous.jsonl",[{"problem":"x","solution":r"\boxed{1}"}])
            with self.assertRaisesRegex(ValueError,"split labels"):prepare_math(root/"ambiguous.jsonl",root/"bad")

    def test_checksum_cache_repair_and_source_lock(self):
        body=b'{"problem":"x","answer":"1"}\n'
        sha=hashlib.sha256(body).hexdigest()
        source={"url":"https://example.org/HARP.jsonl","sha256":sha,"count":1,"split":"all"}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);atomic_json(root/"sources.json",{"HARP":{"sources":[source]}})
            with patch("reprsi.datasets.urlopen",side_effect=lambda *a,**k:BytesIO(body)) as request:
                result=download("HARP",root/"data",root/"sources.json")
                self.assertEqual(result["counts"],{"all":1})
                download("HARP",root/"data",root/"sources.json")
                self.assertEqual(request.call_count,1)
                (root/"data/HARP.jsonl").write_text("corrupt")
                download("HARP",root/"data",root/"sources.json")
                self.assertEqual(request.call_count,1)
                cached=fetch(source,root/"data/cache");cached.write_text("corrupt")
                fetch(source,root/"data/cache")
                self.assertEqual(request.call_count,2)
            with patch("reprsi.datasets.urlopen",side_effect=lambda *a,**k:BytesIO(b"wrong")):
                with self.assertRaisesRegex(ValueError,"checksum"):fetch(source,root/"bad-cache")
            atomic_json(root/"sources.json",{"HARP":{"sources":[{**source,"count":2}]}})
            with self.assertRaisesRegex(ValueError,"manifest changed"):download("HARP",root/"data",root/"sources.json")
        with self.assertRaisesRegex(ValueError,"pinned"):
            source_url({"repo_id":"owner/data","revision":"main","filename":"train.parquet"})

    def test_has_preserves_all_released_subfamilies(self):
        labels=["contains_substring","contains_ordered","contains_count"]
        domain=SimpleNamespace(load_released_tasks=lambda rows:[
            FactoryTask("prompt",[{"input":"R","expected_accepted":True}],None,row["id"]) for row in rows])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/"raw.jsonl"
            write_jsonl(source,[{"id":str(i),"problem_family":label} for i,label in enumerate(labels)])
            report=prepare_released(source,root/"train.jsonl","train","HAS",domain)
            self.assertEqual(report["examples"],3)
            self.assertEqual({r["problem_family"] for r in read_jsonl(root/"train.jsonl")},set(labels))
            write_jsonl(source,[{"id":"0","problem_family":"exact_sequence"}])
            with self.assertRaisesRegex(ValueError,"family metadata"):
                prepare_released(source,root/"bad.jsonl","train","HAS",domain)


class RecoveryTests(unittest.TestCase):
    def test_unavailable_gpu_fails_before_model_download(self):
        from reprsi.policy import Policy
        from reprsi.smoke import smoke_config
        with patch("reprsi.policy.torch.cuda.is_available",return_value=False):
            with self.assertRaisesRegex(ValueError,"CUDA device is unavailable"):
                Policy({**smoke_config(),"device":"cuda:0"})

    def test_string_device_ledger(self):
        policy=SimpleNamespace(device="cpu",gpu_count=0,rollout_tokens=0,synchronize=lambda:None)
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(policy,Path(d)/"compute.jsonl")
            with ledger.charge("test"):policy.rollout_tokens+=2
            self.assertEqual(ledger.total,0)
            self.assertEqual(ledger.entries[0]["generated_tokens"],2)

    def test_prediction_resume_keeps_valid_candidates_and_changes_seed(self):
        class Backend:
            device="cpu"
            gpu_count=0
            rollout_tokens=0
            def __init__(self):self.calls=0;self.seeds=[]
            def synchronize(self):pass
            def set_seed(self,seed):self.seeds.append(seed)
            def encode(self,prompt):return [1]
            def sample(self,prompt,schema=None):
                self.calls+=1;self.rollout_tokens+=1
                return SimpleNamespace(text=json.dumps({"items":math_items(2,self.calls)}),truncated=self.calls==2)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"origin.pt").write_text("checkpoint")
            origins=[{"run":0,"origin":str(root/"origin.pt")}];policy=Backend()
            with self.assertRaisesRegex(ValueError,"resume to retry"):
                generate_groups(policy,{},origins,root/"groups",groups_per_run=1,items=2,max_attempts=1)
            saved=journal(root/"groups/attempts.jsonl")
            self.assertEqual([r["valid"] for r in saved],[True,False])
            result=generate_groups(policy,{},origins,root/"groups",groups_per_run=1,items=2,resume=True,max_attempts=1)
            self.assertEqual(result["groups"],1);self.assertEqual(result["invalid_attempts"],1)
            self.assertEqual(policy.calls,5);self.assertEqual(len(set(policy.seeds)),5)
            self.assertEqual(read_jsonl(root/"groups/groups.jsonl")[0]["curricula"][0],saved[0]["items"])
            generate_groups(policy,{},origins,root/"groups",groups_per_run=1,items=2,resume=True)
            self.assertEqual(policy.calls,5)

    def test_workload_and_smoke_plan_are_separate(self):
        project=Path(__file__).resolve().parents[1];spec=project/"configs/experiments/math.json"
        full=execute(spec,plan=True);smoke=execute(spec,plan=True,profile="smoke")
        self.assertEqual(full["workload"]["prediction_trials"],40000)
        self.assertEqual(full["workload"]["prediction_optimizer_steps"],22000000)
        self.assertNotEqual(full["output"],smoke["output"])
        self.assertEqual(smoke["workload"]["prediction_optimizer_steps"],0)
        self.assertIn("download",[s["name"] for s in full["stages"]])
        self.assertIn("probe",[s["name"] for s in smoke["stages"]])
        self.assertFalse(any(s["argv"][0]=="screen" for s in smoke["stages"]))


if __name__=="__main__":unittest.main()
