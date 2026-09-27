import json
from pathlib import Path
import tempfile
import unittest
from reprsi.smoke import TinyPolicy,smoke_config
from reprsi.mechanism import (prepare_twohop,load_entities,run_interventions,
                             prediction_origins,collect_prediction_trials)
from reprsi.diagnostics import read_jsonl,write_jsonl
from reprsi.experiments import train_fixed


class ControlledRuntimeTests(unittest.TestCase):
    def test_atomic_intervention_and_prediction_collection(self):
        cfg={**smoke_config(),"domain":"twohop","prompts_per_step":8}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/"data";prepare_twohop(data)
            write_jsonl(data/"test.jsonl",read_jsonl(data/"test.jsonl")[:2])
            p=TinyPolicy(cfg);atomic=load_entities(read_jsonl(data/"atomic.jsonl"))
            stats=train_fixed(p,cfg,atomic,root/"atomic",steps=1,eval_every=1,supervised=True)
            self.assertTrue(stats["supervised"]);initial=root/"atomic/student.pt"
            result=run_interventions(p,cfg,data,initial,root/"interventions",0,seeds=[0],steps=1,continuation_steps=1)
            self.assertEqual(result["runs"],3)
            records=json.loads((root/"interventions/results.json").read_text())
            self.assertTrue(records["exposure"]["equal_marginals"])
            self.assertTrue(all("per_state" in r["measurement"] for r in records["runs"]))
            origins=root/"origins";prediction_origins(p,cfg,data,initial,origins,runs=1,steps=1)
            ids=[r["id"] for r in read_jsonl(data/"train.jsonl")[:8]]
            groups=[{"run":0,"group":0,"origin":str(origins/"run_000/student.pt"),
                     "curricula":[ids[i:i+2] for i in range(0,8,2)]}]
            result=collect_prediction_trials(p,cfg,data,groups,initial,root/"trials",0,trials=1,curriculum_steps=1,continuation_steps=1)
            self.assertEqual(result["trials"],4)
            result=collect_prediction_trials(p,cfg,data,groups,initial,root/"trials",0,trials=1,curriculum_steps=1,continuation_steps=1,resume=True)
            self.assertEqual(result["trials"],4)


if __name__=="__main__":unittest.main()
