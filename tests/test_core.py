import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from reprsi.metrics import cohesion,teacher_rewards,rloo,choose_branch,pass_at_k,seed_for
from reprsi.tasks import construct,safe_expression
from reprsi.schema import EXAMPLES,curriculum_schema
from reprsi.diagnostics import make_math_pool,sample_batch,validate_pools
from reprsi.policy import sequence_terms,Rollout,Policy
from reprsi.loop import run
from reprsi.smoke import smoke_config,TinyPolicy


class NumericalTests(unittest.TestCase):
    def test_state_balancing(self):
        records=[];h=[]
        for state,count,vector in [("a",2,[1,0]),("b",9,[0,1])]:
            for i in range(count):
                records.append(dict(family="x",specification="s",state=state,context=str(i)))
                h.append(vector)
        self.assertAlmostEqual(cohesion(h,records)["phi"],1.0)
        np.testing.assert_allclose(cohesion(np.array(h)*7,records)["phi"],1.0)

    def test_rewards_and_ties(self):
        np.testing.assert_allclose(teacher_rewards([1,3,None]),[-1,1,-5])
        np.testing.assert_allclose(rloo([-1,1,-5]),[1,4,-5])
        np.testing.assert_allclose(teacher_rewards([2,None]),[0,-5])
        np.testing.assert_allclose(teacher_rewards([2,2]),[0,0])
        self.assertIsNone(teacher_rewards([None,None]))
        self.assertEqual(choose_branch([None,-0.1,-0.1]),1)
        self.assertAlmostEqual(pass_at_k(4,1,2),0.5)
        self.assertEqual(pass_at_k(32,1,32),1)

    def test_masked_probability_and_kl(self):
        logits=torch.tensor([[1.,99.,2.],[3.,99.,1.]],requires_grad=True)
        ref=torch.zeros_like(logits)
        lp,kl=sequence_terms(logits,ref,[0,2],[[0,2],[0,2]])
        expected=torch.log_softmax(logits[:,[0,2]],-1)
        self.assertAlmostEqual(lp.item(),(expected[0,0]+expected[1,1]).item())
        self.assertGreater(kl.item(),0)
        (-lp+0.01*kl).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[:,1].abs().sum()),0)

    def test_constructor_outputs(self):
        for item,answer in zip(EXAMPLES,["5","2*z + 2","6","(1, 1)"]):
            task=construct(item)
            self.assertEqual(task.answer,answer)
            self.assertEqual(task.verify("work \\boxed{"+answer+"}"),1)
            self.assertEqual(task.verify("\\boxed{999}"),0)
        task=construct(EXAMPLES[0])
        self.assertEqual(task.verify("\\boxed{5} then \\boxed{6}"),0)
        self.assertEqual(task.verify("5"),0)
        with self.assertRaises(ValueError):safe_expression("__import__('os').system('echo unsafe')")

    def test_reject_invalid_graph(self):
        bad=copy.deepcopy(EXAMPLES[0]);bad["composition"][0]["args"][0]="n1"
        with self.assertRaises(ValueError):construct(bad)
        bad=copy.deepcopy(EXAMPLES[0]);bad["parameters"]["a"]=[1,0]
        with self.assertRaises(ValueError):construct(bad)
        bad=copy.deepcopy(EXAMPLES[2]);bad["parameters"]["b"]=9
        with self.assertRaises(ValueError):construct(bad)

    def test_pool_sizes_and_splits(self):
        pools=[make_math_pool(s,n) for s,n in [("reward",64),("monitor",16),("calibration",4)]]
        validate_pools(pools)
        self.assertEqual(list(map(len,pools)),[4096,1024,256])
        batch=sample_batch(pools[0],4,0)
        self.assertEqual(len(batch),256)
        self.assertEqual(batch,sample_batch(pools[0],4,0))


class StateTests(unittest.TestCase):
    def test_bfloat16_master_weights_survive_restore(self):
        p=TinyPolicy(smoke_config())
        p.model.to(torch.bfloat16);p.reference.to(torch.bfloat16);p.reset_optimizer(1e-6)
        before=[x.detach().clone() for x in p.master_parameters]
        rollouts=[p.sample("Compute 1+1") for _ in range(4)]
        p.update(rollouts,[-2,-1,1,2],1e-6)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,p.master_parameters)))
        self.assertTrue(all(x.dtype==torch.float32 for x in p.master_parameters))
        self.assertTrue(all(s["exp_avg"].dtype==torch.float32 for s in p.optimizer.state.values()))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"bf16.pt";p.save(path)
            saved=[x.detach().clone() for x in p.master_parameters]
            p.reset_optimizer(1e-6);p.load(path)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(saved,p.master_parameters)))

    def test_restore_parameters_and_optimizer(self):
        p=TinyPolicy(smoke_config())
        initial=copy.deepcopy(p.model.state_dict())
        samples=[p.sample("What is 1+1?") for _ in range(4)]
        p.update(samples,[-1,0,1,2],0.001)
        self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in p.model.state_dict().items()))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"state.pt";p.save(path)
            model=copy.deepcopy(p.model.state_dict());optim=copy.deepcopy(p.optimizer.state_dict())
            p.update(samples,[1,0,-1,-2],0.001);p.load(path)
            for key,value in model.items():self.assertTrue(torch.equal(value,p.model.state_dict()[key]))
            restored=p.optimizer.state_dict()
            for key,state in optim["state"].items():
                for name,value in state.items():self.assertTrue(torch.equal(value,restored["state"][key][name]))

    def test_huggingface_grammar_sampling_and_replay(self):
        from tokenizers import Tokenizer,models,pre_tokenizers,decoders
        from transformers import PreTrainedTokenizerFast,GPT2Config,GPT2LMHeadModel
        alphabet=sorted(pre_tokenizers.ByteLevel.alphabet())
        vocab={x:i+3 for i,x in enumerate(alphabet)}
        vocab.update({"<eos>":0,"<pad>":1,"<unk>":2})
        tokenizer=Tokenizer(models.BPE(vocab=vocab,merges=[],unk_token="<unk>"))
        tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder=decoders.ByteLevel()
        tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,eos_token="<eos>",pad_token="<pad>",unk_token="<unk>")
        tok.chat_template="{{ messages[0]['content'] }}\nAnswer: "
        model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tok),n_embd=16,n_layer=1,n_head=2,n_positions=256,eos_token_id=0,pad_token_id=1,bos_token_id=None))
        cfg=smoke_config();cfg["output_limit"]=64;cfg["input_limit"]=128
        p=Policy(cfg,model,tok)
        schema={"type":"object","properties":{"x":{"type":"string","enum":["a","b"]}},"required":["x"],"additionalProperties":False}
        sample=p.sample("JSON only",schema=schema)
        self.assertFalse(sample.truncated)
        self.assertIn(json.loads(sample.text)["x"],["a","b"])
        masks=p._allowed(sample)
        for token,mask in zip(sample.completion_ids,masks):self.assertIn(token,mask)
        result=p.update([sample],[1.0],0.001)
        self.assertGreater(result["grad_norm"],0)
        # Replay a complete, real four-family curriculum through the production schema.
        completion=tok.encode(json.dumps({"items":EXAMPLES}),add_special_tokens=False)+[tok.eos_token_id]
        fixture=Rollout(tok.encode("JSON only"),completion,"",schema=curriculum_schema(4))
        for token,mask in zip(completion,p._allowed(fixture)):self.assertIn(token,mask)

    def test_branch_origin_paired_seeds_and_fixed_replicate(self):
        class FakePolicy:
            device=SimpleNamespace(type="cpu")
            rollout_tokens=0
            def __init__(self):self.value=0;self.optimizer_step=0;self.trials=[];self.proposals=0
            def synchronize(self):pass
            def save(self,path):Path(path).write_text(json.dumps([self.value,self.optimizer_step]))
            def load(self,path):self.value,self.optimizer_step=json.loads(Path(path).read_text())
            def sample(self,prompt,schema=None):
                k=self.proposals;self.proposals+=1
                return SimpleNamespace(text=str(k),truncated=False)
            def encode(self,prompt):return [1]
            def hidden(self,batch,layer):return self.value
            def train_curriculum(self,tasks,seed):
                k=tasks[0].candidate;a=len(self.trials)%2
                self.trials.append((k,a,seed,self.value,self.optimizer_step))
                self.value+=k+1+10*a;self.optimizer_step+=1
                return 0.5
            def update(self,proposals,advantages,lr):return {"loss":0}
        domain=SimpleNamespace(teacher_prompt=lambda n:"fixed",curriculum_schema=lambda n:{},
            build_curriculum=lambda text,n:[SimpleNamespace(prompt="training",candidate=int(text))])
        cfg=smoke_config();cfg["rounds"]=1
        p=FakePolicy()
        with tempfile.TemporaryDirectory() as output,patch("reprsi.loop.cohesion",lambda x,b:{"phi":x}):
            run(cfg,p,make_math_pool("reward",1),output,0,domain)
            self.assertEqual(p.value,3) # candidate 2, replicate 0; replicate 1 is 13
            self.assertEqual(p.optimizer_step,1)
            self.assertTrue(all(origin==0 and opt==0 for _,_,_,origin,opt in p.trials))
            for a in range(2):self.assertEqual(len({seed for _,r,seed,_,_ in p.trials if r==a}),1)
        p=FakePolicy()
        def invalid(text,n):raise ValueError("Invalid test fixture")
        domain.build_curriculum=invalid
        def unexpected(*args):raise AssertionError("All-invalid groups must skip updates")
        p.update=unexpected
        with tempfile.TemporaryDirectory() as output,patch("reprsi.loop.cohesion",lambda x,b:{"phi":x}):
            run(cfg,p,make_math_pool("reward",1),output,0,domain)
            self.assertEqual(p.value,0);self.assertEqual(p.optimizer_step,0);self.assertEqual(p.trials,[])


if __name__=="__main__":unittest.main()
