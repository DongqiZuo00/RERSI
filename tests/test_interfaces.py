import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from reprsi.interfaces import make_policy,load_config,TrainingBackend
from reprsi.smoke import TinyPolicy,smoke_config
from reprsi.mechanism import prepare_twohop,TwoHopDomain
from reprsi.curricula import human_curriculum,math_items
from reprsi.diagnostics import make_math_pool,write_jsonl,read_jsonl
from reprsi.tasks import construct,Task
from reprsi.prediction import origins,collect
from reprsi.loop import run
from reprsi.analysis import efficiency_report
from reprsi.storage import atomic_json
from reprsi.workflow import build,execute


class InterfaceTests(unittest.TestCase):
    def test_custom_backend_and_inherited_config(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);atomic_json(root/'base.json',smoke_config())
            atomic_json(root/'custom.json',{'extends':'base.json','backend':'custom','backend_factory':'reprsi.smoke:TinyPolicy'})
            cfg=load_config(root/'custom.json');p=make_policy(cfg)
            self.assertIsInstance(p,TrainingBackend)
            result=run({**cfg,'rounds':1},p,make_math_pool('reward',1),root/'run',0)
            self.assertEqual(result['completed_rounds'],1)
            self.assertTrue((root/'run/evaluation_student.pt').is_file())
            atomic_json(root/'loop.json',{'extends':'loop.json'})
            with self.assertRaisesRegex(ValueError,'cycle'):load_config(root/'loop.json')

    def test_twohop_human_and_recursive_interfaces(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prepare_twohop(root/'data');domain=TwoHopDomain(root/'data')
            for stage,limit in enumerate((8,16,32)):
                tasks=human_curriculum(8,17,stage,domain)
                self.assertTrue(all(int(t.identity.rsplit('C',1)[1])<limit for t in tasks))
                self.assertTrue(all(t.verify(t.answer)==1 for t in tasks))
            cfg={**smoke_config(),'domain':'twohop','rounds':1,'prompts_per_step':1,'specs_per_family':1}
            pool=read_jsonl(root/'data/reward.jsonl')
            result=run(cfg,TinyPolicy(cfg),pool,root/'recursive',0,domain)
            row=json.loads((root/'recursive/round_000000.json').read_text())
            self.assertIsNotNone(row['selected_candidate'])
            result=run({**cfg,'method':'human','max_units':1},TinyPolicy(cfg),pool,root/'human',0,domain)
            self.assertEqual(result['completed_rounds'],1)

    def test_hf_local_checkpoint_with_separate_generation_limits(self):
        from tokenizers import Tokenizer,models,pre_tokenizers,decoders
        from transformers import PreTrainedTokenizerFast,GPT2Config,GPT2LMHeadModel
        alphabet=sorted(pre_tokenizers.ByteLevel.alphabet());vocab={x:i+3 for i,x in enumerate(alphabet)}
        vocab.update({'<eos>':0,'<pad>':1,'<unk>':2})
        tokenizer=Tokenizer(models.BPE(vocab=vocab,merges=[],unk_token='<unk>'))
        tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tokenizer.decoder=decoders.ByteLevel()
        tok=PreTrainedTokenizerFast(tokenizer_object=tokenizer,eos_token='<eos>',pad_token='<pad>',unk_token='<unk>')
        tok.chat_template="{{ messages[0]['content'] }}\nAnswer: "
        model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tok),n_embd=16,n_layer=1,n_head=2,n_positions=256,eos_token_id=0,pad_token_id=1,bos_token_id=None))
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d);tok.save_pretrained(d)
            cfg={**smoke_config(),'backend':'hf','model':d,'revision':'local','dtype':'float32',
                'local_files_only':True,'output_limit':2,'teacher_output_limit':64,'input_limit':128,'decoder_layers':'transformer.h'}
            p=make_policy(cfg)
            schema={'type':'object','properties':{'x':{'type':'string','enum':['a','b']}},'required':['x'],'additionalProperties':False}
            teacher=p.sample('JSON only',schema=schema)
            self.assertFalse(teacher.truncated)
            self.assertGreater(len(teacher.completion_ids),2)
            self.assertIn(json.loads(teacher.text)['x'],['a','b'])
            self.assertLessEqual(len(p.sample('Answer').completion_ids),2)
            stats=p.update([teacher],[1.],.001);self.assertGreater(stats['grad_norm'],0)
            record={'prompt':'1 + 1','prefix':'1 + 1'}
            total=p.answer_logprob(record,'12');mean=p.answer_logprob(record,'12',normalize=True)
            n=len(p.tokenizer.encode('\\boxed{12}',add_special_tokens=False))
            self.assertAlmostEqual(mean,total/n,places=5)

    def test_prediction_math_training_and_resume(self):
        cfg=smoke_config();targets=[Task('Compute 5+5.','10','rational','train/1')]
        evaluation=[Task('Compute 6+6.','12','rational','test/1')]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=TinyPolicy(cfg)
            origins(p,cfg,targets,None,root/'origins',runs=1,steps=1)
            groups=[{'run':0,'group':0,'origin':str(root/'origins/run_000/student.pt'),
                'curricula':[math_items(2,100+i) for i in range(4)]}]
            pool=make_math_pool('reward',1)
            result=collect(TinyPolicy(cfg),cfg,pool,targets,evaluation,groups,None,root/'trial',0,trials=1,curriculum_steps=1,continuation_steps=1)
            self.assertEqual(result['trials'],4)
            rows=read_jsonl(root/'trial/trials.jsonl')
            self.assertTrue(all(r['likelihood_normalization']=='mean' for r in rows))
            result=collect(TinyPolicy(cfg),cfg,pool,targets,evaluation,groups,None,root/'trial',0,trials=1,curriculum_steps=1,continuation_steps=1,resume=True)
            self.assertEqual(result['trials'],4)
            with self.assertRaisesRegex(ValueError,'protocol'):
                collect(TinyPolicy(cfg),cfg,pool,targets,[Task('Changed','1','rational','test/1')],groups,None,root/'trial',0,trials=1,curriculum_steps=1,continuation_steps=1,resume=True)


class BudgetTests(unittest.TestCase):
    def test_comparison_recovers_after_committed_round(self):
        from reprsi.loop import _comparison_checkpoint
        cfg={**smoke_config(),'rounds':1}
        with tempfile.TemporaryDirectory() as d:
            def interrupted(state,ledger,config,cost=None):
                if state.completed:raise RuntimeError('interrupted comparison export')
                return _comparison_checkpoint(state,ledger,config,cost)
            with patch('reprsi.loop._comparison_checkpoint',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    run(cfg,TinyPolicy(cfg),make_math_pool('reward',1),d,0)
            result=run(cfg,TinyPolicy(cfg),make_math_pool('reward',1),d,0,resume=True)
            self.assertEqual(result['comparison_checkpoint']['completed_rounds'],1)
            self.assertEqual(result['comparison_checkpoint']['wall_seconds'],result['wall_seconds'])
            a=torch.load(Path(d)/'evaluation_student.pt',weights_only=True)
            b=torch.load(Path(d)/'student.pt',weights_only=True)
            self.assertTrue(all(torch.equal(a['model'][k],v) for k,v in b['model'].items()))

    def test_budget_excludes_overrun_checkpoint(self):
        cfg={**smoke_config(),'rounds':3,'max_gpu_seconds':.001}
        p=TinyPolicy(cfg);p.gpu_count=1
        with tempfile.TemporaryDirectory() as d:
            result=run(cfg,p,make_math_pool('reward',1),d,0)
            self.assertEqual(result['stop_reason'],'budget')
            self.assertGreater(result['budget_overshoot_seconds'],0)
            self.assertLessEqual(result['comparison_checkpoint']['gpu_seconds'],cfg['max_gpu_seconds'])
            self.assertEqual(result['comparison_checkpoint']['completed_rounds'],0)
            a=torch.load(Path(d)/'evaluation_student.pt',weights_only=True)
            b=torch.load(Path(d)/'initial.pt',weights_only=True)
            self.assertTrue(all(torch.equal(a['model'][k],v) for k,v in b['model'].items()))

    def test_efficiency_threshold_and_censoring(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);directories=[]
            for method,values in [('target',[20,30]),('reprsi',[30,40]),('direct',[10,10])]:
                for seed in range(2):
                    out=root/method/str(seed);directories.append(out)
                    atomic_json(out/'summary.json',{'method':method,'seed':seed,'gpu_seconds':7200,'wall_seconds':3600,
                        'hardware':{'devices':['gpu','gpu']},'comparison_checkpoint':{'gpu_seconds':7000,'wall_seconds':3500,'student_steps':16},'budget_overshoot_seconds':200})
                    atomic_json(out/'greedy.summary.json',{'model':'m','revision':'r','task_digest':'same','benchmark':'MATH','split':'test','samples':1,'greedy':True,'pass@1':values[-1]/100})
                    for step,value in enumerate(values):
                        atomic_json(out/f'round_{step:06d}.json',{'student_steps':8*(step+1),'cumulative_gpu_seconds':3600*(step+1),
                            'cumulative_wall_seconds':1800*(step+1),'evaluation':{'accuracy':value/100}})
            report=efficiency_report(directories)
            self.assertEqual(report['target_accuracy_percent'],30)
            self.assertEqual(report['methods']['reprsi']['matched_gpu_hours']['mean'],1)
            self.assertEqual(report['methods']['direct']['matched_reached'],0)
            self.assertIsNone(report['methods']['direct']['matched_gpu_hours'])


class WorkflowTests(unittest.TestCase):
    def test_all_experiment_plans(self):
        root=Path(__file__).resolve().parents[1]
        for path in (root/'configs/experiments').glob('*.json'):
            result=execute(path,plan=True)
            self.assertTrue(any(s['argv'][0]=='train' for s in result['stages']))
            self.assertTrue(any(s['argv'][0]=='efficiency' for s in result['stages']))
            self.assertEqual(len({s['name'] for s in result['stages']}),len(result['stages']))
            for stage in result['stages']:
                if stage['argv'][0]=='evaluate':self.assertTrue(any('evaluation_student.pt' in x for x in stage['argv']))

    def test_cli_workflow_with_real_tiny_training(self):
        project=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg={**smoke_config(),'checker_factory':'reprsi.smoke:exact_checker','monitor_every':1,'eval_every':1}
            atomic_json(root/'model.json',cfg)
            for split in ('train','test'):
                write_jsonl(root/'data'/(split+'.jsonl'),[{'id':split+'/0','split':split,'benchmark':'MATH','problem':'Compute 5+5.','answer':'10'}])
            for split in ('reward','monitor','calibration'):write_jsonl(root/'diagnostics'/(split+'.jsonl'),make_math_pool(split,1))
            spec={'workspace':str(project),'config':str(root/'model.json'),'output':str(root/'experiment'),
                'data':{'prepared':str(root/'data')},'diagnostics':{'prepared':str(root/'diagnostics')},
                'seeds':[0],'rounds':1,'methods':['reprsi'],'calibration':{'batches':1,'layers':[0]},
                'evaluation':{'samples':2,'hard_samples':2},
                'fresh_student':{'enabled':True,'methods':['reprsi'],'curricula':1,'steps':1,'eval_every':1,'seeds':[0]}}
            atomic_json(root/'experiment.json',spec)
            result=execute(root/'experiment.json')
            self.assertGreater(result['completed_stages'],10)
            self.assertTrue((root/'experiment/reports/fresh_reprsi.json').exists())
            resumed=execute(root/'experiment.json',resume=True)
            self.assertEqual(result['completed_stages'],resumed['completed_stages'])


if __name__=='__main__':unittest.main()
