import json
from copy import deepcopy
import subprocess
import sys
from pathlib import Path
from .interfaces import load_config
from .storage import atomic_json, digest, exclusive


def build(spec, spec_path):
    workspace=(Path(spec_path).resolve().parent/spec.get('workspace','../..')).resolve()
    def path(value):return str((workspace/value).resolve())
    cfg_path=path(spec['config']);cfg=load_config(cfg_path)
    root=Path(path(spec['output']));data=root/'data';diagnostics=root/'diagnostics'
    if spec.get('config_overrides'):
        cfg.update(spec['config_overrides']);cfg_path=str(root/'model.json')
    stages=[];domain=cfg.get('domain','math');sources=spec.get('data',{})
    def add(name,args,outputs,resumable=False,marker=None):
        stages.append({'name':name,'argv':list(map(str,args)),'outputs':list(map(str,outputs)),
            'resumable':resumable,'marker':str(marker) if marker else None})
    dep=[]
    for key in ('harp_root','delta_root'):
        if sources.get(key):
            dep+=['--'+key.replace('_','-'),path(sources[key])]
            if sources.get('setup_dependencies',False):
                add('setup_'+key,['setup-dependency','--name','HARP' if key=='harp_root' else 'DELTA','--output',path(sources[key])],
                    [Path(path(sources[key]))/'src' if key=='harp_root' else Path(path(sources[key]))/'manufactoria'])
    if sources.get('download'):
        raw=root/'raw';command=['download-data','--benchmark',spec['benchmark'],'--output',raw]
        if sources.get('manifest'):command+=['--manifest',path(sources['manifest'])]
        if sources.get('cache'):command+=['--cache',path(sources['cache'])]
        add('download',command,[raw/'manifest.json'])
        if domain=='math':sources={**sources,'source':str(raw/'HARP.jsonl') if spec['benchmark']=='HARP' else str(raw)}
        else:sources={**sources,'train':str(raw/'train.jsonl'),'test':str(raw/'test.jsonl')}
    initial=[];preparation=[]
    if domain=='twohop':
        add('data',['prepare-twohop','--output',data],[data/'definition.json'])
        diagnostics=data;dep+=['--twohop-data',str(data)]
        atomic=root/'atomic'
        add('atomic',['fresh-student','--config',cfg_path,'--curricula',data/'atomic.jsonl','--supervised',
            '--steps',spec.get('atomic_steps',200),'--eval-every',spec.get('atomic_steps',200),
            '--output',atomic,*dep],[atomic/'summary.json'],True,atomic/'progress.json')
        initial=['--initial',str(atomic/'student.pt')];preparation=[str(atomic/'summary.json')]
    else:
        benchmark=spec.get('benchmark','MATH')
        if sources.get('prepared'):
            data=Path(path(sources['prepared']))
        elif domain=='math':
            if benchmark not in ('MATH','HARP'):raise ValueError('Math benchmark must be MATH or HARP')
            add('data',['prepare-'+benchmark.lower(),'--source',path(sources['source']),'--output',data],
                [data/'train.jsonl',data/'test.jsonl'])
        elif domain=='manufactoria':
            for split in ('train','test'):
                add('data_'+split,['prepare-manufactoria','--source',path(sources[split]),'--output',data/(split+'.jsonl'),
                    '--split',split,'--family',sources.get('family','HAS'),*dep],[data/(split+'.jsonl')])
        else:raise ValueError('Custom domains require data.prepared and diagnostics.prepared')
        prepared=spec.get('diagnostics',{}).get('prepared')
        if prepared:diagnostics=Path(path(prepared))
        else:add('diagnostics',['prepare-diagnostics','--domain',domain,'--output',diagnostics,
            *(['--specs-per-family',spec['diagnostics']['specs_per_family']] if spec.get('diagnostics',{}).get('specs_per_family') else [])],
            [diagnostics/'reward.jsonl',diagnostics/'monitor.jsonl',diagnostics/'calibration.jsonl'])
    if spec.get('preflight',True):
        add('probe',['probe','--config',cfg_path,'--diagnostics',diagnostics/'reward.jsonl','--output',root/'probe',*dep],
            [root/'probe/probe.json'])
    if spec.get('data_limit'):
        subset=root/'subset'
        for split in ('train','test'):
            add('subset_'+split,['subset-data','--source',data/(split+'.jsonl'),'--output',subset/(split+'.jsonl'),
                '--count',spec['data_limit']],[subset/(split+'.jsonl')])
        data=subset
    train=data/'train.jsonl';test=data/'test.jsonl';hard=None
    screening=domain=='math' and not spec.get('smoke',False)
    if screening:
        for split in ('train','test'):
            out=root/'screening'/(split+'.jsonl')
            add('screen_'+split,['screen','--config',cfg_path,'--data',data/(split+'.jsonl'),'--output',out,*dep],
                [out.with_suffix('.summary.json'),out.with_suffix('.hard.jsonl')],True,out.with_suffix('.state')/'protocol.json')
        train=root/'screening/train.hard.jsonl';hard=root/'screening/test.hard.jsonl'
        preparation.append(str(root/'screening/train.summary.json'))
    trajectory=root/'calibration_trajectory';calibration=root/'calibration.json'
    cal=spec.get('calibration',{});batches=cal.get('batches',2)
    add('calibration_trajectory',['calibration-trajectory','--config',cfg_path,'--batches',batches,
        '--output',trajectory,*initial,*dep],[trajectory/'summary.json'],True,trajectory/'progress.json')
    checkpoints=[trajectory/f'step_{i*cfg["student_steps"]:03d}.pt' for i in range(batches+1)]
    calcmd=['calibrate','--config',cfg_path,'--diagnostics',diagnostics/'calibration.jsonl','--checkpoints',*checkpoints,
        '--specs-per-family',cal.get('specs_per_family',1),'--output',calibration,*initial,*dep]
    if cal.get('layers') is not None:calcmd+=['--layers',*cal['layers']]
    add('calibration',calcmd,[calibration],True,calibration.with_suffix('.progress.json'))
    methods=spec.get('methods',['reprsi','direct','prompted','matched-search','target','uncertainty','shuffled','human'])
    seeds=spec.get('seeds',list(range(5)))
    if not methods or methods[0]!='reprsi' or len(set(methods))!=len(methods):raise ValueError('Unique methods starting with reprsi are required')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Unique seeds are required')
    if len(methods)>1 and cfg.get('backend')=='tiny':raise ValueError('GPU compute matching requires a GPU backend')
    rounds=spec.get('rounds',cfg['rounds']);samples=spec.get('evaluation',{}).get('samples',32)
    for seed in seeds:
        for method in methods:
            out=root/'benchmark'/method/f'seed_{seed}'
            cmd=['train','--config',cfg_path,'--diagnostics',diagnostics/'reward.jsonl','--monitor',diagnostics/'monitor.jsonl',
                '--calibration',calibration,*(['--target-train',train] if not spec.get('smoke',False) else []),'--eval-data',test,'--seed',seed,'--method',method,
                '--output',out,*initial,*dep]
            if method=='reprsi':cmd+=['--rounds',rounds]
            else:cmd+=['--budget-from',root/'benchmark/reprsi'/f'seed_{seed}'/'summary.json']
            if preparation:cmd+=['--preparation-costs',*preparation]
            add(f'train_{method}_{seed}',cmd,[out/'summary.json'],True,out/'progress.json')
            conditions=[('test32',test,samples,False),('greedy',test,1,True)]
            if hard:conditions.append(('hard128',hard,spec.get('evaluation',{}).get('hard_samples',128),False))
            for name,source,count,greedy in conditions:
                destination=out/(name+'.jsonl')
                cmd=['evaluate','--config',cfg_path,'--data',source,'--checkpoint',out/'evaluation_student.pt',
                    '--seed',seed,'--method',method,'--samples',count,'--output',destination,*initial,*dep]
                if greedy:cmd+=['--greedy']
                add(f'{name}_{method}_{seed}',cmd,[destination.with_suffix('.summary.json')],True,destination.with_suffix('.state')/'protocol.json')
    for method in methods:
        for name in ('test32','greedy',*(['hard128'] if hard else [])):
            destination=root/'reports'/f'{method}_{name}.json'
            cmd=['aggregate','--inputs',*[root/'benchmark'/method/f'seed_{seed}'/(name+'.summary.json') for seed in seeds],'--output',destination]
            if 'direct' in methods and method!='direct':cmd+=['--baseline',*[root/'benchmark/direct'/f'seed_{seed}'/(name+'.summary.json') for seed in seeds]]
            add('aggregate_'+method+'_'+name,cmd,[destination])
    if 'target' in methods:
        destination=root/'reports/efficiency.json'
        add('efficiency',['efficiency','--runs',*[root/'benchmark'/m/f'seed_{s}' for m in methods for s in seeds],
            '--reference-method','target','--output',destination],[destination])
    fresh=spec.get('fresh_student',{})
    if fresh.get('enabled',False):
        fresh_seeds=fresh.get('seeds',seeds)
        for method in fresh.get('methods',['reprsi','target']):
            if method not in methods:raise ValueError('Fresh teacher method must be in benchmark methods')
            directories=[]
            for teacher_seed in seeds:
                frozen=root/'frozen'/f'{method}_{teacher_seed}.jsonl'
                checkpoint=root/'benchmark'/method/f'seed_{teacher_seed}'/'evaluation_teacher.pt'
                if method in ('direct','human'):
                    cmd=['export-reference','--method',method,'--target-train',train,'--config',cfg_path,
                        '--count',fresh.get('curricula',32)*cfg['items'],'--seed',teacher_seed,'--output',frozen,*dep]
                    add(f'freeze_{method}_{teacher_seed}',cmd,[frozen])
                else:
                    cmd=['export-curricula','--config',cfg_path,'--checkpoint',checkpoint,'--count',fresh.get('curricula',32),
                        '--seed',teacher_seed,'--output',frozen,*initial,*dep]
                    add(f'freeze_{method}_{teacher_seed}',cmd,[frozen],True,frozen.with_suffix('.state')/'protocol.json')
                for student_seed in fresh_seeds:
                    out=root/'fresh'/method/f'teacher_{teacher_seed}'/f'student_{student_seed}';directories.append(out)
                    add(f'fresh_{method}_{teacher_seed}_{student_seed}',['fresh-student','--config',cfg_path,
                        '--curricula',frozen,'--steps',fresh.get('steps',800),'--eval-every',fresh.get('eval_every',40),
                        '--eval-data',test,'--seed',student_seed,'--teacher-id',f'{method}_{teacher_seed}',
                        '--output',out,*initial,*dep],[out/'summary.json'],True,out/'progress.json')
            destination=root/'reports'/f'fresh_{method}.json'
            add('fresh_report_'+method,['summarize-fresh','--runs',*directories,'--output',destination],[destination])
    prediction=spec.get('prediction',{})
    if prediction.get('enabled',False):
        pred=root/'prediction';orig=pred/'origins';groups=pred/'groups';trials=pred/'trials'
        add('prediction_origins',['prediction-origins','--config',cfg_path,'--target-train',train,
            '--runs',prediction.get('runs',200),'--steps',prediction.get('origin_steps',400),'--output',orig,*initial,*dep],
            [orig/'origins.jsonl'],True,orig/'origin_initial.pt')
        add('prediction_groups',['prediction-groups','--config',cfg_path,'--origins',orig/'origins.jsonl',
            '--groups-per-run',prediction.get('groups_per_run',10),'--items',prediction.get('items',100),
            '--teacher-output-limit',prediction.get('teacher_output_limit',16384),
            '--max-attempts',prediction.get('max_attempts',20),
            '--output',groups,*dep],[groups/'groups.jsonl'],True,groups/'protocol.json')
        add('prediction_trials',['prediction-trials','--config',cfg_path,'--groups',groups/'groups.jsonl',
            '--diagnostics',diagnostics/'reward.jsonl','--target-train',train,'--eval-data',test,
            '--calibration',calibration,'--trials',prediction.get('trials',5),'--curriculum-steps',prediction.get('curriculum_steps',150),
            '--continuation-steps',prediction.get('continuation_steps',400),'--output',trials,*initial,*dep],
            [trials/'summary.json'],True,trials/'protocol.json')
        for likelihood in ((False,True) if prediction.get('analyze',True) else ()):
            destination=root/'reports'/('prediction_likelihood.json' if likelihood else 'prediction.json')
            add('prediction_analysis_'+str(likelihood),['analyze-prediction','--data',trials/'trials.jsonl','--output',destination,
                *(['--likelihood'] if likelihood else [])],[destination])
    mechanism=spec.get('interventions',{})
    if mechanism.get('enabled',False):
        if domain!='twohop':raise ValueError('Exposure interventions require the twohop domain')
        destination=root/'interventions'
        add('interventions',['intervene','--config',cfg_path,'--calibration',calibration,'--seeds',*seeds,
            '--steps',mechanism.get('steps',500),'--continuation-steps',mechanism.get('continuation_steps',400),
            '--output',destination,*initial,*dep],[destination/'results.json'],True,destination)
    return workspace,root,stages,cfg


def smoke_spec(spec):
    spec=deepcopy(spec);spec['output']=str(Path(spec['output'])/'smoke');spec['smoke']=True
    spec['seeds']=[0];spec['methods']=['reprsi'];spec['rounds']=1;spec['data_limit']=8;spec['atomic_steps']=2
    spec['config_overrides']={**spec.get('config_overrides',{}),'rounds':1,'candidates':2,'items':2,
        'replicates':1,'student_steps':4,'prompts_per_step':1,'completions':2,'specs_per_family':1,
        'eval_every':1,'monitor_every':1}
    spec['diagnostics']={**spec.get('diagnostics',{}),'specs_per_family':1}
    spec['calibration']={'batches':1,'specs_per_family':1,'layers':[0]}
    spec['evaluation']={'samples':2,'hard_samples':2}
    spec['fresh_student']={'enabled':True,'methods':['reprsi'],'curricula':1,'steps':2,'eval_every':1,'seeds':[0]}
    spec['prediction']={'enabled':False};spec['interventions']={'enabled':False}
    return spec


def workload(spec,cfg):
    prediction=spec.get('prediction',{});branches=0;origins=0
    if prediction.get('enabled',False):
        branches=prediction.get('runs',200)*prediction.get('groups_per_run',10)*4*prediction.get('trials',5)
        origins=prediction.get('runs',200)*prediction.get('origin_steps',400)
    return {'prediction_trials':branches,'prediction_optimizer_steps':branches*(prediction.get('curriculum_steps',150)+prediction.get('continuation_steps',400)),
            'prediction_origin_optimizer_steps':origins,'recursive_candidate_optimizer_steps':len(spec.get('seeds',range(5)))*
            spec.get('rounds',cfg['rounds'])*cfg['candidates']*cfg['replicates']*cfg['student_steps'],
            'smoke':spec.get('smoke',False)}


def execute(spec_path, resume=False, plan=False, profile='full'):
    spec_path=Path(spec_path).resolve();spec=json.loads(spec_path.read_text())
    if profile not in ('full','smoke'):raise ValueError('Unknown experiment profile')
    if profile=='smoke':spec=smoke_spec(spec)
    workspace,root,stages,cfg=build(spec,spec_path)
    if plan:return {'workspace':str(workspace),'output':str(root),'stages':stages,'model':cfg['model'],'backend':cfg.get('backend','hf'),'workload':workload(spec,cfg)}
    signature=digest({'spec':spec,'config':cfg,'stages':stages})
    with exclusive(root):
        manifest=root/'workflow.json'
        if manifest.exists():
            state=json.loads(manifest.read_text())
            if not resume or state['signature']!=signature:raise ValueError('Workflow already exists or configuration changed')
        else:
            if resume:raise ValueError('No workflow exists to resume')
            state={'signature':signature,'completed':[]};atomic_json(manifest,state)
        if spec.get('config_overrides'):atomic_json(root/'model.json',cfg)
        for stage in stages:
            outputs=[Path(p) for p in stage['outputs']]
            if stage['name'] in state['completed']:
                if not all(p.exists() for p in outputs):raise ValueError('Completed stage output is missing: '+stage['name'])
                continue
            argv=[sys.executable,'-m','reprsi',*stage['argv']]
            if resume and stage['resumable'] and Path(stage['marker']).exists():argv+=['--resume']
            print('stage='+stage['name'],flush=True)
            subprocess.run(argv,cwd=workspace,check=True)
            if not all(p.exists() for p in outputs):raise RuntimeError('Stage did not produce its outputs: '+stage['name'])
            state['completed'].append(stage['name']);atomic_json(manifest,state)
    return {'completed_stages':len(state['completed']),'output':str(root)}
