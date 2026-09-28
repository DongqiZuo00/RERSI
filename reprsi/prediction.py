from pathlib import Path
import json
import numpy as np
from .storage import digest, file_digest, exclusive, atomic_json, journal, append_json
from .diagnostics import write_jsonl
from .metrics import seed_for, cohesion
from .policy import seed_all
from .evaluation import greedy_score, task_fingerprint
from .experiments import train_fixed
from .loop import Ledger
from .schema import teacher_prompt, curriculum_schema
from .tasks import build_curriculum, Task


def initialize(policy, initial):
    if initial:
        policy.reference_tag='pretrained'
        policy.load(initial)
        tag=file_digest(initial)
        policy.set_reference(tag)
        return tag
    return 'pretrained'


def origins(policy, config, targets, initial, output, runs=200, steps=400, resume=False):
    if not targets or runs<1 or steps<1:raise ValueError('Positive runs, steps, and target training data are required')
    output=Path(output);rows=[]
    initial_tag=initialize(policy,initial)
    seed_checkpoint=output/'origin_initial.pt'
    output.mkdir(parents=True,exist_ok=True)
    if not seed_checkpoint.exists():policy.save(seed_checkpoint)
    for run_id in range(runs):
        if initial:initialize(policy,initial)
        else:policy.load(seed_checkpoint)
        rng=np.random.default_rng(seed_for(2027,'origin_order',run_id))
        tasks=[targets[int(i)] for i in rng.permutation(len(targets))]
        directory=output/f'run_{run_id:03d}'
        train_fixed(policy,{**config,'seed':run_id},tasks,directory,steps,steps,
            resume=resume and (directory/'progress.json').exists(),initial_id=initial_tag)
        rows.append({'run':run_id,'origin':f'run_{run_id:03d}/student.pt','origin_steps':steps})
        write_jsonl(output/'origins.jsonl',rows)
    return {'runs':len(rows),'steps':steps}


def generate_groups(policy, config, source_origins, output, domain=None, groups_per_run=10, items=100, resume=False):
    if not source_origins or groups_per_run<1 or items<1:raise ValueError('Positive group count and curriculum size required')
    if len({r['run'] for r in source_origins})!=len(source_origins):raise ValueError('Duplicate origin run')
    directory=Path(output)
    protocol=digest({'origins':[{**r,'origin':file_digest(r['origin'])} for r in source_origins],
        'groups':groups_per_run,'items':items,'config':config})
    builder=domain.build_curriculum if domain else build_curriculum
    prompt=domain.teacher_prompt(items) if domain else teacher_prompt(items)
    schema=domain.curriculum_schema(items) if domain else curriculum_schema(items)
    with exclusive(directory):
        path=directory/'protocol.json'
        if path.exists():
            if not resume or json.loads(path.read_text())['signature']!=protocol:raise ValueError('Prediction generation protocol changed')
        elif resume:raise ValueError('No prediction generation to resume')
        else:atomic_json(path,{'signature':protocol})
        saved=journal(directory/'groups.jsonl',repair=True);seen={(r['run'],r['group']) for r in saved}
        ledger=Ledger(policy,directory/'compute.jsonl')
        for origin in source_origins:
            for group in range(groups_per_run):
                if (origin['run'],group) in seen:continue
                curricula=[]
                with ledger.charge('fixed_curriculum_generation'):
                    for k in range(4):
                        seed_all(seed_for(2027,'prediction_generation',origin['run'],group,k),policy)
                        proposal=policy.sample(prompt,schema=schema)
                        if proposal.truncated:raise ValueError('Curriculum truncated; increase teacher_output_limit')
                        tasks=builder(proposal.text,items)
                        for task in tasks:policy.encode(task.prompt)
                        curricula.append(json.loads(proposal.text)['items'])
                append_json(directory/'groups.jsonl',{**origin,'group':group,'curricula':curricula})
        return {'groups':len(journal(directory/'groups.jsonl')),'gpu_seconds':ledger.total}


def collect(policy, config, diagnostics, targets, evaluation, groups, initial, output, layer,
            domain=None, trials=5, curriculum_steps=150, continuation_steps=400, resume=False):
    if not diagnostics or not targets or not evaluation:raise ValueError('Diagnostics, target TRAIN data, and evaluation data are required')
    if any(r['split']!='reward' for r in diagnostics):raise ValueError('Prediction features require reward diagnostics')
    if trials<1 or curriculum_steps<1 or continuation_steps<1:raise ValueError('Positive trial schedule required')
    initialize(policy,initial)
    entity=diagnostics[0].get('answer_format')=='entity'
    local_records=[{**r,'prompt':r.get('local_prompt',r['prefix']+' Return the exact answer in \\boxed{...}.'),
        'prefix':r.get('local_prompt',r['prefix'])} for r in diagnostics]
    if entity:
        from .mechanism import EntityTask
        local=[EntityTask(r['id'],r['prompt'],r['reference_state']) for r in local_records]
    else:
        local=[Task(r['prompt'],r['reference_state'],r['family'],r['id']) for r in local_records]
    normalization=config.get('likelihood_normalization','mean')
    if normalization not in ('mean','sum'):raise ValueError('likelihood_normalization must be mean or sum')
    normalized=normalization=='mean'
    def likelihood(records,field):
        return float(np.mean([policy.answer_logprob(r,r[field],normalize=normalized) for r in records]))
    if not groups or len({(g['run'],g['group']) for g in groups})!=len(groups):raise ValueError('Empty or duplicate prediction groups')
    builder=domain.build_curriculum if domain else build_curriculum
    origin_hashes={p:file_digest(p) for p in {g['origin'] for g in groups}}
    directory=Path(output)
    protocol=digest({'groups':[{**g,'origin':origin_hashes[g['origin']]} for g in groups],
        'initial':file_digest(initial) if initial else 'pretrained','layer':layer,'trials':trials,
        'curriculum_steps':curriculum_steps,'continuation_steps':continuation_steps,
        'diagnostics':digest(diagnostics),'targets':task_fingerprint(targets),'evaluation':task_fingerprint(evaluation),'config':config})
    with exclusive(directory):
        meta=directory/'protocol.json'
        if meta.exists():
            if not resume or json.loads(meta.read_text())['signature']!=protocol:raise ValueError('Prediction trial protocol changed')
        else:
            if resume:raise ValueError('No prediction trials to resume')
            atomic_json(meta,{'signature':protocol,'likelihood_normalization':normalization})
        ledger=Ledger(policy,directory/'compute.jsonl');saved=journal(directory/'trials.jsonl',repair=True)
        completed={(r['run'],r['group'],r['candidate'],r['trial']) for r in saved}
        for group in groups:
            if len(group['curricula'])!=4:raise ValueError('Prediction group must contain four curricula')
            if all((group['run'],group['group'],k,a) in completed for k in range(4) for a in range(trials)):continue
            with ledger.charge('prediction_origin_measurement'):
                policy.load(group['origin'])
                before=cohesion(policy.hidden(diagnostics,layer),diagnostics)['phi']
                before_answer=likelihood(diagnostics,'answer');before_state=likelihood(local_records,'reference_state')
            for k,items in enumerate(group['curricula']):
                tasks=builder(json.dumps({'items':items}),len(items));curriculum_digest=digest(items)
                for trial in range(trials):
                    key=(group['run'],group['group'],k,trial)
                    if key in completed:continue
                    seed=seed_for(2027,'prediction_trial',group['run'],group['group'],trial)
                    with ledger.charge('prediction_curriculum_trial'):
                        policy.load(group['origin']);policy.train_steps(tasks,curriculum_steps,seed)
                    with ledger.charge('pre_continuation_features'):
                        phi=cohesion(policy.hidden(diagnostics,layer),diagnostics)['phi']
                        current=100*greedy_score(policy,evaluation)['accuracy']
                        curriculum=100*greedy_score(policy,tasks)['accuracy'];state=100*greedy_score(policy,local)['accuracy']
                        answer_ll=likelihood(diagnostics,'answer');state_ll=likelihood(local_records,'reference_state')
                    policy.reset_optimizer(config['student_lr'])
                    rng=np.random.default_rng(seed_for(seed,'target_order'))
                    continuation=[targets[int(i)] for i in rng.permutation(len(targets))]
                    with ledger.charge('prediction_target_continuation'):
                        policy.train_steps(continuation,continuation_steps,seed_for(seed,'target_training'))
                    with ledger.charge('post_continuation_measurement'):future=100*greedy_score(policy,evaluation)['accuracy']
                    append_json(directory/'trials.jsonl',{'run':key[0],'group':key[1],'candidate':k,'trial':trial,
                        'curriculum_digest':curriculum_digest,'current_accuracy':current,'curriculum_accuracy':curriculum,
                        'state_accuracy':state,'delta_phi':phi-before,'answer_logprob_change':answer_ll-before_answer,
                        'state_logprob_change':state_ll-before_state,'future_accuracy':future,
                        'likelihood_normalization':normalization})
                    completed.add(key)
        return {'trials':len(journal(directory/'trials.jsonl')),'gpu_seconds':ledger.total,'wall_seconds':ledger.wall}
