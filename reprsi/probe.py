from pathlib import Path
import json
import time
import numpy as np
from .interfaces import make_policy
from .diagnostics import make_math_pool,sample_batch
from .schema import teacher_prompt, curriculum_schema
from .tasks import build_curriculum
from .metrics import cohesion
from .storage import atomic_json, exclusive


def probe(config, output, domain=None, records=None, layer=0):
    import torch
    begin=time.perf_counter()
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(device)
    policy=make_policy(config)
    if records is None:
        if domain is not None:raise ValueError('Domain probe requires a diagnostic file')
        records=make_math_pool('reward',1)
    if not records:raise ValueError('Probe diagnostics are empty')
    records=sample_batch(records,config.get('specs_per_family',1),2026)
    count=config['items']
    prompt=domain.teacher_prompt(count) if domain else teacher_prompt(count)
    schema=domain.curriculum_schema(count) if domain else curriculum_schema(count)
    builder=domain.build_curriculum if domain else build_curriculum
    with exclusive(output):
        root=Path(output)
        if (root/'probe.json').exists():raise ValueError('Probe already exists')
        policy.save(root/'initial.pt')
        before=policy.hidden(records,layer)
        proposals=[];invalid=[];tasks=None
        for attempt in range(config.get('probe_max_attempts',20)):
            proposal=policy.sample(prompt,schema=schema)
            try:
                if proposal.truncated:raise ValueError('Teacher generation was truncated')
                current=builder(proposal.text,count)
                if len(current)!=count:raise ValueError('Wrong curriculum length')
                for task in current:policy.encode(task.prompt)
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                invalid.append(str(exc));continue
            proposals.append(proposal)
            if tasks is None:tasks=current
            if len(proposals)==2:break
        if len(proposals)<2:raise ValueError('Probe could not generate two valid curricula: '+str(invalid[-1:]))
        student=policy.train_batch(tasks[:config['prompts_per_step']])
        after=policy.hidden(records,layer)
        if not np.isfinite(after).all():raise FloatingPointError('Hidden states are non-finite')
        policy.save(root/'student.pt')
        policy.load(root/'initial.pt')
        restored=policy.hidden(records,layer)
        restore_error=float(np.max(np.abs(restored-before)))
        if restore_error>1e-5:raise RuntimeError('Checkpoint restoration changed hidden states')
        teacher=policy.update(proposals,[1.,-1.],config['teacher_lr'])
        policy.save(root/'teacher.pt');policy.load(root/'student.pt')
        response=policy.sample(tasks[0].prompt,greedy=True)
        report={'model':config['model'],'revision':config['revision'],'backend':config.get('backend','hf'),
            'hardware':policy.hardware_signature(),'hidden_shape':list(after.shape),
            'initial_phi':cohesion(before,records)['phi'],'updated_phi':cohesion(after,records)['phi'],
            'student_update':student,'teacher_update':teacher,'checkpoint_restore_max_error':restore_error,
            'greedy_success':tasks[0].verify(response.text),'generated_tokens':policy.rollout_tokens,
            'teacher_output_limit':config.get('teacher_output_limit',config['output_limit']),
            'student_output_limit':config['output_limit'],'invalid_proposals':invalid}
        policy.synchronize()
        report['wall_seconds']=time.perf_counter()-begin
        report['cuda_memory']=[{'device':i,'name':torch.cuda.get_device_name(i),
            'total_bytes':torch.cuda.get_device_properties(i).total_memory,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(i),'peak_reserved_bytes':torch.cuda.max_memory_reserved(i)}
            for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        atomic_json(root/'probe.json',report)
        return report
