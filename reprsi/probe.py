from pathlib import Path
import json
import numpy as np
from .interfaces import make_policy
from .diagnostics import make_math_pool
from .schema import teacher_prompt, curriculum_schema
from .tasks import build_curriculum
from .metrics import cohesion
from .storage import atomic_json, exclusive


def probe(config, output, domain=None, records=None, layer=0):
    policy=make_policy(config)
    if records is None:
        if domain is not None:raise ValueError('Domain probe requires a diagnostic file')
        records=make_math_pool('reward',1)
    if not records:raise ValueError('Probe diagnostics are empty')
    count=config['items']
    prompt=domain.teacher_prompt(count) if domain else teacher_prompt(count)
    schema=domain.curriculum_schema(count) if domain else curriculum_schema(count)
    builder=domain.build_curriculum if domain else build_curriculum
    with exclusive(output):
        root=Path(output)
        if (root/'probe.json').exists():raise ValueError('Probe already exists')
        policy.save(root/'initial.pt')
        before=policy.hidden(records,layer)
        proposals=[policy.sample(prompt,schema=schema) for _ in range(2)]
        if any(p.truncated for p in proposals):raise ValueError('Teacher generation was truncated; increase teacher_output_limit')
        tasks=builder(proposals[0].text,count)
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
            'student_output_limit':config['output_limit']}
        atomic_json(root/'probe.json',report)
        return report
