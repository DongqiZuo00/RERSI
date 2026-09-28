from importlib import import_module
from pathlib import Path
from typing import Protocol, Any, runtime_checkable
import json
import os


@runtime_checkable
class TrainingBackend(Protocol):
    config: dict
    device: Any
    reference_device: Any
    gpu_count: int
    rollout_tokens: int
    reference_tag: str
    blocks: Any

    def sample(self, prompt, schema=None, greedy=False, input_limit=None, output_limit=None): ...
    def encode(self, prompt, limit=None): ...
    def update(self, rollouts, advantages, lr): ...
    def train_batch(self, tasks): ...
    def train_steps(self, tasks, steps, seed, start=0): ...
    def train_curriculum(self, tasks, seed): ...
    def supervised_batch(self, tasks): ...
    def hidden(self, records, layer): ...
    def answer_logprob(self, record, answer, layer=None, patch=None, normalize=False): ...
    def answer_probability(self, record, answer, layer=None, patch=None): ...
    def patched_sample(self, record, layer, patch): ...
    def reset_optimizer(self, lr): ...
    def set_reference(self, tag): ...
    def set_seed(self, seed): ...
    def save(self, path): ...
    def load(self, path, restore_random=False, allow_legacy=False): ...
    def synchronize(self): ...
    def hardware_signature(self): ...


@runtime_checkable
class TaskInterface(Protocol):
    identity: str
    prompt: str

    def verify(self, response): ...


@runtime_checkable
class SupervisedTaskInterface(TaskInterface, Protocol):
    answer: str


@runtime_checkable
class DomainInterface(Protocol):
    def teacher_prompt(self, count): ...
    def curriculum_schema(self, count): ...
    def build_curriculum(self, text, count): ...
    def fixed_calibration_curriculum(self, count, seed): ...


@runtime_checkable
class StagedDomainInterface(DomainInterface, Protocol):
    def human_curriculum(self, count, seed, stage): ...


def resolve_factory(value):
    module, separator, name = value.partition(':')
    if not separator or not module or not name:
        raise ValueError('Factory must use module:callable syntax')
    obj = import_module(module)
    for part in name.split('.'):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError('Configured factory is not callable')
    return obj


def load_config(path):
    path = Path(path).resolve()
    def read(current, parents):
        if current in parents:
            raise ValueError('Configuration inheritance cycle')
        cfg = json.loads(current.read_text())
        base = cfg.pop('extends', None)
        if base:
            merged = read((current.parent / base).resolve(), parents | {current})
            merged.update(cfg)
            cfg = merged
        return cfg
    cfg = read(path, set())
    for key in ('model', 'revision', 'tokenizer', 'tokenizer_revision', 'cache_dir', 'chat_template_path'):
        if isinstance(cfg.get(key), str):
            cfg[key] = os.path.expandvars(os.path.expanduser(cfg[key]))
    cfg.setdefault('backend', 'hf')
    cfg.setdefault('teacher_output_limit', max(4096, cfg.get('output_limit', 4096)))
    cfg.setdefault('likelihood_normalization', 'mean')
    return cfg


def make_policy(cfg):
    from .policy import Policy, seed_all
    if '${' in str(cfg.get('model','')):
        raise ValueError('Set the model environment variable or configure an explicit model path')
    seed_all(cfg['seed'])
    backend = cfg.get('backend', 'hf')
    if backend == 'tiny':
        from .smoke import TinyPolicy
        policy = TinyPolicy(cfg)
    elif backend == 'hf':
        policy = Policy(cfg)
    elif backend == 'custom':
        policy = resolve_factory(cfg['backend_factory'])(cfg)
    else:
        raise ValueError('backend must be hf, tiny, or custom')
    if not isinstance(policy, TrainingBackend):
        missing = [k for k, value in TrainingBackend.__dict__.items()
                   if callable(value) and not k.startswith('_') and not callable(getattr(policy, k, None))]
        missing += [k for k in TrainingBackend.__annotations__ if not hasattr(policy,k)]
        raise TypeError('Training backend is incomplete: ' + ', '.join(missing))
    policy.set_seed(cfg['seed'])
    return policy
