from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import random
import numpy as np
import torch
from .metrics import rloo
from .storage import digest


@dataclass
class Rollout:
    prompt_ids: list
    completion_ids: list
    text: str
    schema: dict | None = None
    tiny_allowed: list | None = None
    truncated: bool = False


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact RNG restoration requires the original CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def sequence_terms(logits, ref_logits, targets, allowed=None):
    logprob=logits.new_zeros((),dtype=torch.float32)
    kl=logprob.clone()
    for t, token in enumerate(targets):
        valid=None if allowed is None else allowed[t]
        if valid is None:
            p=logits[t].float().log_softmax(-1)
            q=ref_logits[t].float().log_softmax(-1)
            lp=p[token]
        else:
            if token not in valid: raise ValueError("Sample is outside replayed grammar mask")
            ids=torch.as_tensor(valid,device=logits.device)
            p=logits[t,ids].float().log_softmax(-1)
            q=ref_logits[t,ids].float().log_softmax(-1)
            lp=p[valid.index(token)]
        logprob=logprob+lp
        kl=kl+(p.exp()*(p-q)).sum()
    return logprob,kl


class Policy:
    def __init__(self, config, model=None, tokenizer=None, tiny=False):
        self.config=config;self.device=torch.device(config.get("device","cuda"))
        self.tiny=tiny;self.rollout_tokens=0
        if model is None:
            import transformers as tr
            kw={"revision":config["revision"],"trust_remote_code":False}
            tokenizer=tr.AutoTokenizer.from_pretrained(config["model"],**kw)
            cfg=tr.AutoConfig.from_pretrained(config["model"],**kw)
            cls=tr.Gemma4ForCausalLM if cfg.model_type=="gemma4" else tr.AutoModelForCausalLM
            model=cls.from_pretrained(config["model"],dtype=torch.bfloat16,**kw)
        self.tokenizer=tokenizer
        self.model=model.to(self.device)
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module,torch.nn.Dropout): module.p=0.0
        self.reference_device=torch.device(config.get("reference_device",str(self.device)))
        self.reference=deepcopy(self.model).to(self.reference_device).eval().requires_grad_(False)
        self.checkpoint_signature = digest({"model": config["model"], "revision": config["revision"],
            "parameters": [(n, list(p.shape), str(p.dtype)) for n,p in self.model.named_parameters()],
            "chat_template": getattr(tokenizer, "chat_template", None)})
        self.reference_tag = "pretrained"
        self.gradient_checkpointing = bool(config.get("gradient_checkpointing", False))
        if self.gradient_checkpointing:
            if not hasattr(self.model, "gradient_checkpointing_enable"):
                raise ValueError("This model does not support gradient checkpointing")
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if config.get("deterministic", False):
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        devices={p.device.index for p in [next(self.model.parameters()),next(self.reference.parameters())] if p.device.type=="cuda"}
        self.gpu_count=len(devices)
        self.optimizer=None
        self.grammar_data=None
        self.blocks=self._blocks()
        self.reset_optimizer(config["student_lr"])

    def _blocks(self):
        for path in ("model.layers","model.language_model.layers","transformer.h","layers"):
            obj=self.model
            try:
                for component in path.split("."): obj=getattr(obj,component)
                return obj
            except AttributeError: pass
        raise ValueError("Cannot locate decoder blocks; set up a model-specific block accessor")

    def reset_optimizer(self, lr):
        self.model.zero_grad(set_to_none=True)
        self.optimizer=None
        self.master_parameters=None
        self.parameters=[p for p in self.model.parameters() if p.requires_grad]
        self.has_master=any(p.dtype in (torch.bfloat16,torch.float16) for p in self.parameters)
        self.master_parameters=([torch.nn.Parameter(p.detach().float().clone()) for p in self.parameters]
            if self.has_master else self.parameters)
        self.optimizer=torch.optim.AdamW(self.master_parameters,lr=lr,betas=(0.9,0.95),eps=1e-8,weight_decay=0.0)

    def chat(self, prompt):
        return self.tokenizer.apply_chat_template([
            {"role":"user","content":prompt}], tokenize=False,
            add_generation_prompt=True,enable_thinking=False)

    def encode(self, prompt, limit=None):
        ids=self.tokenizer.encode(self.chat(prompt),add_special_tokens=False)
        limit=limit or self.config["input_limit"]
        if not ids or len(ids)>limit: raise ValueError("Input exceeds fixed token limit; truncation is disabled")
        return ids

    def _grammar(self, schema):
        from lmformatenforcer import JsonSchemaParser,TokenEnforcer,TokenEnforcerTokenizerData
        if self.grammar_data is None:
            tok=self.tokenizer;special=set(tok.all_special_ids)
            sentinel=tok.encode("0",add_special_tokens=False)[-1]
            prefix=tok.decode([sentinel],clean_up_tokenization_spaces=False)
            vocabulary=[]
            for token in range(len(tok)):
                if token in special:continue
                isolated=tok.decode([token],clean_up_tokenization_spaces=False)
                contextual=tok.decode([sentinel,token],clean_up_tokenization_spaces=False)[len(prefix):]
                vocabulary.append((token,contextual,len(contextual)>len(isolated)))
            eos=self.model.generation_config.eos_token_id or tok.eos_token_id
            self.grammar_data=TokenEnforcerTokenizerData(vocabulary,
                lambda ids:tok.decode(ids,clean_up_tokenization_spaces=False).rstrip("\ufffd"),eos,False,len(tok))
        enforcer=TokenEnforcer(self.grammar_data,JsonSchemaParser(schema))
        return lambda batch_id,ids:enforcer.get_allowed_tokens(ids.tolist()).allowed_tokens

    def sample(self,prompt,schema=None,greedy=False,input_limit=None):
        self.model.eval()
        ids=self.encode(prompt,input_limit)
        if self.tiny:
            return self._tiny_sample(ids,schema,greedy)
        from transformers import GenerationConfig
        eos=self.model.generation_config.eos_token_id
        if eos is None: eos=self.tokenizer.eos_token_id
        eos_ids=eos if isinstance(eos,list) else [eos]
        options=GenerationConfig(do_sample=not greedy,temperature=1.0,top_p=1.0,top_k=0,
            max_new_tokens=self.config["output_limit"],eos_token_id=eos,
            pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_ids[0]),repetition_penalty=1.0)
        constraint=self._grammar(schema) if schema else None
        with torch.no_grad():
            seq=self.model.generate(input_ids=torch.tensor([ids],device=self.device),
                attention_mask=torch.ones((1,len(ids)),device=self.device,dtype=torch.long),
                generation_config=options,prefix_allowed_tokens_fn=constraint)
        completion=seq[0,len(ids):].tolist()
        self.rollout_tokens+=len(completion)
        return Rollout(ids,completion,self.tokenizer.decode(completion,skip_special_tokens=True),
            schema=schema,truncated=not completion or completion[-1] not in eos_ids)

    def _allowed(self,rollout):
        if rollout.tiny_allowed is not None:
            return [rollout.tiny_allowed for _ in rollout.completion_ids]
        if rollout.schema is None: return None
        fn=self._grammar(rollout.schema)
        prefix=list(rollout.prompt_ids)
        masks=[]
        for token in rollout.completion_ids:
            valid=list(fn(0,torch.tensor(prefix,device=self.device)))
            masks.append(valid);prefix.append(token)
        return masks

    def update(self,rollouts,advantages,lr):
        if len(rollouts)!=len(advantages) or not rollouts: raise ValueError("Invalid update group")
        for group in self.optimizer.param_groups: group["lr"]=lr
        self.model.train(self.gradient_checkpointing)
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss_value=0.0
        for rollout, advantage in zip(rollouts,advantages):
            full=rollout.prompt_ids+rollout.completion_ids
            if not rollout.completion_ids: raise ValueError("Empty rollout")
            x=torch.tensor([full[:-1]],device=self.device)
            start=len(rollout.prompt_ids)-1
            mask=self._allowed(rollout)
            with torch.no_grad(): ref=self.reference(x.to(self.reference_device),use_cache=False).logits[0,start:].to(self.device)
            logits=self.model(x,use_cache=False).logits[0,start:]
            lp,kl=sequence_terms(logits,ref,rollout.completion_ids,mask)
            loss=(-float(advantage)*lp+self.config["kl_coef"]*kl)/len(rollouts)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite policy loss")
            loss.backward();loss_value+=float(loss.detach())
            del logits,ref,loss,lp,kl
        norm = self._optimizer_step()
        return {"loss":loss_value,"grad_norm":norm}

    def _optimizer_step(self):
        if self.has_master:
            for parameter,master in zip(self.parameters,self.master_parameters):
                master.grad=None if parameter.grad is None else parameter.grad.detach().float()
        norm=torch.nn.utils.clip_grad_norm_(self.master_parameters,1.0)
        if not torch.isfinite(norm): raise FloatingPointError("Non-finite policy gradient")
        self.optimizer.step()
        if self.has_master:
            with torch.no_grad():
                for parameter,master in zip(self.parameters,self.master_parameters):parameter.copy_(master)
        self.model.eval()
        return float(norm)

    def train_batch(self, tasks):
        if not tasks: raise ValueError("Empty training batch")
        rollouts, advantages, rewards_all = [], [], []
        for task in tasks:
            samples = [self.sample(task.prompt) for _ in range(self.config["completions"])]
            rewards = [task.verify(s.text) for s in samples]
            if any(x not in (0, 1) for x in rewards): raise ValueError("Verifier must return a binary reward")
            rollouts.extend(samples); advantages.extend(rloo(rewards)); rewards_all.extend(rewards)
        stats = self.update(rollouts, advantages, self.config["student_lr"])
        return {**stats, "reward": float(np.mean(rewards_all)), "answers": len(rewards_all)}

    def train_steps(self, tasks, steps, seed, start=0):
        if not tasks or steps < 1: raise ValueError("Training needs examples and positive steps")
        values=[];n=self.config["prompts_per_step"]
        from .metrics import seed_for
        for step in range(start, start+steps):
            seed_all(seed_for(seed, "student_step", step))
            batch=[tasks[(step*n+i)%len(tasks)] for i in range(n)]
            values.append(self.train_batch(batch)["reward"])
        return float(np.mean(values))

    def train_curriculum(self,tasks,seed):
        seed_all(seed)
        n=self.config["prompts_per_step"]; steps=self.config["student_steps"]
        if steps*n!=2*len(tasks): raise ValueError("Each ordered curriculum must be traversed twice")
        return self.train_steps(tasks, steps, seed)

    def supervised_batch(self, tasks):
        if not tasks: raise ValueError("Empty supervised batch")
        self.model.train(self.gradient_checkpointing)
        self.model.zero_grad(set_to_none=True); self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups: group["lr"] = self.config["student_lr"]
        loss_value=0.0
        for task in tasks:
            ids=self.encode(task.prompt)
            target=self.tokenizer.encode(task.answer,add_special_tokens=False)
            if not target: raise ValueError("Empty atomic answer")
            x=torch.tensor([ids+target[:-1]],device=self.device)
            logits=self.model(x,use_cache=False).logits[0,len(ids)-1:].float()
            loss=torch.nn.functional.cross_entropy(logits,torch.tensor(target,device=self.device))/len(tasks)
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite supervised loss")
            loss.backward();loss_value+=float(loss.detach())
        return {"loss":loss_value,"grad_norm":self._optimizer_step()}

    def anchor_input(self,record):
        text=self.chat(record["prompt"])
        start=text.index(record["prompt"])
        end=start+len(record["prefix"])
        encoded=self.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded["input_ids"]
        offsets=encoded["offset_mapping"]
        positions=[i for i,(a,b) in enumerate(offsets) if a<end and b>start and b>a]
        if not positions: raise ValueError("No diagnostic anchor token")
        pos=positions[-1]
        if offsets[pos][1]>end: raise ValueError("Tokenizer crosses the diagnostic boundary")
        if len(ids)>self.config["input_limit"]: raise ValueError("Diagnostic prompt exceeds input limit")
        return ids,pos

    def hidden(self,records,layer):
        if not 0<=layer<len(self.blocks): raise ValueError("Readout is a zero-based decoder block index")
        self.model.eval();vectors=[]
        with torch.no_grad():
            for record in records:
                ids,pos=self.anchor_input(record)
                def capture(module,args,result):
                    h=result[0] if isinstance(result,tuple) else result
                    vectors.append(h[0,pos].detach().float().cpu().numpy().copy())
                handle=self.blocks[layer].register_forward_hook(capture)
                try: self.model(torch.tensor([ids],device=self.device),use_cache=False)
                finally: handle.remove()
        return np.stack(vectors)

    def answer_logprob(self,record,answer,layer=None,patch=None):
        ids,pos=self.anchor_input(record)
        rendered=answer if record.get("answer_format")=="entity" else "\\boxed{"+answer+"}"
        target=self.tokenizer.encode(rendered,add_special_tokens=False)
        if not target: raise ValueError("Empty reference answer")
        handle=None
        if patch is not None:
            def replace(module,args,result):
                h=result[0] if isinstance(result,tuple) else result
                h=h.clone();h[0,pos]=torch.as_tensor(patch,device=h.device,dtype=h.dtype)
                return (h,)+result[1:] if isinstance(result,tuple) else h
            handle=self.blocks[layer].register_forward_hook(replace)
        try:
            with torch.no_grad():
                x=torch.tensor([ids+target[:-1]],device=self.device)
                logits=self.model(x,use_cache=False).logits[0,len(ids)-1:].float()
                lp=logits.log_softmax(-1).gather(-1,torch.tensor(target,device=self.device)[:,None]).sum()
                return float(lp)
        finally:
            if handle: handle.remove()

    def answer_probability(self,record,answer,layer=None,patch=None):
        return float(np.exp(self.answer_logprob(record,answer,layer,patch)))

    def patched_sample(self, record, layer, patch):
        _,pos=self.anchor_input(record)
        def replace(module,args,result):
            h=result[0] if isinstance(result,tuple) else result
            if h.shape[1] <= pos: return result  
            h=h.clone();h[0,pos]=torch.as_tensor(patch,device=h.device,dtype=h.dtype)
            return (h,)+result[1:] if isinstance(result,tuple) else h
        handle=self.blocks[layer].register_forward_hook(replace)
        try: return self.sample(record["prompt"],greedy=True)
        finally: handle.remove()

    def set_reference(self, tag):
        self.reference.load_state_dict(self.model.state_dict())
        self.reference_tag=tag

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        def cpu(x):
            if torch.is_tensor(x): return x.detach().cpu()
            if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
            if isinstance(x,list): return [cpu(v) for v in x]
            if isinstance(x,tuple): return tuple(cpu(v) for v in x)
            return x
        temporary=path.with_suffix(".tmp")
        signature=digest({"model":self.config["model"],"revision":self.config["revision"],
            "parameters":[(n,list(p.shape),str(p.dtype)) for n,p in self.model.named_parameters()],
            "chat_template":getattr(self.tokenizer,"chat_template",None)})
        torch.save({"format_version":2,"signature":signature,"reference_tag":self.reference_tag,
            "model":cpu(self.model.state_dict()),"optimizer":cpu(self.optimizer.state_dict()),
            "rng":rng_state(),"master_parameters":cpu(self.master_parameters) if self.has_master else None},temporary)
        temporary.replace(path)

    def load(self,path,restore_random=False,allow_legacy=False):
        state=torch.load(path,map_location="cpu",weights_only=True)
        signature=digest({"model":self.config["model"],"revision":self.config["revision"],
            "parameters":[(n,list(p.shape),str(p.dtype)) for n,p in self.model.named_parameters()],
            "chat_template":getattr(self.tokenizer,"chat_template",None)})
        if state.get("format_version")!=2:
            if not allow_legacy: raise ValueError("Legacy checkpoint: explicit migration is required")
        elif state["signature"]!=signature:
            raise ValueError("Checkpoint backbone, dtype, tokenizer template or shape mismatch")
        elif state.get("reference_tag")!=self.reference_tag:
            raise ValueError("Checkpoint KL reference differs; load the common initialization first")
        self.model.load_state_dict(state["model"])
        self.reset_optimizer(self.config["student_lr"])
        if self.has_master:
            if state.get("master_parameters") is None:raise ValueError("BF16 checkpoint is missing FP32 master weights")
            if len(state["master_parameters"])!=len(self.master_parameters):raise ValueError("Master-weight count mismatch")
            with torch.no_grad():
                for master,saved in zip(self.master_parameters,state["master_parameters"]):
                    if master.shape!=saved.shape:raise ValueError("Master-weight shape mismatch")
                    master.copy_(saved)
        self.optimizer.load_state_dict(state["optimizer"])
        if restore_random: restore_rng(state["rng"])
        del state

    def synchronize(self):
        for device in {self.device,self.reference_device}:
            if device.type=="cuda": torch.cuda.synchronize(device)
