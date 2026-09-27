"""Single-device, serial full-parameter RLOO backend with fixed reference policy.

Branches and teacher/student optimizer states are swapped from local checkpoints.
The reference model remains the initialization. No adapters or representation loss.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import random
import numpy as np
import torch
from .metrics import rloo


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


def sequence_terms(logits, ref_logits, targets, allowed=None):
    """Sum log pi(completion) and exact categorical forward KL on sampled prefixes.

    Grammar masks restrict BOTH distributions before softmax. No length averaging
    of curriculum log probability. Masked entries are gathered out to avoid 0*NaN.
    """
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
        self.parameters=list(self.model.parameters())
        self.has_master=any(p.dtype in (torch.bfloat16,torch.float16) for p in self.parameters)
        # BF16 forward weights require FP32 master weights: 1e-6 updates can round
        # to zero if AdamW directly updates BF16 parameters and moment buffers.
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
        # Use the library-neutral API: LMFE 0.11.3's bundled Transformers bridge
        # imports a tokenizer class removed from transformers.tokenization_utils in v5.
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
            pad_token_id=self.tokenizer.pad_token_id or eos_ids[0],repetition_penalty=1.0)
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
        """One on-policy update per group; reward advantages are stop-gradient."""
        if len(rollouts)!=len(advantages) or not rollouts: raise ValueError("Invalid update group")
        for group in self.optimizer.param_groups: group["lr"]=lr
        self.model.eval()  # eval disables dropout, NOT autograd
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
        if self.has_master:
            for parameter,master in zip(self.parameters,self.master_parameters):
                master.grad=None if parameter.grad is None else parameter.grad.detach().float()
        norm=torch.nn.utils.clip_grad_norm_(self.master_parameters,1.0)
        if not torch.isfinite(norm): raise FloatingPointError("Non-finite policy gradient")
        self.optimizer.step()
        if self.has_master:
            with torch.no_grad():
                for parameter,master in zip(self.parameters,self.master_parameters):parameter.copy_(master)
        return {"loss":loss_value,"grad_norm":float(norm)}

    def train_curriculum(self,tasks,seed):
        seed_all(seed)
        n=self.config["prompts_per_step"]; steps=self.config["student_steps"]
        if steps*n!=2*len(tasks): raise ValueError("Each ordered curriculum must be traversed twice")
        total_reward=0.0;answers=0
        for step in range(steps):
            batch=[tasks[(step*n+i)%len(tasks)] for i in range(n)]
            rollouts=[];advantages=[]
            for task in batch:
                samples=[self.sample(task.prompt) for _ in range(self.config["completions"])]
                rewards=[task.verify(s.text) for s in samples]
                total_reward+=sum(rewards);answers+=len(rewards)
                rollouts.extend(samples);advantages.extend(rloo(rewards))
            self.update(rollouts,advantages,self.config["student_lr"])
        return total_reward/answers

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

    def answer_probability(self,record,answer,layer=None,patch=None):
        """Full reference-answer sequence probability for semantic patching."""
        ids,pos=self.anchor_input(record)
        target=self.tokenizer.encode("\\boxed{"+answer+"}",add_special_tokens=False)
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
                return float(lp.exp())
        finally:
            if handle: handle.remove()

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        def cpu(x):
            if torch.is_tensor(x): return x.detach().cpu()
            if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
            if isinstance(x,list): return [cpu(v) for v in x]
            if isinstance(x,tuple): return tuple(cpu(v) for v in x)
            return x
        temporary=path.with_suffix(".tmp")
        torch.save({"model":cpu(self.model.state_dict()),"optimizer":cpu(self.optimizer.state_dict()),
            "master_parameters":cpu(self.master_parameters) if self.has_master else None},temporary)
        temporary.replace(path)

    def load(self,path):
        state=torch.load(path,map_location="cpu",weights_only=True)
        self.model.load_state_dict(state["model"])
        self.reset_optimizer(self.config["student_lr"])
        if self.has_master:
            if state.get("master_parameters") is None:raise ValueError("BF16 checkpoint is missing FP32 master weights")
            with torch.no_grad():
                for master,saved in zip(self.master_parameters,state["master_parameters"]):master.copy_(saved)
        self.optimizer.load_state_dict(state["optimizer"])
        del state

    def synchronize(self):
        for device in {self.device,self.reference_device}:
            if device.type=="cuda": torch.cuda.synchronize(device)
