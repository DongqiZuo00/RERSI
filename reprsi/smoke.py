"""CPU smoke backend: real tiny causal Transformer and RLOO, finite action space.

Only an integration check. Its tokenization, grammar, and learning rates are not
the experimental protocol. No benchmark claim may be inferred from its outputs.
"""
from types import SimpleNamespace
import json
import torch
from .policy import Policy,Rollout,seed_all


class TinyTokenizer:
    def __init__(self,items):
        self.actions=[]
        for offset in range(3):
            curriculum=[]
            for i in range(items):
                curriculum.append({"family":"rational","parameters":{"a":[1+offset,1],"b":[1+i,1]},
                    "composition":[{"id":"n0","op":"add","args":["a","b"]}],"presentation":"symbolic"})
            self.actions.append(json.dumps({"items":curriculum},separators=(",",":")))
        self.answers=["\\boxed{"+str(i)+"}" for i in range(9)]
        self.strings=self.actions+self.answers
        self.vocab_size=259+len(self.strings)
        self.eos_token_id=0;self.pad_token_id=1

    def apply_chat_template(self,messages,**kwargs):
        return "User: "+messages[0]["content"]+"\nAssistant: "

    def encode(self,text,**kwargs):
        if text in self.strings: return [259+self.strings.index(text)]
        return [ord(c)+3 for c in text]  # generated diagnostics and prompts are ASCII

    def __call__(self,text,**kwargs):
        return {"input_ids":self.encode(text),"offset_mapping":[(i,i+1) for i in range(len(text))]}

    def decode(self,ids,**kwargs):
        return "".join(self.strings[i-259] if i>=259 else chr(i-3) for i in ids if i>=3)


class TinyModel(torch.nn.Module):
    def __init__(self,vocab):
        super().__init__();width=16
        self.embed=torch.nn.Embedding(vocab,width)
        self.position=torch.nn.Embedding(8192,width)
        self.layers=torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(width,2,32,dropout=0,batch_first=True,norm_first=True) for _ in range(2)])
        self.head=torch.nn.Linear(width,vocab,bias=False)

    def forward(self,ids,use_cache=False):
        length=ids.shape[1]
        x=self.embed(ids)+self.position(torch.arange(length,device=ids.device))[None]
        mask=torch.ones((length,length),device=ids.device,dtype=torch.bool).triu(1)
        for layer in self.layers: x=layer(x,src_mask=mask)
        return SimpleNamespace(logits=self.head(x))


class TinyPolicy(Policy):
    def __init__(self,config):
        seed_all(config["seed"]);torch.set_num_threads(1)
        # Disable fused inference paths so forward hooks and gradients share one path.
        torch.backends.mha.set_fastpath_enabled(False)
        tok=TinyTokenizer(config["items"])
        super().__init__(config,TinyModel(tok.vocab_size),tok,tiny=True)

    def _tiny_sample(self,ids,schema,greedy):
        allowed=list(range(259,262)) if schema else list(range(262,self.tokenizer.vocab_size))
        with torch.no_grad():
            logits=self.model(torch.tensor([ids],device=self.device)).logits[0,-1,allowed]
            i=int(logits.argmax()) if greedy else int(torch.multinomial(logits.softmax(-1),1))
        token=allowed[i];self.rollout_tokens+=1
        return Rollout(ids,[token],self.tokenizer.decode([token]),schema=schema,tiny_allowed=allowed)


def smoke_config():
    return {"model":"tiny-transformer-integration-check","revision":"local","device":"cpu",
        "rounds":2,"candidates":3,"items":2,"replicates":2,"student_steps":4,
        "prompts_per_step":1,"completions":4,"teacher_lr":0.001,"student_lr":0.002,
        "kl_coef":0.01,"input_limit":8192,"output_limit":1,"seed":0,"specs_per_family":1,
        "feedback":"cohesion","teacher_updates":True,"domain":"math"}
