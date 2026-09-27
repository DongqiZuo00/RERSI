"""Disjoint exact mathematical diagnostics with fixed pre-continuation anchors."""
from collections import defaultdict
import json
import random
from pathlib import Path
import sympy as sp
from .metrics import seed_for
from .tasks import Z, _text

BOUNDARY = "\n\nContinuation: "


def make_math_pool(split, specifications=64, seed=2026):
    """Four families x specifications x four states x four contexts.

    These are deterministic new diagnostic realizations, not the unreleased
    original manifests. The continuation is recorded for patching calibration.
    """
    if split not in {"reward", "monitor", "calibration"}:
        raise ValueError("Unknown diagnostic split")
    out=[]
    for family in ("rational", "polynomial", "modular", "linear"):
        for spec in range(specifications):
            rng=random.Random(seed_for(seed,split,family,spec))
            # Non-overlapping numeric ranges make rendered contexts distinct across
            # specifications and pool roles, independent of accidental RNG collisions.
            role={"reward":0,"monitor":1,"calibration":2}[split]
            offset=100+role*10000+spec*50+rng.randrange(10)
            denominator=rng.randrange(2,20)
            modulus=rng.choice([5,7,11,13,17,19,23,29,31])
            for state_id in range(4):
                n=offset+state_id+1
                if family=="rational":
                    value=sp.Rational(n,denominator)
                    state=str(value)
                elif family=="polynomial":
                    value=sp.expand((Z+state_id+1)*(Z+offset))
                    state=str(sp.Poly(value,Z).all_coeffs())
                elif family=="modular":
                    value=sp.Integer(state_id)
                    state=f"{modulus}:{state_id}"
                else:
                    value=(sp.Integer(n),sp.Integer(n+2))
                    state=str([[1,0,n],[0,1,n+2]])
                for context in range(4):
                    shift=offset+context+7
                    if family=="rational":
                        expression=f"({n+shift*denominator}/{denominator}) - {shift}"
                        prefix="Compute the exact value of: "+expression
                        cont={"kind":"affine","a":context+2,"b":context-1}
                        suffix=f"Multiply this value by {cont['a']} and add {cont['b']}."
                    elif family=="polynomial":
                        a,b=state_id+1,offset
                        expression=f"(z+{a+shift})*(z+{b}) - {shift}*(z+{b})"
                        prefix="Form the polynomial in z: "+expression
                        cont={"kind":"poly_eval","at":context+1}
                        suffix=f"Evaluate this polynomial at z={context+1}."
                    elif family=="modular":
                        expression=f"({state_id+modulus*shift}+{shift})-{shift}"
                        prefix=f"Compute the residue modulo {modulus} of: "+expression
                        cont={"kind":"mod_affine","a":context+1,"b":context,"m":modulus}
                        suffix=f"Multiply the residue by {context+1}, add {context}, and reduce modulo {modulus}."
                    else:
                        expression=f"x+{shift}*y={n+shift*(n+2)}; y={n+2}"
                        prefix="Solve this system in variable order (x,y): "+expression
                        cont={"kind":"linear_form","coefficients":[context+1,2]}
                        suffix=f"Evaluate {context+1}*x+2*y."
                    answer=continuation_value(value,cont)
                    out.append({"id":f"{split}/{family}/{spec}/{state_id}/{context}",
                        "split":split,"family":family,"specification":f"{split}/{family}/{spec}",
                        "state":state,"context":str(context),"prefix":prefix,
                        "continuation":cont,"prompt":prefix+BOUNDARY+suffix+" Return the answer in \\boxed{...}.",
                        "reference_state":_text(value),"answer":_text(answer)})
    return out


def continuation_value(state, continuation):
    c=continuation
    if c["kind"]=="affine": return state*c["a"]+c["b"]
    if c["kind"]=="poly_eval": return state.subs(Z,c["at"])
    if c["kind"]=="mod_affine": return sp.Integer((int(state)*c["a"]+c["b"])%c["m"])
    if c["kind"]=="linear_form": return sum(a*b for a,b in zip(state,c["coefficients"]))
    if c["kind"]=="dfa_suffix":
        node=int(state)
        for char in c["suffix"]: node=c["transitions"][str(node)][char]
        return sp.Integer(node in c["accepting"])
    raise ValueError("Unknown continuation")


def sample_batch(pool, specs_per_family, seed):
    groups=defaultdict(lambda:defaultdict(list))
    for r in pool: groups[r["family"]][r["specification"]].append(r)
    rng=random.Random(seed)
    selected=[]
    for family, specs in sorted(groups.items()):
        if len(specs)<specs_per_family: raise ValueError("Diagnostic pool too small")
        for spec in rng.sample(sorted(specs),specs_per_family): selected.extend(specs[spec])
    return selected


def validate_pools(pools):
    seen_ids,seen_prompts,seen_specs=set(),set(),set()
    for pool in pools:
        ids={r["id"] for r in pool}
        prompts={r["prompt"] for r in pool}
        specs={r["specification"] for r in pool}
        if len(ids)!=len(pool) or len(prompts)!=len(pool): raise ValueError("Duplicate diagnostic item")
        if ids & seen_ids or prompts & seen_prompts or specs & seen_specs: raise ValueError("Diagnostic split overlap")
        seen_ids.update(ids); seen_prompts.update(prompts); seen_specs.update(specs)
        for r in pool:
            if not r["prompt"].startswith(r["prefix"]+BOUNDARY): raise ValueError("Invalid anchor boundary")


def read_jsonl(path):
    with open(path,encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps(row,ensure_ascii=False)+"\n")
