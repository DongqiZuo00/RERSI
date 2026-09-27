"""State-balanced semantic patching on separate calibration trajectories (B.2)."""
from collections import defaultdict
import numpy as np
from .diagnostics import continuation_value
from .tasks import safe_expression,_text


def select_layer(policy,records,checkpoints,layers):
    if not checkpoints: raise ValueError("Supply independent calibration trajectory checkpoints")
    if any(x["split"]!="calibration" for x in records): raise ValueError("Calibration split required")
    layers=sorted(set(layers))
    grouped=defaultdict(lambda:defaultdict(list))
    for i,r in enumerate(records): grouped[(r["family"],r["specification"])][r["state"]].append(i)
    pairs=[]
    for (family,spec),states in sorted(grouped.items()):
        labels=sorted(states)
        if len(labels)<2: raise ValueError("Patching needs distinct states")
        for i,label in enumerate(labels):
            recipients=states[labels[(i+1)%len(labels)]]
            for j,source in enumerate(states[label]):
                recipient=recipients[(j+1)%len(recipients)]
                value=safe_expression(records[source]["reference_state"])
                answer=_text(continuation_value(value,records[recipient]["continuation"]))
                if answer==records[source]["answer"]: continue
                pairs.append((family,spec,label,source,recipient,answer))
    if not pairs: raise ValueError("No valid counterfactual pairs")
    scores={layer:[] for layer in layers}
    for checkpoint in checkpoints:
        policy.load(checkpoint)
        for layer in layers:
            hidden=policy.hidden(records,layer)
            by_state=defaultdict(list)
            for family,spec,state,source,recipient,answer in pairs:
                base=policy.answer_probability(records[recipient],answer)
                patched=policy.answer_probability(records[recipient],answer,layer,hidden[source])
                by_state[(family,spec,state)].append(patched-base)
            by_spec=defaultdict(list)
            for (family,spec,_),gains in by_state.items(): by_spec[(family,spec)].append(np.mean(gains))
            by_family=defaultdict(list)
            for (family,_),gains in by_spec.items(): by_family[family].append(np.mean(gains))
            scores[layer].append(float(np.mean([np.mean(g) for g in by_family.values()])))
    means={layer:float(np.mean(v)) for layer,v in scores.items()}
    selected=max(layers,key=lambda layer:(means[layer],-layer))
    return {"layer":selected,"per_layer_probability_gain":means,"checkpoints":len(checkpoints),"pairs":len(pairs)}
