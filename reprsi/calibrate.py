from collections import defaultdict
import numpy as np
from .diagnostics import continuation_value
from .tasks import safe_expression,_text
from .storage import atomic_json,digest,file_digest


def select_layer(policy,records,checkpoints,layers,progress_path=None,resume=False):
    if not checkpoints: raise ValueError("Supply independent calibration trajectory checkpoints")
    if any(x["split"]!="calibration" for x in records): raise ValueError("Calibration split required")
    layers=sorted(set(layers))
    if not layers or any(not 0<=x<len(policy.blocks) for x in layers):raise ValueError("Invalid calibration layers")
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
                value=(records[source]["reference_state"] if records[source].get("answer_format")=="entity"
                       else safe_expression(records[source]["reference_state"]))
                answer=_text(continuation_value(value,records[recipient]["continuation"]))
                pairs.append((family,spec,label,source,recipient,answer))
    if not pairs: raise ValueError("No valid counterfactual pairs")
    scores={layer:[] for layer in layers}
    completed={}
    signature=digest({"records":records,"checkpoints":[file_digest(x) for x in checkpoints],"layers":layers})
    if progress_path:
        from pathlib import Path
        import json
        progress_path=Path(progress_path)
        if progress_path.exists():
            saved=json.loads(progress_path.read_text())
            if not resume or saved["signature"]!=signature:raise ValueError("Calibration exists or inputs changed")
            completed=saved["completed"]
        elif resume:raise ValueError("No calibration to resume")
        atomic_json(progress_path,{"signature":signature,"completed":completed})
    for checkpoint_index,checkpoint in enumerate(checkpoints):
        policy.load(checkpoint)
        for layer in layers:
            key=f"{checkpoint_index}/{layer}"
            if key in completed:
                scores[layer].append(completed[key]);continue
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
            value=float(np.mean([np.mean(g) for g in by_family.values()]))
            scores[layer].append(value);completed[key]=value
            if progress_path:atomic_json(progress_path,{"signature":signature,"completed":completed})
    means={layer:float(np.mean(v)) for layer,v in scores.items()}
    selected=max(layers,key=lambda layer:(means[layer],-layer))
    return {"layer":selected,"per_layer_probability_gain":means,"checkpoints":len(checkpoints),"pairs":len(pairs),"calibration_signature":signature}
