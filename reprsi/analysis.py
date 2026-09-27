from collections import defaultdict
import itertools
import json
from pathlib import Path
import numpy as np
from .storage import atomic_json


def mean_sd(values):
    x=np.asarray(values,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise ValueError("Finite observations are required")
    return {"mean":float(x.mean()),"sd":float(x.std(ddof=1)) if len(x)>1 else None,"n":len(x)}


def aggregate(summaries,baseline=None):
    if not summaries:raise ValueError("No evaluation summaries")
    protocol=("model","revision","task_digest","samples","greedy","benchmark","split")
    anchor=summaries[0]
    if any(any(r.get(k)!=anchor.get(k) for k in protocol) for r in summaries):raise ValueError("Evaluation protocols differ")
    seeds=[r["seed"] for r in summaries]
    if len(seeds)!=len(set(seeds)):raise ValueError("Duplicate root seed")
    if len({r["method"] for r in summaries})!=1:raise ValueError("Aggregate one method at a time")
    metrics=sorted(k for k in anchor if k.startswith("pass@"))
    if not metrics or any(r.get(k) is None for r in summaries for k in metrics):raise ValueError("No nonempty pass@k results")
    result={k:anchor.get(k) for k in protocol}
    result.update(method=anchor["method"],seeds=sorted(seeds),paper_five_seed_complete=set(seeds)==set(range(5)),
                  scores={k:mean_sd([100*r[k] for r in summaries]) for k in metrics},unit="percentage_points")
    if baseline is not None:
        by_seed={r["seed"]:r for r in baseline}
        if len(by_seed)!=len(baseline) or set(by_seed)!=set(seeds):raise ValueError("Baseline seed pairing differs")
        if any(any(r.get(k)!=anchor.get(k) for k in protocol) for r in baseline):raise ValueError("Baseline evaluation protocol differs")
        if len({r["method"] for r in baseline})!=1:raise ValueError("Mixed baseline methods")
        result["baseline"]=baseline[0]["method"]
        result["paired_differences"]={k:mean_sd([100*(r[k]-by_seed[r["seed"]][k]) for r in summaries]) for k in metrics}
    return result


def ranking_accuracy(target,prediction,groups):
    scores=[]
    for group in set(groups):
        indices=[i for i,g in enumerate(groups) if g==group]
        for i,j in itertools.combinations(indices,2):
            if target[i]==target[j]:continue
            scores.append(0.5 if prediction[i]==prediction[j] else float((target[i]-target[j])*(prediction[i]-prediction[j])>0))
    return {"accuracy":100*float(np.mean(scores)) if scores else None,"pairs":len(scores)}


def ridge_predict(x,y,test,penalty):
    mean=x.mean(0);scale=x.std(0);scale[scale==0]=1
    z=(x-mean)/scale;center=y.mean()
    coefficient=np.linalg.solve(z.T@z+penalty*np.eye(x.shape[1]),z.T@(y-center))
    return ((test-mean)/scale)@coefficient+center


def fit_grouped(x,y,runs,train_mask,test_mask):
    train_runs=sorted(set(runs[train_mask]))
    if len(train_runs)<4:raise ValueError("Need at least four training runs for grouped four-fold CV")
    rng=np.random.default_rng(2027);rng.shuffle(train_runs);folds=np.array_split(train_runs,4)
    penalties=(.001,.01,.1,1.,10.,100.,1000.);scores=[]
    for penalty in penalties:
        errors=[]
        for fold in folds:
            validation=train_mask & np.isin(runs,fold);fitting=train_mask & ~validation
            pred=ridge_predict(x[fitting],y[fitting],x[validation],penalty)
            errors.extend(np.abs(y[validation]-pred))
        scores.append(float(np.mean(errors)))
    penalty=penalties[int(np.argmin(scores))]
    return ridge_predict(x[train_mask],y[train_mask],x[test_mask],penalty),penalty


def prediction_analysis(rows,likelihood=False,repetitions=20):
    features=["current_accuracy","curriculum_accuracy","state_accuracy"]
    if likelihood:features += ["answer_logprob_change","state_logprob_change"]
    groups=defaultdict(list)
    for row in rows:groups[(row["run"],row["group"],row["candidate"])].append(row)
    observations=[]
    for key,trials in sorted(groups.items()):
        trial_ids=[r.get("trial",0) for r in trials]
        if len(set(trial_ids))!=len(trial_ids):raise ValueError("Duplicate trial in a curriculum observation")
        identity={r["curriculum_digest"] for r in trials}
        if len(identity)!=1:raise ValueError("Curriculum changed between replicates")
        observations.append({"run":key[0],"group":key[1],"candidate":key[2],"curriculum_digest":identity.pop(),
            **{k:float(np.mean([r[k] for r in trials])) for k in features+["delta_phi","future_accuracy"]}})
    runs=np.array([r["run"] for r in observations]);unique=sorted(set(runs.tolist()))
    if len(unique)<5:raise ValueError("Need at least five complete runs for held-out prediction analysis")
    group_ids=[(r["run"],r["group"]) for r in observations]
    if any(sum(g==key for g in group_ids)!=4 for key in set(group_ids)):raise ValueError("Each group must contain four curricula")
    rng=np.random.default_rng(2027);rng.shuffle(unique);n_train=int(len(unique)*.8)
    train=np.isin(runs,unique[:n_train]);test=~train
    a={r["curriculum_digest"] for i,r in enumerate(observations) if train[i]}
    b={r["curriculum_digest"] for i,r in enumerate(observations) if test[i]}
    if a&b:raise ValueError("Curriculum content overlaps fitting and held-out runs")
    x=np.array([[r[k] for k in features] for r in observations]);delta=np.array([r["delta_phi"] for r in observations])
    y=np.array([r["future_accuracy"] for r in observations]);test_groups=[g for g,keep in zip(group_ids,test) if keep]
    if not np.isfinite(x).all() or not np.isfinite(delta).all() or not np.isfinite(y).all():raise ValueError("Non-finite prediction data")
    def condition(values):
        pred,penalty=fit_grouped(values,y,runs,train,test)
        return {"mae_pp":float(np.mean(np.abs(y[test]-pred))),"ranking":ranking_accuracy(y[test],pred,test_groups),"ridge_penalty":penalty}
    result={"behavioral":condition(x),"behavioral_plus_cohesion":condition(np.column_stack((x,delta))),
            "features":features,"train_runs":sorted(map(int,unique[:n_train])),"test_runs":sorted(map(int,unique[n_train:])),
            "observations":len(observations),"trials_per_observation":sorted({len(v) for v in groups.values()})}
    shuffled=[];rng=np.random.default_rng(2027)
    for _ in range(repetitions):
        values=delta.copy()
        for group in sorted(set(group_ids)):
            indices=np.array([i for i,g in enumerate(group_ids) if g==group]);values[indices]=rng.permutation(values[indices])
        shuffled.append(condition(np.column_stack((x,values))))
    result["shuffled"]={"repetitions":repetitions,"mean_mae_pp":float(np.mean([r["mae_pp"] for r in shuffled])),
        "mean_ranking_accuracy":float(np.mean([r["ranking"]["accuracy"] for r in shuffled])) if shuffled[0]["ranking"]["accuracy"] is not None else None}
    return result


def fresh_curves(run_directories):
    teachers=defaultdict(list);seen=set();protocol=None
    for directory in run_directories:
        directory=Path(directory);summary=json.loads((directory/"summary.json").read_text())
        key=(summary["teacher_id"],summary["student_seed"])
        if key in seen:raise ValueError("Duplicate teacher/student pair")
        seen.add(key)
        current=(summary["model"],summary["revision"],summary["inputs"]["evaluation"],summary["steps"])
        if protocol is not None and current!=protocol:raise ValueError("Fresh-student protocols differ")
        protocol=current
        records=[json.loads(p.read_text()) for p in sorted(directory.glob("round_*.json"))]
        initial=directory/"initial_metrics.json"
        if initial.exists():records.insert(0,json.loads(initial.read_text()))
        curve={r["student_steps"]:100*r["evaluation"]["accuracy"] for r in records if "evaluation" in r}
        teachers[summary["teacher_id"]].append(curve)
    if not teachers:raise ValueError("No fresh-student runs")
    all_curves=[c for curves in teachers.values() for c in curves]
    steps=set(all_curves[0])
    if not steps or any(set(c)!=steps for c in all_curves):raise ValueError("Evaluation checkpoints differ or are absent")
    within={str(t):{str(s):mean_sd([c[s] for c in curves]) for s in sorted(steps)} for t,curves in teachers.items()}
    between={str(s):mean_sd([np.mean([c[s] for c in curves]) for curves in teachers.values()]) for s in sorted(steps)}
    return {"within_teacher":within,"between_teacher_means":between,"unit":"percent"}


def first_crossing(curve,target):
    if not np.isfinite(target):raise ValueError("Invalid target")
    if any(curve[i][0]>curve[i+1][0] for i in range(len(curve)-1)):raise ValueError("Compute checkpoints must be ordered")
    return next((float(cost) for cost,mean in curve if mean>=target),None)


def plot_fresh(report,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(6,4))
    for teacher,curve in report["within_teacher"].items():
        steps=sorted(map(int,curve));mean=np.array([curve[str(s)]["mean"] for s in steps])
        sd=np.array([curve[str(s)]["sd"] or 0 for s in steps])
        axis.plot(steps,mean,label=f"Teacher {teacher}");axis.fill_between(steps,mean-sd,mean+sd,alpha=.15)
    axis.set(xlabel="Student optimizer steps",ylabel="Greedy accuracy (%)",ylim=(0,100))
    axis.legend();fig.tight_layout();fig.savefig(output);plt.close(fig)
