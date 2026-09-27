from collections import defaultdict
import hashlib
import math
import numpy as np


def seed_for(root, *parts):
    payload = "|".join(map(str, (root,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def cohesion(hidden, records, eps=1e-8):
    h = np.asarray(hidden, dtype=np.float64)
    if h.ndim != 2 or len(h) != len(records) or not np.isfinite(h).all():
        raise ValueError("Invalid diagnostic representations")
    z = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), eps)
    groups = defaultdict(lambda: defaultdict(list))
    contexts = set()
    for i, r in enumerate(records):
        key = (r["family"], r["specification"])
        identity = (key, r["state"], r["context"])
        if identity in contexts:
            raise ValueError("Duplicate diagnostic context")
        contexts.add(identity)
        groups[key][r["state"]].append(i)
    by_family = defaultdict(list)
    for (family, _), states in groups.items():
        if len(states) < 2 or any(len(ids) < 2 for ids in states.values()):
            raise ValueError("Every type needs >=2 states and >=2 contexts/state")
        positive, centroids = [], []
        for ids in states.values():
            x = z[ids]
            n = len(x)
            positive.append(((x.sum(0) @ x.sum(0)) - (x * x).sum()) / (n * (n - 1)))
            centroids.append(x.mean(0))
        m = np.stack(centroids)
        n = len(m)
        negative = ((m.sum(0) @ m.sum(0)) - (m * m).sum()) / (n * (n - 1))
        by_family[family].append((float(np.mean(positive)), float(negative)))
    if not by_family:
        raise ValueError("Empty diagnostic batch")
    c_plus, c_minus = np.mean([np.mean(v, axis=0) for v in by_family.values()], axis=0)
    return {"phi": float(c_plus - c_minus), "c_plus": float(c_plus), "c_minus": float(c_minus)}


def teacher_rewards(raw, eps=1e-8):
    valid = [float(x) for x in raw if x is not None]
    if not valid:
        return None
    if not all(math.isfinite(x) for x in valid):
        raise ValueError("Non-finite reward is an evaluation failure")
    mu = float(np.mean(valid))
    sigma = float(np.std(valid, ddof=0))
    return np.array([-5.0 if x is None else (x - mu) / max(sigma, eps) for x in raw])


def rloo(rewards):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all():
        raise ValueError("RLOO needs at least two finite rewards")
    return r - (r.sum() - r) / (len(r) - 1)


def choose_branch(raw):
    valid = [i for i, x in enumerate(raw) if x is not None]
    return max(valid, key=lambda i: (raw[i], -i)) if valid else None


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
