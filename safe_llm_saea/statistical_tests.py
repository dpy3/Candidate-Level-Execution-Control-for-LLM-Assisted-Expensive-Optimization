import numpy as np
from typing import List, Dict
try:
    from scipy.stats import mannwhitneyu, ttest_ind
except Exception:
    mannwhitneyu = None
    ttest_ind = None

def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def compare(group_a: List[float], group_b: List[float]) -> Dict[str, float]:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    res = {}
    if mannwhitneyu is not None:
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        res["mannwhitney_u"] = float(u)
        res["mannwhitney_p"] = float(p)
    if ttest_ind is not None:
        t, p2 = ttest_ind(a, b, equal_var=False)
        res["t_value"] = float(t)
        res["t_p"] = float(p2)
    return res

def average_ranks(matrix: List[Dict[str, float]]) -> Dict[str, float]:
    methods = sorted({m for row in matrix for m in row.keys()})
    ranks = {m: 0.0 for m in methods}
    counts = {m: 0 for m in methods}
    for row in matrix:
        sorted_items = sorted(row.items(), key=lambda x: x[1])
        for rank, (m, _) in enumerate(sorted_items, start=1):
            ranks[m] += rank
            counts[m] += 1
    out = {}
    for m in methods:
        if counts[m] > 0:
            out[m] = ranks[m] / float(counts[m])
    return out

def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, bool]:
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    decisions = {}
    for i, (name, p) in enumerate(items, start=1):
        threshold = alpha / (m - i + 1)
        decisions[name] = p <= threshold
        if not decisions[name]:
            for j in range(i+1, m+1):
                if j-1 < len(items):
                    decisions[items[j-1][0]] = False
            break
    return decisions

def cliffs_delta(a: List[float], b: List[float]) -> float:
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    gt = 0
    lt = 0
    for x in A:
        gt += int(np.sum(x > B))
        lt += int(np.sum(x < B))
    n1 = len(A)
    n2 = len(B)
    return float((gt - lt) / (n1 * n2))
