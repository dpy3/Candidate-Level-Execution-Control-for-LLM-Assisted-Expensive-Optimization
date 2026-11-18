from typing import Callable, Tuple
import numpy as np
try:
    from opfunu.cec_based.cec2022 import F12022, F22022, F32022, F42022, F52022, F62022, F72022, F82022, F92022, F102022, F112022, F122022
except Exception:
    F12022=F22022=F32022=F42022=F52022=F62022=F72022=F82022=F92022=F102022=F112022=F122022=None

def get_function(fid: int, dim: int) -> Callable:
    if F12022 is None:
        raise RuntimeError("opfunu cec2022 not available")
    funcs = [F12022, F22022, F32022, F42022, F52022, F62022, F72022, F82022, F92022, F102022, F112022, F122022]
    obj = funcs[fid-1](dim)
    def f(x):
        return float(obj.evaluate(np.asarray(x, dtype=float)))
    return f

def get_bounds(dim: int) -> Tuple[float, float]:
    return -100.0, 100.0
