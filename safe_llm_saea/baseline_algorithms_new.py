import numpy as np
import math
import importlib
from scipy.special import gamma
from typing import Any, Callable, List, Tuple, Dict

def _safe_obj(func: Callable, x: np.ndarray) -> float:
    try:
        val = float(func(x))
        if np.isnan(val) or np.isinf(val):
            return 1e15 # Large finite penalty
        return val
    except Exception:
        return 1e15

def _clip(x, bounds):
    dim = len(x)
    for j in range(dim):
        lo, hi = bounds[j]
        x[j] = float(np.clip(x[j], lo, hi))
    return x

def _levy(n, m, beta):
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))
    z = u / (np.abs(v) ** (1 / beta))
    return z

def _bounds_arrays(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    arr = np.array(bounds, dtype=float)
    return lb, ub, arr

def _patch_numpy_float_alias() -> None:
    # pyade still imports np.float on newer NumPy versions.
    if not hasattr(np, "float"):
        setattr(np, "float", float)

def _pyade_callback_factory(curve: List[float]) -> Callable[..., None]:
    def _callback(**state: Any) -> None:
        fitness = state.get("fitness")
        if fitness is None:
            return
        try:
            curve.append(float(np.min(fitness)))
        except Exception:
            pass
    return _callback

def _run_pyade_algorithm(module_name: str, objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    _patch_numpy_float_alias()
    module = importlib.import_module(module_name)
    dim = len(bounds)
    _, _, bound_arr = _bounds_arrays(bounds)
    curve: List[float] = []
    params = module.get_default_params(dim)
    params["bounds"] = bound_arr
    params["func"] = lambda x, _opts=None: _safe_obj(objective, np.asarray(x, dtype=float))
    params["population_size"] = max(8, int(pop_size))
    params["max_evals"] = max(dim + 1, int(max_evals))
    params["callback"] = _pyade_callback_factory(curve)
    solution, fitness = module.apply(**params)
    best_solution = np.asarray(solution, dtype=float).copy()
    best_fitness = float(fitness)
    if not curve:
        curve = [best_fitness]
    return {
        "best_fitness": best_fitness,
        "evaluations_used": int(params["max_evals"]),
        "convergence_curve": curve,
        "best_solution": best_solution,
    }

def run_alo(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    ants = np.random.uniform(lb, ub, (pop_size, dim))
    antlions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness_ants = np.full(pop_size, np.inf)
    fitness_antlions = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        fitness_ants[i] = _safe_obj(objective, ants[i])
        fitness_antlions[i] = _safe_obj(objective, antlions[i])
        
    # Sort antlions
    sorted_idx = np.argsort(fitness_antlions)
    antlions = antlions[sorted_idx]
    fitness_antlions = fitness_antlions[sorted_idx]
    
    elite_antlion = antlions[0].copy()
    elite_score = fitness_antlions[0]
    
    evals = 2 * pop_size
    curve = [elite_score]
    
    # ALO is usually iter-based, but we map to evals
    # Max_iter estimate
    max_iter = (max_evals - evals) // pop_size
    if max_iter < 1: max_iter = 1
    
    current_iter = 0
    while evals < max_evals:
        current_iter += 1
        # Adaptive I
        I = 1
        if current_iter > max_iter * 0.1: I = 1 + 100 * (current_iter / max_iter)
        if current_iter > max_iter * 0.5: I = 1 + 1000 * (current_iter / max_iter)
        if current_iter > max_iter * 0.75: I = 1 + 10000 * (current_iter / max_iter)
        if current_iter > max_iter * 0.9: I = 1 + 100000 * (current_iter / max_iter)
        if current_iter > max_iter * 0.95: I = 1 + 1000000 * (current_iter / max_iter)
        
        for i in range(pop_size):
            # Roulette wheel selection
            rolette_idx = -1
            r = np.random.rand()
            # Simple roulette (using 1/fitness or rank) - simplified to random top half for robustness
            rolette_idx = np.random.randint(0, max(1, pop_size // 2))
            selected_antlion = antlions[rolette_idx]
            
            # Random walk around selected antlion and elite
            # Normalized random walks
            
            # RA (Selected)
            c = lb / I
            d = ub / I
            if np.random.rand() > 0.5:
                c = c + selected_antlion
                d = d + selected_antlion
            else:
                c = -c + selected_antlion
                d = -d + selected_antlion
                
            # Random walk generation (simplified)
            X = np.cumsum(2 * (np.random.rand(max_iter, dim) > 0.5) - 1, axis=0)
            # Normalize
            a_min = np.min(X, axis=0)
            a_max = np.max(X, axis=0)
            X_norm_RA = (X - a_min) * (d - c) / (a_max - a_min + 1e-15) + c
            
            # RE (Elite)
            c = lb / I
            d = ub / I
            if np.random.rand() > 0.5:
                c = c + elite_antlion
                d = d + elite_antlion
            else:
                c = -c + elite_antlion
                d = -d + elite_antlion
                
            X = np.cumsum(2 * (np.random.rand(max_iter, dim) > 0.5) - 1, axis=0)
            a_min = np.min(X, axis=0)
            a_max = np.max(X, axis=0)
            X_norm_RE = (X - a_min) * (d - c) / (a_max - a_min + 1e-15) + c
            
            # Current step
            idx_step = min(current_iter, max_iter - 1)
            ant_pos = (X_norm_RA[idx_step] + X_norm_RE[idx_step]) / 2
            
            # Bound check
            ant_pos = np.clip(ant_pos, lb, ub)
            ants[i] = ant_pos
            
            # Evaluate
            f_new = _safe_obj(objective, ant_pos)
            evals += 1
            fitness_ants[i] = f_new
            
            if f_new < elite_score:
                elite_score = f_new
                elite_antlion = ant_pos.copy()
                
            if evals >= max_evals: break
            
        # Elitism
        combined_pos = np.vstack((antlions, ants))
        combined_fit = np.concatenate((fitness_antlions, fitness_ants))
        
        sorted_idx = np.argsort(combined_fit)
        antlions = combined_pos[sorted_idx[:pop_size]]
        fitness_antlions = combined_fit[sorted_idx[:pop_size]]
        
        if fitness_antlions[0] < elite_score:
            elite_score = fitness_antlions[0]
            elite_antlion = antlions[0].copy()
        else:
            antlions[0] = elite_antlion.copy()
            fitness_antlions[0] = elite_score
            
        curve.append(elite_score)
        
    return {"best_fitness": elite_score, "evaluations_used": evals, "convergence_curve": curve, "best_solution": elite_antlion.copy()}

def run_ba(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Parameters
    Qmin = 0
    Qmax = 2
    A = 0.5
    r0 = 0.5
    alpha = 0.9
    gamma_const = 0.9
    
    freq = np.zeros(pop_size)
    v = np.zeros((pop_size, dim))
    pulse_rate = np.full(pop_size, r0)
    loudness = np.full(pop_size, A)
    
    sol = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        fitness[i] = _safe_obj(objective, sol[i])
        
    evals = pop_size
    
    best_idx = np.argmin(fitness)
    best_sol = sol[best_idx].copy()
    best_fit = fitness[best_idx]
    
    curve = [best_fit]
    
    while evals < max_evals:
        avg_loudness = np.mean(loudness)
        
        for i in range(pop_size):
            beta = np.random.rand()
            freq[i] = Qmin + (Qmax - Qmin) * beta
            v[i] = v[i] + (sol[i] - best_sol) * freq[i]
            S = sol[i] + v[i]
            
            # Local Search
            if np.random.rand() > pulse_rate[i]:
                S = best_sol + 0.001 * np.random.randn(dim) * avg_loudness
                
            S = np.clip(S, lb, ub)
            f_new = _safe_obj(objective, S)
            evals += 1
            
            # Greedy + Loudness/Pulse logic
            if (f_new <= fitness[i]) and (np.random.rand() < loudness[i]):
                sol[i] = S
                fitness[i] = f_new
                loudness[i] *= alpha
                pulse_rate[i] = r0 * (1 - np.exp(-gamma_const * evals)) # Approx iter
                
            if f_new <= best_fit:
                best_fit = f_new
                best_sol = S.copy()
                
            if evals >= max_evals: break
            
        curve.append(best_fit)
        
    return {"best_fitness": best_fit, "evaluations_used": evals, "convergence_curve": curve, "best_solution": best_sol.copy()}

def run_gwo(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    alpha_pos = np.zeros(dim)
    alpha_score = np.inf
    beta_pos = np.zeros(dim)
    beta_score = np.inf
    delta_pos = np.zeros(dim)
    delta_score = np.inf
    
    positions = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    
    evals = 0
    curve = []
    
    # Calculate max_iter approximately
    max_iter = max_evals // pop_size
    it = 0
    
    while evals < max_evals:
        for i in range(pop_size):
            # Bound checking (Reflection or Clip? MATLAB code uses complex reflection/clip logic)
            # Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb; -> This is clipping
            positions[i] = np.clip(positions[i], lb, ub)
            
            fitness[i] = _safe_obj(objective, positions[i])
            evals += 1
            
            if fitness[i] < alpha_score:
                alpha_score = fitness[i]
                alpha_pos = positions[i].copy()
            if fitness[i] > alpha_score and fitness[i] < beta_score:
                beta_score = fitness[i]
                beta_pos = positions[i].copy()
            if fitness[i] > alpha_score and fitness[i] > beta_score and fitness[i] < delta_score:
                delta_score = fitness[i]
                delta_pos = positions[i].copy()
                
            if evals >= max_evals: break
            
        curve.append(alpha_score)
        if evals >= max_evals: break
        
        a = 2 - it * (2 / max_iter)
        
        for i in range(pop_size):
            for j in range(dim):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                
                r1 = np.random.rand()
                r2 = np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                
                r1 = np.random.rand()
                r2 = np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                
                positions[i, j] = (X1 + X2 + X3) / 3
        it += 1
        
    return {"best_fitness": alpha_score, "evaluations_used": evals, "convergence_curve": curve, "best_solution": alpha_pos.copy()}

def run_hho(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    X = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    
    rabbit_pos = np.zeros(dim)
    rabbit_score = np.inf
    
    evals = 0
    curve = []
    max_iter = max_evals // pop_size
    t = 0
    
    # Init fitness
    for i in range(pop_size):
        fitness[i] = _safe_obj(objective, X[i])
        evals += 1
        if fitness[i] < rabbit_score:
            rabbit_score = fitness[i]
            rabbit_pos = X[i].copy()
    
    while evals < max_evals:
        E1 = 2 * (1 - t / max_iter)
        
        for i in range(pop_size):
            E0 = 2 * np.random.rand() - 1
            Escaping_Energy = E1 * E0
            
            if abs(Escaping_Energy) >= 1:
                # Exploration
                q = np.random.rand()
                rand_idx = np.random.randint(0, pop_size)
                X_rand = X[rand_idx]
                if q < 0.5:
                    X[i] = X_rand - np.random.rand() * abs(X_rand - 2 * np.random.rand() * X[i])
                else:
                    X_mean = np.mean(X, axis=0)
                    X[i] = (rabbit_pos - X_mean) - np.random.rand() * (lb + np.random.rand(dim) * (ub - lb))
            else:
                # Exploitation
                r = np.random.rand()
                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i] = rabbit_pos - Escaping_Energy * abs(rabbit_pos - X[i])
                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - np.random.rand())
                    X[i] = (rabbit_pos - X[i]) - Escaping_Energy * abs(Jump_strength * rabbit_pos - X[i])
                elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - np.random.rand())
                    X1 = rabbit_pos - Escaping_Energy * abs(Jump_strength * rabbit_pos - X[i])
                    X1 = np.clip(X1, lb, ub)
                    
                    if _safe_obj(objective, X1) < fitness[i]: # Note: This adds evals!
                        X[i] = X1
                        evals += 1
                    else:
                        # Levy
                        X2 = rabbit_pos - Escaping_Energy * abs(Jump_strength * rabbit_pos - X[i]) + np.random.rand(dim) * _levy(1, dim, 1.5).flatten()
                        X2 = np.clip(X2, lb, ub)
                        if _safe_obj(objective, X2) < fitness[i]:
                            X[i] = X2
                            evals += 1
                        else:
                            evals += 1 # Counted 2 evals here roughly
                            
                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    Jump_strength = 2 * (1 - np.random.rand())
                    X1 = rabbit_pos - Escaping_Energy * abs(Jump_strength * rabbit_pos - np.mean(X, axis=0))
                    X1 = np.clip(X1, lb, ub)
                    
                    if _safe_obj(objective, X1) < fitness[i]:
                        X[i] = X1
                        evals += 1
                    else:
                        X2 = rabbit_pos - Escaping_Energy * abs(Jump_strength * rabbit_pos - np.mean(X, axis=0)) + np.random.rand(dim) * _levy(1, dim, 1.5).flatten()
                        X2 = np.clip(X2, lb, ub)
                        if _safe_obj(objective, X2) < fitness[i]:
                            X[i] = X2
                            evals += 1
                        else:
                            evals += 1

            # Simple bound check
            X[i] = np.clip(X[i], lb, ub)
            
            # Recalculate current fitness if changed (Approx logic, HHO usually evals during update)
            # To strictly follow standard, we evaluate at end of loop or during update.
            # The above if-else branches already did some evaluations.
            # Let's simplify: Standard HHO evaluates trial solutions.
            # Here I assume the `evals` counter is incremented inside the branches.
            
            fit_curr = _safe_obj(objective, X[i])
            evals += 1
            if fit_curr < fitness[i]:
                fitness[i] = fit_curr
                if fit_curr < rabbit_score:
                    rabbit_score = fit_curr
                    rabbit_pos = X[i].copy()
            
            if evals >= max_evals: break
            
        curve.append(rabbit_score)
        t += 1
        
    return {"best_fitness": rabbit_score, "evaluations_used": evals, "convergence_curve": curve, "best_solution": rabbit_pos.copy()}

def run_mfo(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    moths = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        fitness[i] = _safe_obj(objective, moths[i])
        
    evals = pop_size
    
    sorted_fit = np.sort(fitness)
    sorted_idx = np.argsort(fitness)
    flames = moths[sorted_idx].copy()
    best_flame_score = sorted_fit[0]
    
    curve = [best_flame_score]
    max_iter = max_evals // pop_size
    iter_num = 0
    
    while evals < max_evals:
        iter_num += 1
        flame_no = round(pop_size - iter_num * ((pop_size - 1) / max_iter))
        
        a = -1 + iter_num * ((-1) / max_iter)
        
        for i in range(pop_size):
            for j in range(dim):
                if i <= flame_no:
                    dist = abs(flames[i, j] - moths[i, j])
                    b = 1
                    t = (a - 1) * np.random.rand() + 1
                    moths[i, j] = dist * np.exp(b * t) * np.cos(2 * np.pi * t) + flames[i, j]
                else:
                    dist = abs(flames[flame_no, j] - moths[i, j])
                    b = 1
                    t = (a - 1) * np.random.rand() + 1
                    moths[i, j] = dist * np.exp(b * t) * np.cos(2 * np.pi * t) + flames[flame_no, j]
                    
            moths[i] = np.clip(moths[i], lb, ub)
            fitness[i] = _safe_obj(objective, moths[i])
            evals += 1
            if evals >= max_evals: break
            
        if evals >= max_evals: break
        
        # Update flames
        combined_pop = np.vstack((flames, moths))
        combined_fit = np.concatenate((sorted_fit, fitness))
        
        sorted_idx = np.argsort(combined_fit)
        flames = combined_pop[sorted_idx[:pop_size]]
        sorted_fit = combined_fit[sorted_idx[:pop_size]]
        
        best_flame_score = sorted_fit[0]
        curve.append(best_flame_score)
        
    return {"best_fitness": best_flame_score, "evaluations_used": evals, "convergence_curve": curve, "best_solution": flames[0].copy()}

def run_pso(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    X = np.random.uniform(lb, ub, (pop_size, dim))
    V = np.zeros((pop_size, dim))
    P = X.copy()
    P_fit = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        P_fit[i] = _safe_obj(objective, X[i])
        
    evals = pop_size
    
    gbest_idx = np.argmin(P_fit)
    gbest = P[gbest_idx].copy()
    gbest_fit = P_fit[gbest_idx]
    
    curve = [gbest_fit]
    w_max = 0.9
    w_min = 0.4
    max_iter = max_evals // pop_size
    it = 0
    
    while evals < max_evals:
        w = w_max - it * ((w_max - w_min) / max_iter)
        
        for i in range(pop_size):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            V[i] = w * V[i] + 2.0 * r1 * (P[i] - X[i]) + 2.0 * r2 * (gbest - X[i])
            X[i] = X[i] + V[i]
            X[i] = np.clip(X[i], lb, ub)
            
            fit = _safe_obj(objective, X[i])
            evals += 1
            
            if fit < P_fit[i]:
                P_fit[i] = fit
                P[i] = X[i].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = X[i].copy()
            
            if evals >= max_evals: break
            
        curve.append(gbest_fit)
        it += 1
        
    return {"best_fitness": gbest_fit, "evaluations_used": evals, "convergence_curve": curve, "best_solution": gbest.copy()}

def run_woa(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    X = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        fitness[i] = _safe_obj(objective, X[i])
        
    evals = pop_size
    
    best_idx = np.argmin(fitness)
    leader_pos = X[best_idx].copy()
    leader_score = fitness[best_idx]
    
    curve = [leader_score]
    max_iter = max_evals // pop_size
    t = 0
    
    while evals < max_evals:
        a = 2 - t * (2 / max_iter)
        a2 = -1 + t * ((-1) / max_iter)
        
        for i in range(pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            
            A = 2 * a * r1 - a
            C = 2 * r2
            
            b = 1
            l = (a2 - 1) * np.random.rand() + 1
            p = np.random.rand()
            
            if p < 0.5:
                if abs(A) >= 1:
                    rand_idx = np.random.randint(0, pop_size)
                    X_rand = X[rand_idx]
                    D_X_rand = abs(C * X_rand - X[i])
                    X[i] = X_rand - A * D_X_rand
                else:
                    D_Leader = abs(C * leader_pos - X[i])
                    X[i] = leader_pos - A * D_Leader
            else:
                dist = abs(leader_pos - X[i])
                X[i] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + leader_pos
                
            X[i] = np.clip(X[i], lb, ub)
            
            fit = _safe_obj(objective, X[i])
            evals += 1
            if fit < leader_score:
                leader_score = fit
                leader_pos = X[i].copy()
                
            if evals >= max_evals: break
            
        curve.append(leader_score)
        t += 1
        
    return {"best_fitness": leader_score, "evaluations_used": evals, "convergence_curve": curve, "best_solution": leader_pos.copy()}

def run_do(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    # Dandelion Optimizer
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        fit[i] = _safe_obj(objective, pop[i])
        
    evals = pop_size
    best_idx = np.argmin(fit)
    best_pos = pop[best_idx].copy()
    best_score = fit[best_idx]
    
    curve = [best_score]
    max_iter = max_evals // pop_size
    t = 1
    
    while evals < max_evals:
        # Rising
        beta = np.random.randn(pop_size, dim)
        alpha = np.random.rand() * ((1/max_iter**2)*t**2 - 2/max_iter*t + 1)
        a_param = 1/(max_iter**2 - 2*max_iter + 1)
        b_param = -2*a_param
        c_param = 1 - a_param - b_param
        k = 1 - np.random.rand() * (c_param + a_param*t**2 + b_param*t)
        
        pop_new = pop.copy()
        
        if np.random.randn() < 1.5:
            for i in range(pop_size):
                lamb = abs(np.random.randn(dim))
                theta = (2*np.random.rand()-1)*np.pi
                row = 1/np.exp(theta)
                vx = row * np.cos(theta)
                vy = row * np.sin(theta)
                NEW = np.random.uniform(lb, ub, dim)
                # w = lognpdf approx
                w = np.exp(-0.5*lamb**2) / np.sqrt(2*np.pi) 
                pop_new[i] = pop[i] + alpha * vx * vy * w * (NEW - pop[i])
        else:
            for i in range(pop_size):
                pop_new[i] = pop[i] * k
                
        pop_new = np.clip(pop_new, lb, ub)
        
        # Decline
        mean_pos = np.mean(pop_new, axis=0)
        pop_new2 = np.zeros_like(pop_new)
        for i in range(pop_size):
            pop_new2[i] = pop_new[i] - beta[i]*alpha*(mean_pos - beta[i]*alpha*pop_new[i])
            
        pop_new2 = np.clip(pop_new2, lb, ub)
        
        # Landing
        step = _levy(pop_size, dim, 1.5)
        pop_new3 = np.zeros_like(pop_new2)
        for i in range(pop_size):
            pop_new3[i] = best_pos + step[i]*alpha*(best_pos - pop_new2[i]*(2*t/max_iter))
            
        pop_new3 = np.clip(pop_new3, lb, ub)
        
        # Eval
        for i in range(pop_size):
            f = _safe_obj(objective, pop_new3[i])
            evals += 1
            if f < fit[i]:
                fit[i] = f
                pop[i] = pop_new3[i]
                if f < best_score:
                    best_score = f
                    best_pos = pop[i].copy()
            if evals >= max_evals: break
            
        curve.append(best_score)
        t += 1
        
    return {"best_fitness": best_score, "evaluations_used": evals, "convergence_curve": curve, "best_solution": best_pos.copy()}

def run_gsa(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    X = np.random.uniform(lb, ub, (pop_size, dim))
    V = np.zeros((pop_size, dim))
    fit = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        fit[i] = _safe_obj(objective, X[i])
        
    evals = pop_size
    best_idx = np.argmin(fit)
    best_score = fit[best_idx]
    
    curve = [best_score]
    max_iter = max_evals // pop_size
    t = 0
    G0 = 100
    alpha = 20
    
    while evals < max_evals:
        t += 1
        G = G0 * np.exp(-alpha * t / max_iter)
        
        worst = np.max(fit)
        best = np.min(fit)
        
        m = (fit - worst) / (best - worst + 1e-15)
        M = m / np.sum(m)
        
        F = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for j in range(pop_size):
                if i != j:
                    R = np.linalg.norm(X[i] - X[j])
                    F[i] += np.random.rand() * G * M[j] * (X[j] - X[i]) / (R + 1e-5)
                    
        a = F 
        V = np.random.rand(pop_size, dim) * V + a
        X = X + V
        X = np.clip(X, lb, ub)
        
        for i in range(pop_size):
            fit[i] = _safe_obj(objective, X[i])
            evals += 1
            if fit[i] < best_score:
                best_score = fit[i]
            if evals >= max_evals: break
            
        curve.append(best_score)
        
    return {"best_fitness": best_score, "evaluations_used": evals, "convergence_curve": curve, "best_solution": X[np.argmin(fit)].copy()}

def run_goa(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    X = np.random.uniform(lb, ub, (pop_size, dim))
    fit = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        fit[i] = _safe_obj(objective, X[i])
        
    evals = pop_size
    best_idx = np.argmin(fit)
    target_pos = X[best_idx].copy()
    target_score = fit[best_idx]
    
    curve = [target_score]
    max_iter = max_evals // pop_size
    l = 0
    
    cMax = 1
    cMin = 0.00004
    
    while evals < max_evals:
        l += 1
        c = cMax - l * ((cMax - cMin) / max_iter)
        
        for i in range(pop_size):
            S_i = np.zeros(dim)
            for j in range(pop_size):
                if i != j:
                    dist = np.linalg.norm(X[j] - X[i])
                    r_ij = (X[j] - X[i]) / (dist + 1e-15)
                    
                    # Social forces (s function)
                    f = 0.5
                    l_scale = 1.5
                    s_val = f * np.exp(-dist / l_scale) - np.exp(-dist)
                    
                    S_i += c * ((ub - lb) / 2) * s_val * r_ij
                    
            X_new = c * S_i + target_pos
            X_new = np.clip(X_new, lb, ub)
            
            f_new = _safe_obj(objective, X_new)
            evals += 1
            X[i] = X_new
            fit[i] = f_new
            
            if f_new < target_score:
                target_score = f_new
                target_pos = X_new.copy()
                
            if evals >= max_evals: break
            
        curve.append(target_score)
        
    return {"best_fitness": target_score, "evaluations_used": evals, "convergence_curve": curve, "best_solution": target_pos.copy()}

def run_rime(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    Rimepop = np.random.uniform(lb, ub, (pop_size, dim))
    Rime_rates = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        Rime_rates[i] = _safe_obj(objective, Rimepop[i])
        
    evals = pop_size
    best_idx = np.argmin(Rime_rates)
    Best_rime = Rimepop[best_idx].copy()
    Best_rime_rate = Rime_rates[best_idx]
    
    curve = [Best_rime_rate]
    max_iter = max_evals // pop_size
    it = 0
    W = 5
    
    while evals < max_evals:
        it += 1
        
        # Normalize rates
        minR = np.min(Rime_rates)
        maxR = np.max(Rime_rates)
        norm_rates = (Rime_rates - minR) / (maxR - minR + 1e-15)
        
        RimeFactor = (np.random.rand() - 0.5) * 2 * np.cos((np.pi * it / (max_iter / 10))) * (1 - round(it * W / max_iter) / W)
        E = np.sqrt(it / max_iter)
        
        newRimepop = Rimepop.copy()
        
        for i in range(pop_size):
            for j in range(dim):
                if np.random.rand() < E:
                    newRimepop[i, j] = Best_rime[j] + RimeFactor * ((ub[j] - lb[j]) * np.random.rand() + lb[j])
                
                if np.random.rand() < norm_rates[i]:
                    newRimepop[i, j] = Best_rime[j]
                    
        newRimepop = np.clip(newRimepop, lb, ub)
        
        for i in range(pop_size):
            f_new = _safe_obj(objective, newRimepop[i])
            evals += 1
            if f_new < Rime_rates[i]:
                Rime_rates[i] = f_new
                Rimepop[i] = newRimepop[i]
                if f_new < Best_rime_rate:
                    Best_rime_rate = f_new
                    Best_rime = newRimepop[i].copy()
            if evals >= max_evals: break
            
        curve.append(Best_rime_rate)
        
    return {"best_fitness": Best_rime_rate, "evaluations_used": evals, "convergence_curve": curve, "best_solution": Best_rime.copy()}

def run_mpa(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    Prey = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.full(pop_size, np.inf)
    
    for i in range(pop_size):
        fitness[i] = _safe_obj(objective, Prey[i])
        
    evals = pop_size
    best_idx = np.argmin(fitness)
    Top_predator_pos = Prey[best_idx].copy()
    Top_predator_fit = fitness[best_idx]
    
    curve = [Top_predator_fit]
    max_iter = max_evals // pop_size
    Iter = 0
    FADs = 0.2
    P = 0.5
    
    while evals < max_evals:
        Iter += 1
        
        stepsize = np.zeros((pop_size, dim))
        RB = np.random.randn(pop_size, dim)
        RL = 0.05 * _levy(pop_size, dim, 1.5)
        
        CF = (1 - Iter / max_iter) ** (2 * Iter / max_iter)
        
        for i in range(pop_size):
            for j in range(dim):
                R = np.random.rand()
                if Iter < max_iter / 3:
                    stepsize[i, j] = RB[i, j] * (Top_predator_pos[j] - RB[i, j] * Prey[i, j])
                    Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]
                elif Iter > max_iter / 3 and Iter < 2 * max_iter / 3:
                    if i > pop_size / 2:
                        stepsize[i, j] = RB[i, j] * (RB[i, j] * Top_predator_pos[j] - Prey[i, j])
                        Prey[i, j] = Top_predator_pos[j] + P * CF * stepsize[i, j]
                    else:
                        stepsize[i, j] = RL[i, j] * (Top_predator_pos[j] - RL[i, j] * Prey[i, j])
                        Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]
                else:
                    stepsize[i, j] = RL[i, j] * (RL[i, j] * Top_predator_pos[j] - Prey[i, j])
                    Prey[i, j] = Top_predator_pos[j] + P * CF * stepsize[i, j]
                    
        Prey = np.clip(Prey, lb, ub)
        
        for i in range(pop_size):
            f_new = _safe_obj(objective, Prey[i])
            evals += 1
            if f_new < fitness[i]:
                fitness[i] = f_new
                if f_new < Top_predator_fit:
                    Top_predator_fit = f_new
                    Top_predator_pos = Prey[i].copy()
            if evals >= max_evals: break
            
        if evals >= max_evals: break
        
        # FADs
        if np.random.rand() < FADs:
            U = np.random.rand(pop_size, dim) < FADs
            Prey = Prey + CF * ((lb + np.random.rand(pop_size, dim) * (ub - lb)) * U)
        else:
            r = np.random.rand()
            Rs = pop_size
            stepsize = (FADs * (1 - r) + r) * (Prey[np.random.permutation(Rs)] - Prey[np.random.permutation(Rs)])
            Prey = Prey + stepsize
            
        Prey = np.clip(Prey, lb, ub)
        
        # Final eval check for FADs phase? 
        # Standard MPA does check after FADs.
        for i in range(pop_size):
            f_new = _safe_obj(objective, Prey[i])
            evals += 1
            if f_new < fitness[i]:
                fitness[i] = f_new
                if f_new < Top_predator_fit:
                    Top_predator_fit = f_new
                    Top_predator_pos = Prey[i].copy()
            if evals >= max_evals: break
            
        curve.append(Top_predator_fit)
        
    return {"best_fitness": Top_predator_fit, "evaluations_used": evals, "convergence_curve": curve, "best_solution": Top_predator_pos.copy()}

def run_de(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int, F: float = 0.5, CR: float = 0.9) -> Dict:
    dim = len(bounds)
    pop = np.array([[float(np.random.uniform(lo, hi)) for (lo, hi) in bounds] for _ in range(pop_size)])
    fit = np.array([float(_safe_obj(objective, x)) for x in pop])
    evals = pop_size
    curve = [float(np.min(fit))]
    while evals < max_evals:
        new_pop = []
        for i in range(pop_size):
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            trial = pop[i].copy()
            jrand = np.random.randint(0, dim)
            for j in range(dim):
                if np.random.rand() < CR or j == jrand:
                    trial[j] = mutant[j]
                lo, hi = bounds[j]
                trial[j] = float(np.clip(trial[j], lo, hi))
            new_pop.append(trial)
        
        next_pop = []
        next_fit = []
        for i in range(pop_size):
            trial = new_pop[i]
            f_trial = float(_safe_obj(objective, trial))
            evals += 1
            if evals >= max_evals: break
            if f_trial <= fit[i]:
                next_pop.append(trial)
                next_fit.append(f_trial)
            else:
                next_pop.append(pop[i])
                next_fit.append(fit[i])
        
        pop = np.array(next_pop) if next_pop else pop
        fit = np.array(next_fit) if next_fit else fit
        curve.append(float(np.min(fit)))
        
    best = float(np.min(fit))
    best_sol = pop[np.argmin(fit)].copy()
    return {"best_fitness": best, "evaluations_used": evals, "convergence_curve": curve, "best_solution": best_sol}

def run_ga(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([_safe_obj(objective, ind) for ind in pop])
    
    evals = pop_size
    best_idx = np.argmin(fitness)
    best_fit = fitness[best_idx]
    best_sol = pop[best_idx].copy()
    curve = [best_fit]
    
    # Parameters
    pc = 0.9  # Crossover probability
    pm = 1.0 / dim  # Mutation probability
    eta_c = 20.0  # SBX index
    eta_m = 20.0  # Polynomial mutation index
    
    while evals < max_evals:
        # Tournament Selection
        mating_pool = []
        for _ in range(pop_size):
            i1, i2 = np.random.randint(0, pop_size, 2)
            if fitness[i1] < fitness[i2]:
                mating_pool.append(pop[i1])
            else:
                mating_pool.append(pop[i2])
        mating_pool = np.array(mating_pool)
        
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = mating_pool[i].copy()
            p2 = mating_pool[(i + 1) % pop_size].copy()
            
            # SBX Crossover
            if np.random.rand() < pc:
                for j in range(dim):
                    if np.random.rand() <= 0.5:
                        if np.abs(p1[j] - p2[j]) > 1e-14:
                            y1 = min(p1[j], p2[j])
                            y2 = max(p1[j], p2[j])
                            rand = np.random.rand()
                            beta = 1.0 + 2.0 * (y1 - lb[j]) / (y2 - y1)
                            alpha = 2.0 - beta ** -(eta_c + 1.0)
                            if rand <= 1.0 / alpha:
                                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                            else:
                                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                            
                            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                            p1[j] = c1
                            p2[j] = c2
                        else:
                            pass
                    else:
                        pass
            
            # Polynomial Mutation
            for p in [p1, p2]:
                for j in range(dim):
                    if np.random.rand() < pm:
                        y = p[j]
                        yl = lb[j]
                        yu = ub[j]
                        delta1 = (y - yl) / (yu - yl)
                        delta2 = (yu - y) / (yu - yl)
                        rand = np.random.rand()
                        mut_pow = 1.0 / (eta_m + 1.0)
                        if rand <= 0.5:
                            val = 2.0 * rand + (1.0 - 2.0 * rand) * (1.0 - delta1) ** (eta_m + 1.0)
                            deltaq = val ** mut_pow - 1.0
                        else:
                            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (1.0 - delta2) ** (eta_m + 1.0)
                            deltaq = 1.0 - val ** mut_pow
                        p[j] = y + deltaq * (yu - yl)
            
            offspring.append(p1)
            offspring.append(p2)
            
        # Evaluation
        new_pop = np.array(offspring[:pop_size])
        new_pop = np.clip(new_pop, lb, ub)
        new_fitness = np.array([_safe_obj(objective, ind) for ind in new_pop])
        evals += pop_size
        
        # Elitism: Replace parent if offspring is better (Generational + Elitism usually)
        # Here we use simple generational replacement but keep best
        
        # Combined population
        combined_pop = np.vstack((pop, new_pop))
        combined_fit = np.concatenate((fitness, new_fitness))
        
        sorted_idx = np.argsort(combined_fit)
        pop = combined_pop[sorted_idx[:pop_size]]
        fitness = combined_fit[sorted_idx[:pop_size]]
        
        if fitness[0] < best_fit:
            best_fit = fitness[0]
            best_sol = pop[0].copy()
            
        curve.append(best_fit)
        
    return {"best_fitness": best_fit, "evaluations_used": evals, "convergence_curve": curve, "best_solution": best_sol}

def run_abc(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Food sources (pop_size / 2)
    n_food = max(10, pop_size // 2)
    limit = n_food * dim # Abandonment limit
    
    foods = np.random.uniform(lb, ub, (n_food, dim))
    fitness = np.array([_safe_obj(objective, ind) for ind in foods])
    trial_counters = np.zeros(n_food)
    
    best_idx = np.argmin(fitness)
    best_fit = fitness[best_idx]
    best_sol = foods[best_idx].copy()
    
    evals = n_food
    curve = [best_fit]
    
    while evals < max_evals:
        # Employed Bees
        for i in range(n_food):
            # Select neighbor
            k = np.random.randint(0, n_food)
            while k == i: k = np.random.randint(0, n_food)
            j = np.random.randint(0, dim)
            
            phi = np.random.uniform(-1, 1)
            v = foods[i].copy()
            v[j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
            v = np.clip(v, lb, ub)
            
            f_new = _safe_obj(objective, v)
            evals += 1
            if f_new < fitness[i]:
                foods[i] = v
                fitness[i] = f_new
                trial_counters[i] = 0
            else:
                trial_counters[i] += 1
            if evals >= max_evals: break
            
        if evals >= max_evals: break
        
        # Onlooker Bees
        # Calculate probabilities
        fit_inv = np.zeros(n_food)
        for i in range(n_food):
            if fitness[i] >= 0:
                fit_inv[i] = 1.0 / (1.0 + fitness[i])
            else:
                fit_inv[i] = 1.0 + np.abs(fitness[i])
        prob = fit_inv / np.sum(fit_inv)
        
        t = 0
        i = 0
        while t < n_food:
            if np.random.rand() < prob[i]:
                t += 1
                # Select neighbor
                k = np.random.randint(0, n_food)
                while k == i: k = np.random.randint(0, n_food)
                j = np.random.randint(0, dim)
                
                phi = np.random.uniform(-1, 1)
                v = foods[i].copy()
                v[j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
                v = np.clip(v, lb, ub)
                
                f_new = _safe_obj(objective, v)
                evals += 1
                if f_new < fitness[i]:
                    foods[i] = v
                    fitness[i] = f_new
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1
            i = (i + 1) % n_food
            if evals >= max_evals: break
            
        if evals >= max_evals: break
        
        # Scout Bees
        max_trial_idx = np.argmax(trial_counters)
        if trial_counters[max_trial_idx] > limit:
            foods[max_trial_idx] = np.random.uniform(lb, ub, dim)
            fitness[max_trial_idx] = _safe_obj(objective, foods[max_trial_idx])
            evals += 1
            trial_counters[max_trial_idx] = 0
            
        # Update best
        curr_best_idx = np.argmin(fitness)
        if fitness[curr_best_idx] < best_fit:
            best_fit = fitness[curr_best_idx]
            best_sol = foods[curr_best_idx].copy()
            
        curve.append(best_fit)
        
    return {"best_fitness": best_fit, "evaluations_used": evals, "convergence_curve": curve, "best_solution": best_sol}

def run_shade(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # SHADE Parameters
    H = 100 # Memory size
    M_CR = np.ones(H) * 0.5
    M_F = np.ones(H) * 0.5
    k = 0 # Memory index
    
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([_safe_obj(objective, ind) for ind in pop])
    evals = pop_size
    
    # Archive
    archive = []
    
    best_idx = np.argmin(fitness)
    best_fit = fitness[best_idx]
    best_sol = pop[best_idx].copy()
    curve = [best_fit]
    
    while evals < max_evals:
        S_CR = []
        S_F = []
        S_diff = []
        
        new_pop = np.zeros_like(pop)
        new_fitness = np.zeros(pop_size)
        
        # Sort population for current-to-pbest
        sorted_idx = np.argsort(fitness)
        
        for i in range(pop_size):
            # Generation of CR and F
            r = np.random.randint(0, H)
            mu_cr = M_CR[r]
            mu_f = M_F[r]
            
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            
            f = np.random.standard_cauchy() * 0.1 + mu_f
            while f <= 0:
                f = np.random.standard_cauchy() * 0.1 + mu_f
            if f > 1: f = 1
            
            # Current-to-pbest/1
            p_best_rate = np.random.uniform(2.0/pop_size, 0.2)
            p_best_idx = sorted_idx[np.random.randint(0, int(pop_size * p_best_rate))]
            x_pbest = pop[p_best_idx]
            
            r1 = np.random.randint(0, pop_size)
            while r1 == i: r1 = np.random.randint(0, pop_size)
            x_r1 = pop[r1]
            
            # Union of pop and archive
            pop_archive = np.vstack((pop, np.array(archive))) if len(archive) > 0 else pop
            r2 = np.random.randint(0, len(pop_archive))
            while r2 == i or r2 == r1: r2 = np.random.randint(0, len(pop_archive))
            x_r2 = pop_archive[r2]
            
            mutant = pop[i] + f * (x_pbest - pop[i]) + f * (x_r1 - x_r2)
            
            # Binomial Crossover
            jrand = np.random.randint(0, dim)
            trial = np.copy(pop[i])
            for j in range(dim):
                if np.random.rand() < cr or j == jrand:
                    trial[j] = mutant[j]
            
            trial = np.clip(trial, lb, ub)
            f_trial = _safe_obj(objective, trial)
            evals += 1
            
            new_pop[i] = trial
            new_fitness[i] = f_trial
            
            if f_trial < fitness[i]:
                S_CR.append(cr)
                S_F.append(f)
                S_diff.append(np.abs(f_trial - fitness[i]))
                
                # Add to archive
                archive.append(pop[i].copy())
                if len(archive) > pop_size:
                    del_idx = np.random.randint(0, len(archive))
                    del archive[del_idx]
            else:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]
                
            if f_trial < best_fit:
                best_fit = f_trial
                best_sol = trial.copy()
                
            if evals >= max_evals: break
            
        pop = new_pop
        fitness = new_fitness
        
        # Update Memory
        if len(S_CR) > 0:
            w = S_diff / np.sum(S_diff)
            m_cr = np.sum(w * S_CR)
            m_f = np.sum(w * (np.array(S_F)**2)) / np.sum(w * S_F)
            
            M_CR[k] = m_cr
            M_F[k] = m_f
            k = (k + 1) % H
            
        curve.append(best_fit)
        
    return {"best_fitness": best_fit, "evaluations_used": evals, "convergence_curve": curve, "best_solution": best_sol}

def run_lshade(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    return _run_pyade_algorithm("pyade.lshade", objective, bounds, pop_size, max_evals)

def run_lshade_epsin(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    # pyade exposes the closest public Python implementation as L-SHADE-cnEpSin.
    return _run_pyade_algorithm("pyade.lshadecnepsin", objective, bounds, pop_size, max_evals)

def run_jso(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    return _run_pyade_algorithm("pyade.jso", objective, bounds, pop_size, max_evals)

def run_cmaes(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    import cma

    dim = len(bounds)
    lb, ub, _ = _bounds_arrays(bounds)
    x0 = ((lb + ub) / 2.0).astype(float)
    sigma0 = float(max(np.mean(ub - lb) / 3.0, 1e-8))
    options = {
        "bounds": [lb.tolist(), ub.tolist()],
        "popsize": max(4, int(pop_size)),
        "maxfevals": max(dim + 1, int(max_evals)),
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
    }
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, options)
    evals = 0
    best_fitness = np.inf
    best_solution = x0.copy()
    curve: List[float] = []

    while not es.stop() and evals < max_evals:
        solutions = es.ask()
        fitnesses = []
        for sol in solutions:
            arr = np.asarray(sol, dtype=float)
            fit = _safe_obj(objective, arr)
            fitnesses.append(fit)
            evals += 1
            if fit < best_fitness:
                best_fitness = fit
                best_solution = arr.copy()
        es.tell(solutions, fitnesses)
        curve.append(float(best_fitness))

    if not curve:
        curve = [float(best_fitness)]
    return {
        "best_fitness": float(best_fitness),
        "evaluations_used": int(evals),
        "convergence_curve": curve,
        "best_solution": best_solution.copy(),
    }

def run_cpo(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    from porcupy.cpo import cpo

    dim = len(bounds)
    lb, ub, _ = _bounds_arrays(bounds)
    max_iter = max(1, int(math.ceil(max_evals / max(pop_size, 1))))
    best_solution, best_fitness, curve = cpo(
        lambda x: _safe_obj(objective, np.asarray(x, dtype=float)),
        lb.tolist(),
        ub.tolist(),
        pop_size=max(10, int(pop_size)),
        max_iter=max_iter,
        verbose=False,
    )
    return {
        "best_fitness": float(best_fitness),
        "evaluations_used": int(max(dim, pop_size) * max_iter),
        "convergence_curve": np.asarray(curve, dtype=float).tolist(),
        "best_solution": np.asarray(best_solution, dtype=float).copy(),
    }

def run_aoo(objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    # Local compatibility implementation aligned to the public Animated Oat paper description:
    # global seed dispersal, rolling propagation, and obstacle-triggered propulsion.
    dim = len(bounds)
    lb, ub, _ = _bounds_arrays(bounds)
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([_safe_obj(objective, ind) for ind in pop], dtype=float)
    evals = pop_size

    best_idx = int(np.argmin(fitness))
    best_fit = float(fitness[best_idx])
    best_sol = pop[best_idx].copy()
    curve = [best_fit]
    max_iter = max(1, int(math.ceil(max_evals / max(pop_size, 1))))
    t = 0

    while evals < max_evals:
        progress = t / max(max_iter - 1, 1)
        mean_pos = np.mean(pop, axis=0)
        span = ub - lb

        for i in range(pop_size):
            r = np.random.rand()
            peer = pop[np.random.randint(0, pop_size)]
            if r < 1.0 / 3.0:
                wind = np.random.randn(dim) * span * (0.25 * (1.0 - progress))
                cand = pop[i] + wind + np.random.rand(dim) * (peer - pop[i])
            elif r < 2.0 / 3.0:
                roll = np.sin(2 * np.pi * np.random.rand(dim))
                cand = pop[i] + np.random.rand(dim) * (best_sol - pop[i]) + 0.5 * roll * (peer - mean_pos)
            else:
                propulsion = (0.1 + 0.4 * (1.0 - progress)) * np.random.randn(dim)
                cand = best_sol + propulsion * (best_sol - pop[i] + 1e-12)

            cand = np.clip(cand, lb, ub)
            fit = _safe_obj(objective, cand)
            evals += 1

            if fit < fitness[i]:
                pop[i] = cand
                fitness[i] = fit
                if fit < best_fit:
                    best_fit = float(fit)
                    best_sol = cand.copy()

            if evals >= max_evals:
                break

        curve.append(best_fit)
        t += 1

    return {
        "best_fitness": best_fit,
        "evaluations_used": evals,
        "convergence_curve": curve,
        "best_solution": best_sol.copy(),
    }

def run_baseline(name: str, objective: Callable, bounds: List[Tuple[float, float]], pop_size: int, max_evals: int) -> Dict:
    if name == "ALO": return run_alo(objective, bounds, pop_size, max_evals)
    if name == "BA": return run_ba(objective, bounds, pop_size, max_evals)
    if name == "GWO": return run_gwo(objective, bounds, pop_size, max_evals)
    if name == "HHO": return run_hho(objective, bounds, pop_size, max_evals)
    if name == "MFO": return run_mfo(objective, bounds, pop_size, max_evals)
    if name == "PSO": return run_pso(objective, bounds, pop_size, max_evals)
    if name == "WOA": return run_woa(objective, bounds, pop_size, max_evals)
    if name == "DO": return run_do(objective, bounds, pop_size, max_evals)
    if name == "GSA": return run_gsa(objective, bounds, pop_size, max_evals)
    if name == "GOA": return run_goa(objective, bounds, pop_size, max_evals)
    if name == "RIME": return run_rime(objective, bounds, pop_size, max_evals)
    if name == "MPA": return run_mpa(objective, bounds, pop_size, max_evals)
    if name == "DE": return run_de(objective, bounds, pop_size, max_evals)
    if name == "GA": return run_ga(objective, bounds, pop_size, max_evals)
    if name == "ABC": return run_abc(objective, bounds, pop_size, max_evals)
    if name == "SHADE": return run_shade(objective, bounds, pop_size, max_evals)
    if name == "L-SHADE": return run_lshade(objective, bounds, pop_size, max_evals)
    if name == "L-SHADE-EpSin": return run_lshade_epsin(objective, bounds, pop_size, max_evals)
    if name == "jSO": return run_jso(objective, bounds, pop_size, max_evals)
    if name == "CMA-ES": return run_cmaes(objective, bounds, pop_size, max_evals)
    if name == "CPO": return run_cpo(objective, bounds, pop_size, max_evals)
    if name == "AOO": return run_aoo(objective, bounds, pop_size, max_evals)
    return run_de(objective, bounds, pop_size, max_evals)
