import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from typing import Callable, List, Tuple, Dict
import warnings

# Suppress GP warnings for convergence
warnings.filterwarnings("ignore")

class SAEA_DE:
    """
    A simple Surrogate-Assisted Differential Evolution (SAEA-DE) using Gaussian Process.
    This serves as a strong baseline for expensive optimization.
    Strategy:
    1. Initialize population.
    2. Train GP on all evaluated history.
    3. In each generation:
       - Generate offsprings using DE logic.
       - Predict fitness of offsprings using GP (Cheap).
       - Select 'n_infill' best offsprings (by predicted fitness or LCB) to evaluate on real function.
       - Update archive and retrain GP.
    """
    def __init__(self, objective: Callable, bounds: List[Tuple[float, float]], 
                 pop_size: int = 30, max_evals: int = 500, n_infill: int = 1):
        self.objective = objective
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.n_infill = n_infill # Number of points to evaluate per generation
        
        # Archive
        self.X_obs = []
        self.y_obs = []
        
        # Best found
        self.best_x = None
        self.best_y = np.inf
        
        # History
        self.convergence_curve = []
        
        # GP Model
        # Matern kernel is robust for optimization landscapes
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
        
    def _clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
    
    def optimize(self) -> Dict:
        # 1. Initialization
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.pop_size, self.dim))
        
        # Evaluate initial population
        for i in range(self.pop_size):
            if len(self.y_obs) >= self.max_evals: break
            y = self.objective(pop[i])
            self.X_obs.append(pop[i])
            self.y_obs.append(y)
            if y < self.best_y:
                self.best_y = y
                self.best_x = pop[i].copy()
            self.convergence_curve.append(self.best_y)
            
        if len(self.y_obs) >= self.max_evals:
            return self._get_result()
            
        # Main Loop
        # DE Parameters
        F = 0.5
        CR = 0.9
        
        while len(self.y_obs) < self.max_evals:
            # Train GP
            # Limit training size to keep speed up (sliding window of last 300 points if too large)
            # But for expensive opt, we usually use all data up to ~500-1000.
            train_X = np.array(self.X_obs)
            train_y = np.array(self.y_obs)
            
            # Simple fallback if variance is 0
            if np.std(train_y) < 1e-9:
                # Random search if flat
                next_x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                y = self.objective(next_x)
                self.X_obs.append(next_x)
                self.y_obs.append(y)
                if y < self.best_y:
                    self.best_y = y
                self.convergence_curve.append(self.best_y)
                continue
                
            try:
                self.gp.fit(train_X, train_y)
            except Exception:
                # Fallback if GP fails
                pass

            # Generate Candidates using DE logic (Prescreening)
            # We generate more candidates than needed and pick the best based on GP LCB
            n_candidates = self.pop_size * 5 
            candidates = []
            
            # Current population (from archive, take top pop_size)
            # Sort archive to get current population
            sorted_idx = np.argsort(self.y_obs)
            current_pop = np.array(self.X_obs)[sorted_idx[:self.pop_size]]
            
            for _ in range(n_candidates):
                # Random indices
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = current_pop[idxs]
                
                # Mutation
                mutant = a + F * (b - c)
                
                # Crossover
                trial = a.copy() # Use 'a' as target vector roughly
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == jrand:
                        trial[j] = mutant[j]
                
                trial = self._clip(trial)
                candidates.append(trial)
            
            candidates = np.array(candidates)
            
            # Predict with GP (LCB - Lower Confidence Bound for minimization)
            # LCB = mu - kappa * sigma
            # kappa = 2.0 (exploration-exploitation balance)
            mu, sigma = self.gp.predict(candidates, return_std=True)
            lcb = mu - 2.0 * sigma
            
            # Select best 'n_infill' candidates
            best_idx = np.argsort(lcb)[:self.n_infill]
            infill_points = candidates[best_idx]
            
            # Evaluate real function
            for x in infill_points:
                if len(self.y_obs) >= self.max_evals: break
                y = self.objective(x)
                self.X_obs.append(x)
                self.y_obs.append(y)
                if y < self.best_y:
                    self.best_y = y
                    self.best_x = x.copy()
                self.convergence_curve.append(self.best_y)
                
        return self._get_result()

    def _get_result(self):
        return {
            "best_fitness": self.best_y,
            "evaluations_used": len(self.y_obs),
            "convergence_curve": self.convergence_curve
        }

def run_saea(objective, bounds, pop_size, max_evals):
    # Wrapper for compatibility
    optimizer = SAEA_DE(objective, bounds, pop_size, max_evals, n_infill=1)
    return optimizer.optimize()
