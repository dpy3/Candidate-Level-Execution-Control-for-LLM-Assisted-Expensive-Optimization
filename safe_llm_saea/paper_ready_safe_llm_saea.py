import os
import json
import hashlib
import sys
import time
import math
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from types import SimpleNamespace

if sys.version_info >= (3, 13):
    try:
        import cgi  # type: ignore
    except ModuleNotFoundError:
        import types
        from email.message import Message

        def _parse_header(value: str):
            message = Message()
            message["content-type"] = value
            params = {key: val for key, val in message.get_params()[1:]}
            return message.get_content_type(), params

        cgi = types.ModuleType("cgi")
        cgi.parse_header = _parse_header
        sys.modules["cgi"] = cgi

from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Optional
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

from safe_llm_saea.csa_protocol import (
    AuthorizationError,
    RuntimeAdmissionGate,
    candidate_id,
    make_trace_event,
)
from safe_llm_saea.calibrated_regret import (
    apply_regret_linear_model,
    regret_feature_matrix,
)
from safe_llm_saea.backbone_risk import (
    apply_backbone_risk_model,
    backbone_risk_feature_matrix,
)

# Silence GP convergence warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import requests
except Exception:
    requests = None

try:
    from safe_llm_saea.api_key_manager import get_key
except ImportError:
    def get_key(idx):
        return os.getenv("DEEPSEEK_API_KEY", "")

@dataclass
class ExperimentConfig:
    algorithm_name: str = "L-ADE-Sniper"
    problem_name: str = "CEC2022"
    dimension: int = 10
    max_evaluations: int = 50000
    population_size: int = 30
    runs: int = 1
    llm_enhancement: bool = True
    gating_enabled: bool = True
    max_llm_calls_per_run: int = 10
    max_llm_cost_usd_per_run: float = 0.05
    fusion_weight: float = 0.86
    de_F: float = 0.42
    de_CR: float = 0.94
    min_remaining_budget: int = 300
    min_stagnation_generations: int = 5
    min_uncertainty_threshold: float = 0.1
    min_call_interval: int = 5
    min_confidence: float = 0.1
    enable_stage2: bool = True
    stage2_stagnation_generations: int = 8
    stage2_eval_ratio_threshold: float = 0.6
    stage2_fusion_weight_boost: float = 0.05
    stage2_call_interval_factor: float = 0.8
    stage2_F_factor: float = 1.1
    stage2_CR_factor: float = 0.95
    consistency_calls: int = 1
    consistency_threshold: float = 0.1
    schema_strict: bool = False
    mark_call_on_failure: bool = True
    force_initial_llm_call: bool = True
    max_llm_calls_stage1: int = 3
    max_llm_calls_stage2: int = 7
    llm_suggestions_k: int = 3
    diversity_weight: float = 0.15
    problem_description: str = "a black-box function"
    use_surrogate: bool = True
    enable_sniper: bool = True
    force_llm_every_gen: bool = False
    enable_micro_refine: bool = False
    micro_refine_interval: int = 5
    micro_refine_steps: int = 6
    micro_refine_sigma: float = 0.03
    precision_progress_threshold: float = 0.85
    llm_stagnation_trigger: int = 5
    sniper_interval: int = 5
    precision_target_f: float = 0.1
    precision_target_cr: float = 0.95
    weak_admission_control: bool = False
    protect_runtime_llm: bool = False
    protect_runtime_sniper: bool = False
    protect_runtime_surrogate: bool = False
    proposal_trace_mode: str = "capture"
    proposal_replay_path: str = ""
    proposal_replay_strict: bool = True
    enforce_full_evaluation_budget: bool = False
    surrogate_min_history: int = 0
    surrogate_acceptance_margin: float = 0.0
    surrogate_admission_policy: str = "improvement_threshold"
    surrogate_unavailable_policy: str = "reject"
    regret_uncertainty_scale: float = 1.0
    regret_risk_multiplier: float = 1.645
    regret_abstention_threshold: float = 0.0
    regret_model_feature_mean: Tuple[float, ...] = ()
    regret_model_feature_scale: Tuple[float, ...] = ()
    regret_model_coefficients: Tuple[float, ...] = ()
    regret_model_residual_radius: float = 0.0
    semantic_commit_policy: str = "population_improvement"
    risk_model_feature_mean: Tuple[float, ...] = ()
    risk_model_feature_scale: Tuple[float, ...] = ()
    risk_model_coefficients: Tuple[float, ...] = ()
    risk_model_intercept: float = 0.0
    risk_probability_threshold: float = 0.5
    allow_llm_control_updates: bool = True
    semantic_projection_alpha: float = 0.0

# Strategy Constants
STRATEGY_EXPLORATION = "exploration"   # High F, Rand/1
STRATEGY_BALANCED = "balanced"         # SHADE-like
STRATEGY_PRECISION = "precision"       # L-BFGS-B / Small-step DE
STRATEGY_EMERGENCY = "emergency"       # Restart / High Variance


def _clean_pool_value(value: str) -> str:
    return str(value or "").strip().strip("`").strip().strip("\"'").strip()


def _read_pool_env(*env_names: str) -> List[str]:
    values: List[str] = []
    seen = set()
    for env_name in env_names:
        raw = os.getenv(env_name, "")
        if not raw:
            continue
        normalized = raw.replace(";", "\n").replace(",", "\n")
        for item in normalized.splitlines():
            value = _clean_pool_value(item)
            if value and value not in seen:
                values.append(value)
                seen.add(value)
    return values


def _build_endpoint_pool(primary_api_key: str, primary_base_url: str) -> List[Dict[str, Any]]:
    key_pool = []
    if primary_api_key:
        key_pool.append(primary_api_key)
    key_pool.extend(
        value
        for value in _read_pool_env("SAFE_LLM_SAEA_API_KEY_POOL", "GPT_AGENT_API_KEY_POOL")
        if value != primary_api_key
    )

    base_url_pool = []
    if primary_base_url:
        base_url_pool.append(primary_base_url)
    base_url_pool.extend(
        value
        for value in _read_pool_env("SAFE_LLM_SAEA_BASE_URL_POOL", "GPT_AGENT_BASE_URL_POOL")
        if value != primary_base_url
    )

    if not key_pool or not base_url_pool:
        return []

    endpoint_count = max(len(key_pool), len(base_url_pool))
    endpoints: List[Dict[str, Any]] = []
    for index in range(endpoint_count):
        endpoints.append(
            {
                "index": index,
                "api_key": key_pool[index % len(key_pool)],
                "base_url": base_url_pool[index % len(base_url_pool)],
            }
        )
    return endpoints

class LLMAssistedDE:
    def __init__(self, config: ExperimentConfig, api_key: str = "", base_url: str = "https://api.deepseek.com", use_api: bool = False, llm_model: str = "deepseek-chat"):
        self.config = config
        if self.config.surrogate_unavailable_policy not in {"response_order", "reject"}:
            raise ValueError(
                "surrogate_unavailable_policy must be 'response_order' or 'reject'"
            )
        if self.config.surrogate_admission_policy not in {
            "improvement_threshold", "rank_top", "mean_top", "response_order",
            "calibrated_regret", "backbone_risk",
        }:
            raise ValueError(
                "surrogate_admission_policy must be improvement_threshold, rank_top, "
                "mean_top, response_order, calibrated_regret, or backbone_risk"
            )
        if self.config.regret_uncertainty_scale <= 0:
            raise ValueError("regret_uncertainty_scale must be positive")
        if self.config.regret_risk_multiplier < 0:
            raise ValueError("regret_risk_multiplier must be non-negative")
        if self.config.semantic_commit_policy not in {
            "population_improvement", "incumbent_improvement",
        }:
            raise ValueError(
                "semantic_commit_policy must be population_improvement or "
                "incumbent_improvement"
            )
        if not 0.0 <= float(self.config.semantic_projection_alpha) <= 1.0:
            raise ValueError("semantic_projection_alpha must be in [0, 1]")
        primary_api_key = (
            _clean_pool_value(api_key)
            or os.getenv("ds_api", "").strip()
            or os.getenv("DEEPSEEK_API_KEY", "").strip()
            or os.getenv("IFLOW_API_KEY", "").strip()
            or os.getenv("GPT_AGENT_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        if primary_api_key:
            print(f"[LLM-Assisted DE] Initialized. Using Key ending in ...{primary_api_key[-4:]}")
        else:
            print("[LLM-Assisted DE] Initialized without API key. API mode will be disabled if requested.")

        primary_base_url = (
            _clean_pool_value(base_url)
            or os.getenv("SAFE_LLM_SAEA_BASE_URL", "").strip()
            or os.getenv("DEEPSEEK_BASE_URL", "").strip()
            or "https://api.deepseek.com"
        )
        self.api_endpoints = _build_endpoint_pool(primary_api_key, primary_base_url)
        self.active_endpoint_index = 0
        self.endpoint_switch_count = 0
        self.api_key = primary_api_key
        self.base_url = primary_base_url
        self.use_api = bool(use_api)
        self.llm_model = (
            (llm_model or "").strip()
            or os.getenv("SAFE_LLM_SAEA_LLM_MODEL", "").strip()
            or os.getenv("DEEPSEEK_MODEL", "").strip()
            or "deepseek-chat"
        )
        self.api_client_mode = "disabled"
        if self.use_api and not self.api_endpoints:
            self.use_api = False
        
        self.client = None
        if self.use_api:
            self._activate_endpoint(0, reason="initialization")
        self.population = None
        self.fitness = None
        self.best_idx = None
        self.best_fit = None
        self.evaluations = 0
        self.convergence = []
        self.best_solution_history = []
        self.last_call_gen = -1
        self.stage = 1
        self.stage1_calls = 0
        self.stage2_calls = 0
        self.api_success_calls = 0
        self.total_cost_usd = 0.0
        self.total_tokens = 0
        self.llm_usage_details = []  # Detailed token tracking
        self.llm_call_points = []
        self.proposal_trace_mode = str(self.config.proposal_trace_mode or "capture").strip().lower()
        self.proposal_replay_path = str(self.config.proposal_replay_path or "").strip()
        self.proposal_replay_strict = bool(self.config.proposal_replay_strict)
        self.proposal_replay_events: List[Dict[str, Any]] = []
        self.proposal_replay_cursor = 0
        self.proposal_replay_source_sha256 = ""
        if self.proposal_trace_mode == "replay":
            self._load_proposal_replay()
        
        # LPSR Parameters
        self.pop_size_initial = config.population_size
        self.pop_size_min = 4
        
        # LLM Memory
        self.successful_llm_strategies = []
        self.last_llm_params = None

        # Field Commander State
        self.current_F_base = float(self.config.de_F)
        self.current_CR_base = float(self.config.de_CR)
        self.commander_mode = STRATEGY_BALANCED # Start balanced
        self.last_command_evals = 0
        self.stagnation_counter = 0
        self.no_improve_generations = 0
        self.last_generation_best = np.inf
        self.effective_min_call_interval = int(self.config.min_call_interval)
        self.dynamic_interval_max = max(3, int(self.config.min_call_interval) * 4)
        self.sniper_patience_generations = max(6, int(self.config.sniper_interval) * 2)
        self.fusion_weight_current = float(self.config.fusion_weight)
        self.low_dim_budget_mode = bool(
            self.config.dimension <= 4
            and self.config.max_evaluations <= max(96, int(self.config.population_size) * 8)
        )
        self.llm_recent_gains: List[float] = []
        self.low_gain_streak = 0
        self.high_gain_streak = 0
        self.llm_trust_score = max(0.2, min(0.95, float(self.config.min_confidence)))
        self.llm_gain_ema = 0.0
        self.call_failure_streak = 0
        self.last_call_progress_bucket = -1
        self.last_population_diversity = 0.0
        self.diversity_history: List[float] = []
        self.last_uncertainty = 0.0
        self.landscape_risk_score = 0.0
        self.rugged_mode = False
        self.last_precision_refine_eval = -10**9
        self.last_emergency_eval = -10**9
        self.llm_cooldown_until_eval = 0
        self.runtime_llm_enabled = bool(self.config.llm_enhancement) and (not self.low_dim_budget_mode)
        self.runtime_sniper_enabled = bool(self.config.enable_sniper)
        self.runtime_surrogate_enabled = bool(self.config.use_surrogate)
        self.generation_diagnostics: List[Dict[str, Any]] = []
        self.first_llm_disable_generation: Optional[int] = None
        self.first_sniper_disable_generation: Optional[int] = None
        self.first_surrogate_disable_generation: Optional[int] = None
        self.admission_events: List[Dict[str, Any]] = []
        self.pending_directional_delta: Optional[np.ndarray] = None
        self.pending_directional_alpha = 0.0
        self.pending_directional_diversity_floor = 1.0
        self.directional_injection_evaluator: Optional[Callable] = None
        self.last_directional_injection: Dict[str, Any] = {}
        self._pending_admission_event: Optional[Dict[str, Any]] = None
        
        # SHADE Parameters
        self.H = 100
        self.M_CR = np.ones(self.H) * 0.5
        self.M_F = np.ones(self.H) * 0.5
        self.k = 0
        self.archive = []
        
        # GP Surrogate
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
        self.X_history = []
        self.y_history = []
        self.gp_trained = False

    def optimize(self, objective_function: Callable, bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
        self.population = None
        self.bounds = bounds
        dim = len(bounds)
        pop_size = self.config.population_size
        
        # Initialize Population
        self.population = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (pop_size, dim)
        )
        self.fitness = np.full(pop_size, np.inf)
        
        # --- DOMINANCE GUARD: Math-Only Subpopulation ---
        # Dedicate the last 15% of the population strictly to pure SHADE/DE math optimization.
        # This acts as a true structural "Dominance Guard". If LLM explores poorly and ruins F/CR
        # for the main population, this subpopulation ensures we at least match standard mathematical convergence.
        self.math_only_size = max(4, int(pop_size * 0.15))
        self.math_only_indices = list(range(pop_size - self.math_only_size, pop_size))
        
        start = time.time()
        max_gen = max(1, self.config.max_evaluations // pop_size)
        max_generation_guard = max_gen * 10
        gen = 0
        
        while self.evaluations < self.config.max_evaluations:
            # 1. Evaluate initial population if needed
            for i in range(len(self.population)):
                if self.fitness[i] == np.inf:
                    self.fitness[i] = float(objective_function(self.population[i]))
                    self.evaluations += 1
                    self.X_history.append(self.population[i].copy())
                    self.y_history.append(self.fitness[i])
                    if self.evaluations >= self.config.max_evaluations: break
            
            self.best_idx = int(np.argmin(self.fitness))
            self.best_fit = float(self.fitness[self.best_idx])
            self.convergence.append(self.best_fit)
            self.best_solution_history.append(self.population[self.best_idx].copy().tolist())
            if self.best_fit + 1e-12 < self.last_generation_best:
                self.no_improve_generations = 0
                self.last_generation_best = self.best_fit
            else:
                self.no_improve_generations += 1
            
            progress = self.evaluations / self.config.max_evaluations
            self.last_population_diversity = self._estimate_population_diversity(bounds)
            self.diversity_history.append(float(self.last_population_diversity))
            if len(self.diversity_history) > 20:
                self.diversity_history = self.diversity_history[-20:]
            if len(self.convergence) > 10 and abs(self.convergence[-1] - self.convergence[-10]) < 1e-9:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            
            if self.config.enable_stage2 and self.stage == 1:
                if self.stagnation_counter >= self.config.stage2_stagnation_generations or progress >= self.config.stage2_eval_ratio_threshold:
                    self.stage = 2
                    self.current_F_base = min(1.0, self.current_F_base * self.config.stage2_F_factor)
                    self.current_CR_base = min(1.0, self.current_CR_base * self.config.stage2_CR_factor)
                    self.effective_min_call_interval = max(1, int(self.config.min_call_interval * self.config.stage2_call_interval_factor))
                    self.fusion_weight_current = min(0.99, self.config.fusion_weight + self.config.stage2_fusion_weight_boost)

            if gen % 100 == 0:
                print(f"[Optimizer] Gen {gen}/{max_gen} | Evals {self.evaluations}/{self.config.max_evaluations} | Best: {self.best_fit:.4e}", flush=True)

            # Train GP periodically
            if self.runtime_surrogate_enabled and len(self.X_history) > 0 and (gen % 5 == 0): # Less frequent training to save time
                try:
                    # Train on recent history (sliding window) to keep it fast
                    window = 500
                    X_train = np.array(self.X_history[-window:])
                    y_train = np.array(self.y_history[-window:])
                    self.gp.fit(X_train, y_train)
                    self.gp_trained = True
                except Exception:
                    pass
            self.last_uncertainty = self._estimate_gp_uncertainty()
            self._adapt_strategy_guardrails(progress)
            pre_runtime_state = {
                "runtime_llm_enabled": bool(self.runtime_llm_enabled),
                "runtime_sniper_enabled": bool(self.runtime_sniper_enabled),
                "runtime_surrogate_enabled": bool(self.runtime_surrogate_enabled),
            }
            safety_update = self._update_runtime_safety(progress)

            # LLM Strategy Call
            llm_called = False
            llm_suggestions_used = 0
            best_before_llm = float(self.best_fit)
            if (
                self.runtime_llm_enabled
                and (self.config.force_llm_every_gen or self.config.gating_enabled)
                and self._should_call_llm(gen, max_gen)
            ):
                prev_best = self.best_fit
                prev_calls = self.stage1_calls + self.stage2_calls
                suggestions = self._get_llm_suggestions(bounds, gen)
                if suggestions:
                    llm_called = True
                    llm_suggestions_used = len(suggestions)
                    self._integrate_suggestions(suggestions, objective_function)
                if (self.stage1_calls + self.stage2_calls) > prev_calls:
                    gain = max(0.0, prev_best - self.best_fit)
                    self._update_llm_feedback(gain)
            
            # Evolve based on Strategy
            self._evolve(bounds, objective_function)

            if self.config.enable_micro_refine and self.evaluations < self.config.max_evaluations:
                if gen % max(1, self.config.micro_refine_interval) == 0:
                    self._micro_refine(objective_function, bounds)
            
            # Sniper Mode: L-BFGS-B Refinement
            if self.runtime_sniper_enabled and self.commander_mode == STRATEGY_PRECISION and self.evaluations < self.config.max_evaluations:
                if self.no_improve_generations <= self.sniper_patience_generations and gen % max(1, int(self.config.sniper_interval)) == 0:
                    if self._allow_sniper_refine(progress):
                        x_refined, f_refined = self._local_refine(
                            self.population[self.best_idx],
                            objective_function,
                            bounds,
                            x0_value=self.best_fit,
                        )
                        if f_refined < self.best_fit:
                            self.fitness[self.best_idx] = f_refined
                            self.population[self.best_idx] = x_refined
                            self.best_fit = f_refined
                            self.last_generation_best = self.best_fit
                            self.no_improve_generations = 0
                            print(f"[Sniper] Refined to {self.best_fit:.2e}")

            post_runtime_state = {
                "runtime_llm_enabled": bool(self.runtime_llm_enabled),
                "runtime_sniper_enabled": bool(self.runtime_sniper_enabled),
                "runtime_surrogate_enabled": bool(self.runtime_surrogate_enabled),
            }
            if (
                pre_runtime_state["runtime_llm_enabled"]
                and (not post_runtime_state["runtime_llm_enabled"])
                and self.first_llm_disable_generation is None
            ):
                self.first_llm_disable_generation = gen
            if (
                pre_runtime_state["runtime_sniper_enabled"]
                and (not post_runtime_state["runtime_sniper_enabled"])
                and self.first_sniper_disable_generation is None
            ):
                self.first_sniper_disable_generation = gen
            if (
                pre_runtime_state["runtime_surrogate_enabled"]
                and (not post_runtime_state["runtime_surrogate_enabled"])
                and self.first_surrogate_disable_generation is None
            ):
                self.first_surrogate_disable_generation = gen
            self.generation_diagnostics.append(
                {
                    "generation": int(gen),
                    "progress": float(progress),
                    "stage": int(self.stage),
                    "evaluations": int(self.evaluations),
                    "best_fitness_before_llm": float(best_before_llm),
                    "best_fitness_after_generation": float(self.best_fit),
                    "no_improve_generations": int(self.no_improve_generations),
                    "stagnation_counter": int(self.stagnation_counter),
                    "last_population_diversity": float(self.last_population_diversity),
                    "last_uncertainty": float(self.last_uncertainty),
                    "landscape_risk_score": float(self.landscape_risk_score),
                    "rugged_mode": bool(self.rugged_mode),
                    "llm_trust_score": float(self.llm_trust_score),
                    "pre_runtime_state": pre_runtime_state,
                    "post_runtime_state": post_runtime_state,
                    "safety_update": safety_update,
                    "llm_called": bool(llm_called),
                    "llm_suggestions_used": int(llm_suggestions_used),
                    "stage1_calls": int(self.stage1_calls),
                    "stage2_calls": int(self.stage2_calls),
                    "api_success_calls": int(self.api_success_calls),
                }
            )
            
            gen += 1
            if gen >= max_gen and not self.config.enforce_full_evaluation_budget:
                break
            if gen >= max_generation_guard:
                raise RuntimeError("Evaluation budget was not exhausted before the generation safety guard")
            
        # Final Polish (if budget allows)
        if self.runtime_sniper_enabled and self.evaluations < self.config.max_evaluations and self.no_improve_generations <= self.sniper_patience_generations:
             if self._allow_sniper_refine(1.0):
                 elite_count = 1 if self.config.dimension < 20 else min(3, len(self.population))
                 if self.low_dim_budget_mode and self.runtime_surrogate_enabled:
                     elite_count = min(3, len(self.population))
                 elite_indices = [int(idx) for idx in np.argsort(self.fitness)[:elite_count]]
                 for idx in elite_indices:
                     if self.config.max_evaluations - self.evaluations < 5:
                         break
                     x_refined, f_refined = self._local_refine(
                         self.population[idx],
                         objective_function,
                         bounds,
                         x0_value=self.fitness[idx],
                     )
                     if f_refined < self.fitness[idx]:
                         self.fitness[idx] = f_refined
                         self.population[idx] = x_refined
                     if f_refined < self.best_fit:
                         self.best_idx = idx
                         self.best_fit = f_refined

        elapsed = time.time() - start
        
        min_fit = np.min(self.fitness)
        if min_fit == 0.0:
            print(f"CRITICAL: Found 0.0 fitness in population!")
            zero_idx = np.where(self.fitness == 0.0)[0]
            print(f"Indices with 0.0: {zero_idx}")
            print(f"Solutions at 0.0: {self.population[zero_idx]}")
            # Re-evaluate
            for idx in zero_idx:
                 val = objective_function(self.population[idx])
                 print(f"Re-eval solution {idx}: {val}")

        return {
            "best_fitness": float(min_fit),
            "best_solution": self.population[self.best_idx].tolist(),
            "api_used": bool(self.use_api),
            "api_client_mode": str(self.api_client_mode),
            "active_endpoint_index": int(self.active_endpoint_index),
            "endpoint_pool_size": int(len(self.api_endpoints)),
            "endpoint_switch_count": int(self.endpoint_switch_count),
            "active_base_url": str(self.base_url),
            "stage1_calls": int(self.stage1_calls),
            "stage2_calls": int(self.stage2_calls),
            "api_success_calls": int(self.api_success_calls),
            "total_cost_usd": float(self.total_cost_usd),
            "total_tokens": int(self.total_tokens),
            "evaluations_used": int(self.evaluations),
            "execution_time": float(elapsed),
            "runtime_llm_enabled": bool(self.runtime_llm_enabled),
            "runtime_sniper_enabled": bool(self.runtime_sniper_enabled),
            "runtime_surrogate_enabled": bool(self.runtime_surrogate_enabled),
            "convergence_curve": [float(x) for x in self.convergence],
            "best_solution_history": [list(map(float, solution)) for solution in self.best_solution_history],
            "diagnostics": {
                "generation_diagnostics": self.generation_diagnostics,
                "first_llm_disable_generation": self.first_llm_disable_generation,
                "first_sniper_disable_generation": self.first_sniper_disable_generation,
                "first_surrogate_disable_generation": self.first_surrogate_disable_generation,
                "api_client_mode": str(self.api_client_mode),
                "active_endpoint_index": int(self.active_endpoint_index),
                "endpoint_pool_size": int(len(self.api_endpoints)),
                "endpoint_switch_count": int(self.endpoint_switch_count),
                "active_base_url": str(self.base_url),
                "api_success_calls": int(self.api_success_calls),
                "total_cost_usd": float(self.total_cost_usd),
                "total_tokens": int(self.total_tokens),
                "llm_usage_details": self.llm_usage_details,
                "admission_funnel": self._aggregate_admission_funnel(),
                "admission_events": self.admission_events,
                "csa_trace": self._build_csa_trace(),
                "trace_manifest": {
                    "event_count": int(len(self.admission_events)),
                    "semantic_evaluations": int(sum(event.get("evaluated_candidates", 0) for event in self.admission_events)),
                    "runtime_authorization_events": int(sum(bool(event.get("runtime_authorization_enforced", False)) for event in self.admission_events)),
                },
                "proposal_trace": self.admission_events,
                "proposal_trace_mode": str(self.proposal_trace_mode),
                "proposal_replay_path": str(self.proposal_replay_path),
                "proposal_replay_source_sha256": str(self.proposal_replay_source_sha256),
                "proposal_replay_events_available": int(len(self.proposal_replay_events)),
                "proposal_replay_events_consumed": int(self.proposal_replay_cursor),
                "runtime_llm_enabled": bool(self.runtime_llm_enabled),
                "runtime_sniper_enabled": bool(self.runtime_sniper_enabled),
                "runtime_surrogate_enabled": bool(self.runtime_surrogate_enabled),
            },
        }

    def _should_call_llm(self, gen: int, max_gen: int) -> bool:
        if not self.runtime_llm_enabled: return False
        if self.proposal_trace_mode != "replay" and ((not self.use_api) or (self.client is None)): return False
        if self.evaluations < int(self.llm_cooldown_until_eval): return False
        if self.evaluations + self.config.min_remaining_budget >= self.config.max_evaluations: return False
        if self.total_cost_usd >= float(self.config.max_llm_cost_usd_per_run): return False
        if self.stage1_calls + self.stage2_calls >= self.config.max_llm_calls_per_run: return False
        if self.stage == 1 and self.stage1_calls >= self.config.max_llm_calls_stage1: return False
        if self.stage == 2 and self.stage2_calls >= self.config.max_llm_calls_stage2: return False
        if self.config.force_llm_every_gen:
            self.last_call_gen = gen
            return True
        if self.config.dimension >= 20:
            progress = self.evaluations / max(1, self.config.max_evaluations)
            if progress < 0.18 and self.stagnation_counter < (self.config.llm_stagnation_trigger + 2):
                return False
        if self.low_dim_budget_mode:
            progress = self.evaluations / max(1, self.config.max_evaluations)
            if gen == 0:
                return False
            if progress < 0.25 and self.stagnation_counter < max(1, self.config.llm_stagnation_trigger - 1):
                return False
        
        if gen == 0 and self.config.force_initial_llm_call:
            self.last_call_gen = gen
            self.last_call_progress_bucket = 0
            return True
            
        extra_wait = 0
        if self.low_gain_streak >= 3:
            extra_wait = max(1, self.effective_min_call_interval // 2)
        if self.llm_trust_score < self.config.min_confidence:
            extra_wait += 1

        prog = self.evaluations / self.config.max_evaluations
        progress_bucket = int(prog * 20)
        interval_ready = (gen - self.last_call_gen) >= (self.effective_min_call_interval + extra_wait)
        quality_ready = (self.llm_trust_score >= self.config.min_confidence * 0.65) or (self.stagnation_counter >= self.config.llm_stagnation_trigger)
        uncertainty_ready = (not self.gp_trained) or (self.last_uncertainty >= self.config.min_uncertainty_threshold)
        milestone_ready = progress_bucket > self.last_call_progress_bucket
        stagnation_ready = self.stagnation_counter >= int(self.config.llm_stagnation_trigger)
        diversity_drop_ready = self._is_diversity_declining()

        reward_ready = (self.high_gain_streak >= 2) and (self.llm_trust_score >= self.config.min_confidence)
        
        # --- 核心改造 1: 动态按需调用 (Conditional Trigger) ---
        # 如果地形风险极高(>=1.6)，说明陷入了严重的局部最优或停滞，强制请求大模型支援
        # 添加滞后阈值（hysteresis threshold）以减少误触发
        risk_threshold = 1.6 if self.llm_trust_score >= self.config.min_confidence else 1.8
        if self.landscape_risk_score >= risk_threshold and interval_ready:
            self.last_call_gen = gen
            self.last_call_progress_bucket = progress_bucket
            return True

        if interval_ready and stagnation_ready and diversity_drop_ready and quality_ready and uncertainty_ready:
            self.last_call_gen = gen
            self.last_call_progress_bucket = progress_bucket
            return True

        extra_window_ready = (gen - self.last_call_gen) >= max(1, self.effective_min_call_interval - 1)
        if stagnation_ready and diversity_drop_ready and extra_window_ready:
            if not uncertainty_ready and self.llm_trust_score < self.config.min_confidence:
                return False
            self.last_call_gen = gen
            self.last_call_progress_bucket = progress_bucket
            return True

        if interval_ready and milestone_ready and reward_ready and uncertainty_ready:
            self.last_call_gen = gen
            self.last_call_progress_bucket = progress_bucket
            return True
            
        return False

    def _new_admission_event(self, gen: int, progress: float) -> Dict[str, Any]:
        event = {
            "generation": int(gen),
            "stage": int(self.stage),
            "progress": float(progress),
            "commander_mode": str(self.commander_mode),
            "rugged_mode": bool(self.rugged_mode),
            "weak_admission_control": bool(self.config.weak_admission_control),
            "runtime_surrogate_enabled": bool(self.runtime_surrogate_enabled),
            "surrogate_guidance_ready": bool(self._surrogate_guidance_ready()),
            "surrogate_min_history": int(self.config.surrogate_min_history),
            "surrogate_acceptance_margin": float(self.config.surrogate_acceptance_margin),
            "surrogate_admission_policy": str(self.config.surrogate_admission_policy),
            "surrogate_unavailable_policy": str(self.config.surrogate_unavailable_policy),
            "surrogate_fallback_reason": "",
            "llm_trust_score": float(self.llm_trust_score),
            "last_uncertainty": float(self.last_uncertainty),
            "llm_triggered_calls": 1,
            "raw_candidates": 0,
            "dimension_valid_candidates": 0,
            "invalid_dimension_candidates": 0,
            "returned_candidates": 0,
            "duplicate_removed": 0,
            "dedup_retained_candidates": 0,
            "fused_candidates_added": 0,
            "post_fusion_candidates": 0,
            "local_filter_removed": 0,
            "local_filter_retained_candidates": 0,
            "validated_candidates": 0,
            "surrogate_screened_out": 0,
            "surrogate_mu": [],
            "surrogate_sigma": [],
            "surrogate_lcb": [],
            "incumbent_at_admission": None,
            "incumbent_vector_at_admission": [],
            "regret_predicted_mean": [],
            "regret_predicted_sigma": [],
            "regret_risk_upper": [],
            "regret_uncertainty_scale": float(self.config.regret_uncertainty_scale),
            "regret_risk_multiplier": float(self.config.regret_risk_multiplier),
            "regret_abstention_threshold": float(self.config.regret_abstention_threshold),
            "regret_abstained": False,
            "surrogate_admission_threshold": None,
            "surrogate_selected_indices": [],
            "selected_candidates": 0,
            "admitted_candidates": 0,
            "evaluation_budget_skipped": 0,
            "evaluated_candidates": 0,
            "population_accepted_candidates": 0,
            "best_update_candidates": 0,
            "low_confidence_rejections": 0,
            "parse_failures": 0,
            "api_failures": 0,
            "empty_response_calls": 0,
            "call_outcome": "started",
            "trace_mode": str(self.proposal_trace_mode),
            "source_call_index": None,
            "system_prompt": "",
            "user_prompt": "",
            "raw_response": "",
            "returned_candidate_vectors": [],
            "validated_candidate_vectors": [],
            "selected_candidate_vectors": [],
            "admitted_candidate_vectors": [],
            "evaluated_candidate_vectors": [],
            "evaluated_candidate_objectives": [],
            "evaluated_candidate_regrets": [],
            "population_accepted_candidate_vectors": [],
            "best_update_candidate_vectors": [],
            "runtime_authorization_enforced": True,
            "evaluation_authorizations": [],
            "candidate_lineage": {},
            "parameters_applied": {},
            "usage": {},
        }
        self.admission_events.append(event)
        self._pending_admission_event = event
        return event

    def _load_proposal_replay(self) -> None:
        if not self.proposal_replay_path:
            raise ValueError("proposal_replay_path is required when proposal_trace_mode='replay'")
        with open(self.proposal_replay_path, "rb") as handle:
            payload_bytes = handle.read()
        self.proposal_replay_source_sha256 = hashlib.sha256(payload_bytes).hexdigest()
        payload = json.loads(payload_bytes.decode("utf-8"))
        diagnostics = payload.get("diagnostics", {}) if isinstance(payload, dict) else {}
        events = diagnostics.get("proposal_trace") or diagnostics.get("admission_events") or []
        if not isinstance(events, list):
            raise ValueError(f"Invalid proposal trace in {self.proposal_replay_path}")
        self.proposal_replay_events = [dict(event) for event in events]

    def _replay_llm_suggestions(self, event: Dict[str, Any], dim: int) -> List[np.ndarray]:
        if self.proposal_replay_cursor >= len(self.proposal_replay_events):
            event["call_outcome"] = "replay_exhausted"
            self._close_admission_event("replay_exhausted")
            if self.proposal_replay_strict:
                raise RuntimeError("Proposal replay exhausted before the optimizer stopped requesting calls")
            return []

        source_index = int(self.proposal_replay_cursor)
        source = self.proposal_replay_events[source_index]
        self.proposal_replay_cursor += 1
        vectors = source.get("returned_candidate_vectors") or []
        suggestions: List[np.ndarray] = []
        for candidate in vectors:
            if isinstance(candidate, (list, tuple, np.ndarray)) and len(candidate) == dim:
                suggestions.append(np.asarray(candidate, dtype=float))

        parameters = dict(source.get("parameters_applied") or {})
        if parameters and self.config.allow_llm_control_updates:
            self._apply_llm_parameters(
                parameters.get("mode", self.commander_mode),
                float(parameters.get("F_base", self.current_F_base)),
                float(parameters.get("CR_base", self.current_CR_base)),
            )
        usage = dict(source.get("usage") or {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        cost_usd = float(usage.get("cost_usd", 0.0) or 0.0)
        self.total_tokens += total_tokens
        self.total_cost_usd += cost_usd
        self.llm_usage_details.append(
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_usd": cost_usd,
                "replayed": True,
                "source_call_index": source_index,
            }
        )
        event.update(
            {
                "source_call_index": source_index,
                "raw_candidates": int(source.get("raw_candidates", len(vectors))),
                "dimension_valid_candidates": int(len(suggestions)),
                "returned_candidates": int(len(suggestions)),
                "returned_candidate_vectors": [candidate.tolist() for candidate in suggestions],
                "parameters_applied": parameters,
                "usage": usage,
                "call_outcome": "replay_success" if suggestions else "replay_empty",
            }
        )
        if self.stage == 1:
            self.stage1_calls += 1
        else:
            self.stage2_calls += 1
        if suggestions:
            self.api_success_calls += 1
        return suggestions

    def _close_admission_event(self, outcome: Optional[str] = None) -> None:
        if self._pending_admission_event is None:
            return
        if outcome:
            self._pending_admission_event["call_outcome"] = str(outcome)
        self._pending_admission_event = None

    def _aggregate_admission_funnel(self) -> Dict[str, Any]:
        numeric_keys = [
            "llm_triggered_calls",
            "raw_candidates",
            "dimension_valid_candidates",
            "invalid_dimension_candidates",
            "returned_candidates",
            "duplicate_removed",
            "dedup_retained_candidates",
            "fused_candidates_added",
            "post_fusion_candidates",
            "local_filter_removed",
            "local_filter_retained_candidates",
            "validated_candidates",
            "surrogate_screened_out",
            "selected_candidates",
            "admitted_candidates",
            "evaluation_budget_skipped",
            "evaluated_candidates",
            "population_accepted_candidates",
            "best_update_candidates",
            "low_confidence_rejections",
            "parse_failures",
            "api_failures",
            "empty_response_calls",
        ]
        totals = {key: 0 for key in numeric_keys}
        outcome_counts: Dict[str, int] = {}
        for event in self.admission_events:
            for key in numeric_keys:
                totals[key] += int(event.get(key, 0))
            outcome = str(event.get("call_outcome", "unknown"))
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        totals["n_events"] = len(self.admission_events)
        totals["outcome_counts"] = outcome_counts
        return totals

    def _build_csa_trace(self) -> List[Dict[str, Any]]:
        trace = []
        for event in self.admission_events:
            trace.append(
                make_trace_event(
                    event.get("returned_candidate_vectors", []),
                    event.get("validated_candidate_vectors", []),
                    event.get("selected_candidate_vectors", []),
                    event.get("admitted_candidate_vectors", []),
                    event.get("evaluated_candidate_vectors", []),
                    population_accepted_vectors=event.get(
                        "population_accepted_candidate_vectors", []
                    ),
                    best_update_vectors=event.get("best_update_candidate_vectors", []),
                    generation=event.get("generation"),
                    surrogate_guidance_ready=event.get("surrogate_guidance_ready"),
                    surrogate_admission_policy=event.get("surrogate_admission_policy"),
                    surrogate_unavailable_policy=event.get("surrogate_unavailable_policy"),
                    surrogate_fallback_reason=event.get("surrogate_fallback_reason"),
                    runtime_authorization_enforced=event.get(
                        "runtime_authorization_enforced", False
                    ),
                    evaluation_authorizations=event.get(
                        "evaluation_authorizations", []
                    ),
                    candidate_lineage=event.get("candidate_lineage", {}),
                )
            )
        return trace

    def _update_llm_feedback(self, gain: float):
        self.llm_recent_gains.append(float(gain))
        if len(self.llm_recent_gains) > 6:
            self.llm_recent_gains = self.llm_recent_gains[-6:]

        ref = max(1.0, abs(float(self.best_fit)))
        rel_gain = float(gain) / ref
        
        if rel_gain > 1e-5 and self.last_llm_params is not None:
            self.successful_llm_strategies.append(self.last_llm_params)
            if len(self.successful_llm_strategies) > 3:
                self.successful_llm_strategies = self.successful_llm_strategies[-3:]
        
        self.llm_gain_ema = 0.8 * self.llm_gain_ema + 0.2 * rel_gain
        if rel_gain < 1e-4:
            self.low_gain_streak += 1
            self.high_gain_streak = 0
            target_trust = 0.35
        else:
            self.high_gain_streak += 1
            self.low_gain_streak = 0
            if rel_gain < 5e-4:
                target_trust = 0.65
            else:
                target_trust = 0.9
        self.llm_trust_score = float(np.clip(0.85 * self.llm_trust_score + 0.15 * target_trust, 0.2, 0.98))
        self.call_failure_streak = 0

        if rel_gain < 1e-4:
            self.effective_min_call_interval = min(self.dynamic_interval_max, self.effective_min_call_interval + 2)
            cooldown_span = self._budget_aware_cooldown(0.06)
            self.llm_cooldown_until_eval = max(self.llm_cooldown_until_eval, self.evaluations + cooldown_span)
        elif rel_gain >= 2e-3:
            self.effective_min_call_interval = max(1, self.effective_min_call_interval - 1)
            self.llm_cooldown_until_eval = 0
        elif rel_gain >= 5e-4:
            self.effective_min_call_interval = max(1, self.effective_min_call_interval)
            self.llm_cooldown_until_eval = 0

        if self.low_gain_streak >= 3:
            interval_step = 2 if self.llm_trust_score < self.config.min_confidence else 1
            self.effective_min_call_interval = min(self.dynamic_interval_max, self.effective_min_call_interval + interval_step)
            self.fusion_weight_current = min(0.97, self.fusion_weight_current + 0.03)
            cooldown_span = self._budget_aware_cooldown(0.10)
            self.llm_cooldown_until_eval = max(self.llm_cooldown_until_eval, self.evaluations + cooldown_span)
        elif self.high_gain_streak >= 2:
            self.effective_min_call_interval = max(1, self.effective_min_call_interval - 1)
            self.fusion_weight_current = max(0.55, self.fusion_weight_current - 0.02)
            self.llm_cooldown_until_eval = 0

    def _budget_aware_cooldown(self, ratio: float) -> int:
        pop_size = max(1, int(self.config.population_size))
        min_span = max(1, pop_size // 2)
        scaled_span = int(max(min_span, round(float(ratio) * float(self.config.max_evaluations))))
        max_span = max(pop_size * 3, int(0.25 * float(self.config.max_evaluations)))
        return max(1, min(scaled_span, max_span))

    def _surrogate_guidance_ready(self) -> bool:
        if (not self.runtime_surrogate_enabled) or (not self.gp_trained):
            return False
        configured = int(self.config.surrogate_min_history)
        if configured > 0:
            min_history = configured
        else:
            min_history = max(
                int(self.config.population_size) + 4,
                min(int(self.config.dimension) * 2, max(1, int(self.config.max_evaluations) // 4)),
            )
        if len(self.X_history) < min_history:
            return False
        if self.low_dim_budget_mode and self.last_uncertainty > float(self.config.min_uncertainty_threshold) * 1.8:
            return False
        return True

    def _adapt_strategy_guardrails(self, progress: float):
        uncertainty_term = 0.0
        if self.config.min_uncertainty_threshold > 0:
            uncertainty_term = self.last_uncertainty / max(1e-12, float(self.config.min_uncertainty_threshold))
        stagnation_term = self.stagnation_counter / max(1.0, float(self.config.llm_stagnation_trigger))
        
        diversity_gradient = 0.0
        if len(self.diversity_history) >= 2:
            diversity_gradient = self.diversity_history[-1] - self.diversity_history[-2]
            
        raw = (uncertainty_term * 0.5) + (stagnation_term * 0.3) + (abs(diversity_gradient) * 10.0 * 0.2)
        self.landscape_risk_score = float(np.clip(raw, 0.0, 3.0))
        self.rugged_mode = bool(progress >= 0.45 and self.landscape_risk_score >= 1.6)
        if self.low_dim_budget_mode:
            self.rugged_mode = bool(progress >= 0.75 and self.landscape_risk_score >= 2.2)
        if not self.runtime_llm_enabled:
            if self.commander_mode in [STRATEGY_EMERGENCY, STRATEGY_EXPLORATION]:
                self.commander_mode = STRATEGY_BALANCED
            if progress >= float(self.config.precision_progress_threshold):
                precision_ready = (self.landscape_risk_score <= 1.8) and (self.no_improve_generations <= self.sniper_patience_generations)
                self.commander_mode = STRATEGY_PRECISION if precision_ready else STRATEGY_BALANCED
            else:
                self.commander_mode = STRATEGY_BALANCED
            if self.rugged_mode:
                self.fusion_weight_current = float(np.clip(self.fusion_weight_current, 0.84, 0.95))
            else:
                self.fusion_weight_current = float(np.clip(self.fusion_weight_current, 0.58, 0.92))
            return
        if self.low_dim_budget_mode:
            if progress >= min(float(self.config.precision_progress_threshold), 0.70):
                precision_ready = self.no_improve_generations <= (self.sniper_patience_generations + 1)
                self.commander_mode = STRATEGY_PRECISION if precision_ready else STRATEGY_BALANCED
            else:
                self.commander_mode = STRATEGY_BALANCED
            self.fusion_weight_current = float(np.clip(self.fusion_weight_current, 0.72, 0.94))
            self.effective_min_call_interval = min(
                self.dynamic_interval_max,
                max(self.effective_min_call_interval, int(self.config.min_call_interval)),
            )
            return
        trigger = int(self.config.llm_stagnation_trigger)
        moderate_stagnation = self.stagnation_counter >= (trigger + 2)
        severe_stagnation = self.stagnation_counter >= (trigger + 6)
        emergency_cooldown = max(25, int(0.08 * self.config.max_evaluations))
        emergency_ready = (self.evaluations - self.last_emergency_eval) >= emergency_cooldown
        low_diversity = self.last_population_diversity <= 0.10

        if self.rugged_mode and self.commander_mode == STRATEGY_PRECISION:
            self.commander_mode = STRATEGY_BALANCED
        if progress < float(self.config.precision_progress_threshold):
            if severe_stagnation and self.no_improve_generations >= 6 and progress < 0.75 and emergency_ready:
                self.commander_mode = STRATEGY_EMERGENCY
            elif moderate_stagnation and (low_diversity or self.rugged_mode) and progress < 0.85:
                self.commander_mode = STRATEGY_EXPLORATION
            elif self.commander_mode in [STRATEGY_EMERGENCY, STRATEGY_EXPLORATION] and self.no_improve_generations <= 2:
                self.commander_mode = STRATEGY_BALANCED
        if progress >= float(self.config.precision_progress_threshold) and self.commander_mode != STRATEGY_EMERGENCY:
            precision_ready = (self.landscape_risk_score <= 1.8) and (self.no_improve_generations <= self.sniper_patience_generations)
            self.commander_mode = STRATEGY_PRECISION if precision_ready else STRATEGY_BALANCED

        if self.rugged_mode:
            self.fusion_weight_current = float(np.clip(self.fusion_weight_current, 0.84, 0.95))
            self.effective_min_call_interval = min(self.dynamic_interval_max, max(self.effective_min_call_interval, self.config.min_call_interval + 1))
        else:
            self.fusion_weight_current = float(np.clip(self.fusion_weight_current, 0.58, 0.92))

    def _allow_sniper_refine(self, progress: float) -> bool:
        if self.rugged_mode and progress < 0.80:
            return False
        if self.config.dimension >= 20 and progress < 0.90:
            return False
        diversity_limit = 0.22
        if self.low_dim_budget_mode:
            diversity_limit = 0.28
        if self.config.dimension >= 20 and progress >= float(self.config.precision_progress_threshold):
            diversity_limit = 0.30
        if self.last_population_diversity > diversity_limit and progress < 0.95:
            return False
        if self._surrogate_guidance_ready() and self.last_uncertainty > self.config.min_uncertainty_threshold * 1.35 and progress < 0.95:
            return False
        return True

    def _register_llm_failure(self):
        self.call_failure_streak += 1
        self.low_gain_streak = min(self.low_gain_streak + 1, 10)
        self.high_gain_streak = 0
        self.llm_trust_score = float(np.clip(self.llm_trust_score * 0.9, 0.2, 0.98))
        if self.call_failure_streak >= 2:
            self.effective_min_call_interval = min(self.dynamic_interval_max, self.effective_min_call_interval + 1)

    def _update_runtime_safety(self, progress: float):
        update_info = {
            "llm_disable_reasons": [],
            "sniper_disable_reasons": [],
            "surrogate_disable_reasons": [],
        }
        precision_stack_protected = self.config.dimension >= 20 and progress < float(self.config.precision_progress_threshold)
        sniper_endgame_protected = self.config.dimension >= 20 and progress < 0.90
        if self.runtime_llm_enabled:
            if self.config.protect_runtime_llm:
                update_info["llm_disable_reasons"].append("protected_runtime_llm")
            else:
                disable_llm = False
                if self.low_gain_streak >= 4 and progress >= 0.30:
                    disable_llm = True
                    update_info["llm_disable_reasons"].append("low_gain_streak>=4_and_progress>=0.30")
                if self.call_failure_streak >= 3 and progress >= 0.20:
                    disable_llm = True
                    update_info["llm_disable_reasons"].append("call_failure_streak>=3_and_progress>=0.20")
                if self.rugged_mode and self.llm_trust_score < (self.config.min_confidence * 0.9) and self.low_gain_streak >= 2 and progress >= 0.45:
                    disable_llm = True
                    update_info["llm_disable_reasons"].append("rugged_mode_low_trust_low_gain_progress>=0.45")
                if disable_llm:
                    self.runtime_llm_enabled = False
                    self.llm_cooldown_until_eval = self.config.max_evaluations

        if self.runtime_sniper_enabled:
            if self.config.protect_runtime_sniper:
                update_info["sniper_disable_reasons"].append("protected_runtime_sniper")
            else:
                if (not precision_stack_protected) and (not sniper_endgame_protected) and self.no_improve_generations > (self.sniper_patience_generations + 2) and progress < 0.90:
                    self.runtime_sniper_enabled = False
                    update_info["sniper_disable_reasons"].append("no_improve_generations_exceeded_patience")
                if (not precision_stack_protected) and (not sniper_endgame_protected) and self.rugged_mode and self.last_population_diversity > 0.18 and progress < 0.92:
                    self.runtime_sniper_enabled = False
                    update_info["sniper_disable_reasons"].append("rugged_mode_high_diversity_before_0.92")

        if self.runtime_surrogate_enabled:
            if self.config.protect_runtime_surrogate:
                update_info["surrogate_disable_reasons"].append("protected_runtime_surrogate")
            else:
                high_uncertainty = self.last_uncertainty > (self.config.min_uncertainty_threshold * 1.6)
                if self.low_dim_budget_mode:
                    if (
                        (not precision_stack_protected)
                        and progress >= 0.85
                        and high_uncertainty
                        and len(self.X_history) < max(int(self.config.population_size) + 4, int(self.config.dimension) * 8)
                    ):
                        self.runtime_surrogate_enabled = False
                        update_info["surrogate_disable_reasons"].append("low_dim_budget_mode_high_uncertainty_late_sparse_history")
                elif (not precision_stack_protected) and self.rugged_mode and high_uncertainty and progress >= 0.40:
                    self.runtime_surrogate_enabled = False
                    update_info["surrogate_disable_reasons"].append("rugged_mode_high_uncertainty_progress>=0.40")
        return update_info

    def _estimate_population_diversity(self, bounds: List[Tuple[float, float]]) -> float:
        if self.population is None or len(self.population) == 0:
            return 0.0
        ranges = np.array([max(1e-12, (hi - lo)) for lo, hi in bounds], dtype=float)
        pop_std = np.std(self.population, axis=0)
        return float(np.mean(pop_std / ranges))

    def _estimate_gp_uncertainty(self) -> float:
        if (not self.gp_trained) or (self.population is None) or (self.fitness is None):
            return 0.0
        try:
            pop_size = len(self.population)
            top_k = min(5, pop_size)
            rnd_k = min(5, pop_size)
            top_idx = np.argsort(self.fitness)[:top_k]
            rnd_idx = np.random.choice(pop_size, size=rnd_k, replace=False)
            idx = np.unique(np.concatenate([top_idx, rnd_idx]))
            _, sigma = self.gp.predict(self.population[idx], return_std=True)
            return float(np.mean(sigma))
        except Exception:
            return 0.0

    def _is_diversity_declining(self) -> bool:
        if len(self.diversity_history) < 6:
            return False
        recent = self.diversity_history[-3:]
        prev = self.diversity_history[-6:-3]
        recent_mean = sum(recent) / len(recent)
        prev_mean = sum(prev) / len(prev)
        if prev_mean <= 1e-12:
            return False
        return recent_mean <= prev_mean * 0.93

    def _llm_eval_budget(self) -> int:
        base = 2
        if self.commander_mode == STRATEGY_EXPLORATION:
            base = 3
        elif self.commander_mode == STRATEGY_EMERGENCY:
            base = 4
        elif self.commander_mode == STRATEGY_PRECISION:
            base = 2
        if self.llm_trust_score > 0.75:
            base += 1
        if self.low_gain_streak >= 3:
            base -= 1
        if self.stage == 2 and self.commander_mode == STRATEGY_PRECISION:
            base = min(base, 2)
        if self.rugged_mode:
            base = min(base, 2)
        if self.rugged_mode and self.config.dimension >= 20:
            base = 1
        if self.low_dim_budget_mode:
            progress = self.evaluations / max(1, self.config.max_evaluations)
            base = min(base, 2)
            if self.commander_mode != STRATEGY_PRECISION or progress < 0.70:
                base = 1
        return int(np.clip(base, 1, 5))

    def _apply_llm_parameters(self, mode: str, f_base: float, cr_base: float):
        progress = self.evaluations / max(1, self.config.max_evaluations)
        chosen_mode = self.commander_mode
        proposed_mode = mode if mode in [STRATEGY_EXPLORATION, STRATEGY_BALANCED, STRATEGY_PRECISION, STRATEGY_EMERGENCY] else chosen_mode
        if self.low_dim_budget_mode:
            if proposed_mode in [STRATEGY_EXPLORATION, STRATEGY_EMERGENCY]:
                proposed_mode = STRATEGY_BALANCED
            if proposed_mode == STRATEGY_PRECISION and progress < 0.60:
                proposed_mode = STRATEGY_BALANCED
        if proposed_mode == STRATEGY_EXPLORATION and progress < 0.15 and not self.rugged_mode:
            chosen_mode = STRATEGY_EXPLORATION
        elif proposed_mode == STRATEGY_PRECISION:
            if progress >= float(self.config.precision_progress_threshold) and self.landscape_risk_score <= 1.6:
                chosen_mode = STRATEGY_PRECISION
        elif proposed_mode == STRATEGY_EMERGENCY:
            if self.stagnation_counter >= (self.config.llm_stagnation_trigger + 2):
                chosen_mode = STRATEGY_EMERGENCY
        elif proposed_mode == STRATEGY_BALANCED:
            chosen_mode = STRATEGY_BALANCED
        if chosen_mode == STRATEGY_EXPLORATION:
            self.current_F_base = float(np.clip(f_base, 0.45, 0.85))
            self.current_CR_base = float(np.clip(cr_base, 0.35, 0.95))
        elif chosen_mode == STRATEGY_PRECISION:
            self.current_F_base = float(np.clip(f_base, 0.05, 0.30))
            self.current_CR_base = float(np.clip(cr_base, 0.80, 0.99))
        elif chosen_mode == STRATEGY_EMERGENCY:
            self.current_F_base = float(np.clip(f_base, 0.65, 1.00))
            self.current_CR_base = float(np.clip(cr_base, 0.10, 0.70))
        else:
            self.current_F_base = float(np.clip(f_base, 0.25, 0.70))
            self.current_CR_base = float(np.clip(cr_base, 0.55, 0.98))
        self.commander_mode = chosen_mode

    def _update_usage_cost(self, response: Any, payload: Dict[str, Any] = None):
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt_tokens = float(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = float(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = float(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
        self.total_tokens += int(total_tokens)

        # Calculate character lengths if payload is provided
        system_chars = 0
        user_chars = 0
        if payload and "messages" in payload:
            for msg in payload["messages"]:
                if msg["role"] == "system":
                    system_chars += len(msg["content"])
                elif msg.get("role") == "user":
                    user_chars += len(msg["content"])

        response_content = ""
        if hasattr(response, "choices") and len(response.choices) > 0:
            response_content = response.choices[0].message.content

        prompt_rate = float(os.getenv("SAFE_LLM_SAEA_COST_PER_1K_PROMPT", "0.00014"))
        completion_rate = float(os.getenv("SAFE_LLM_SAEA_COST_PER_1K_COMPLETION", "0.00028"))
        call_cost_usd = (prompt_tokens / 1000.0) * prompt_rate + (completion_tokens / 1000.0) * completion_rate
        self.llm_usage_details.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "system_chars": system_chars,
            "user_chars": user_chars,
            "response_chars": len(response_content),
            "cost_usd": float(call_cost_usd),
        })
        self.total_cost_usd += call_cost_usd

    def _build_chat_completions_url(self) -> str:
        base = (self.base_url or "https://api.deepseek.com").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/chat/completions"

    def _activate_endpoint(self, endpoint_index: int, reason: str = "") -> None:
        if not self.api_endpoints:
            raise RuntimeError("No API endpoints are available")
        endpoint = self.api_endpoints[int(endpoint_index) % len(self.api_endpoints)]
        self.active_endpoint_index = int(endpoint["index"])
        self.api_key = str(endpoint["api_key"])
        self.base_url = str(endpoint["base_url"])
        force_requests_transport = os.getenv("SAFE_LLM_SAEA_FORCE_REQUESTS_TRANSPORT", "").strip().lower() in {"1", "true", "yes", "on"}
        if force_requests_transport and requests is not None:
            self.client = SimpleNamespace(kind="requests")
            self.api_client_mode = "requests"
        elif OpenAI is None:
            if requests is None:
                self.client = None
                self.api_client_mode = "disabled"
                raise RuntimeError("API client dependencies are unavailable")
            self.client = SimpleNamespace(kind="requests")
            self.api_client_mode = "requests"
        else:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                self.api_client_mode = "openai"
            except Exception:
                if requests is None:
                    self.client = None
                    self.api_client_mode = "disabled"
                    raise
                self.client = SimpleNamespace(kind="requests")
                self.api_client_mode = "requests"
        print(
            f"[LLM-Assisted DE] Active endpoint #{self.active_endpoint_index + 1}/{len(self.api_endpoints)} "
            f"base={self.base_url} mode={self.api_client_mode}"
            + (f" reason={reason}" if reason else "")
        )

    def _is_failover_error(self, error_text: str) -> bool:
        text = str(error_text or "").lower()
        keywords = (
            "401",
            "402",
            "403",
            "429",
            "unauthorized",
            "insufficient_quota",
            "quota",
            "credit",
            "balance",
            "payment",
            "rate limit",
            "too many requests",
            "connection error",
            "connect",
            "transport",
            "timeout",
        )
        return any(keyword in text for keyword in keywords)

    def _next_endpoint_index(self, attempted_indices: set[int]) -> Optional[int]:
        if len(self.api_endpoints) <= 1:
            return None
        for offset in range(1, len(self.api_endpoints) + 1):
            candidate = (self.active_endpoint_index + offset) % len(self.api_endpoints)
            if candidate not in attempted_indices:
                return candidate
        return None

    def _request_chat_completion_once(self, payload: Dict[str, Any], timeout: int) -> Any:
        if self.api_client_mode == "openai" and self.client is not None:
            return self.client.chat.completions.create(**payload, timeout=timeout)
        if self.api_client_mode == "requests" and requests is not None:
            response = requests.post(
                self._build_chat_completions_url(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            if response.status_code >= 400:
                body = response.text[:800]
                raise RuntimeError(f"HTTP {response.status_code} from {self.base_url}: {body}")
            data = response.json()
            message = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
            usage = data.get("usage") or {}
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=message))],
                usage=SimpleNamespace(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
            )
        raise RuntimeError("No API transport is available")

    def _request_chat_completion(self, payload: Dict[str, Any], timeout: int) -> Any:
        attempted_indices: set[int] = set()
        last_error: Optional[Exception] = None
        while True:
            attempted_indices.add(int(self.active_endpoint_index))
            try:
                return self._request_chat_completion_once(payload, timeout)
            except Exception as e:
                last_error = e
                if self.api_client_mode == "openai" and requests is not None and self._is_failover_error(str(e)):
                    self.client = SimpleNamespace(kind="requests")
                    self.api_client_mode = "requests"
                    print("[LLM-Assisted DE] Switching API transport from OpenAI client to requests after request failure.")
                    continue
                if not self._is_failover_error(str(e)):
                    raise
                next_index = self._next_endpoint_index(attempted_indices)
                if next_index is None:
                    break
                self.endpoint_switch_count += 1
                self._activate_endpoint(next_index, reason=f"failover after error: {str(e)[:160]}")
        if last_error is not None and len(self.api_endpoints) > 1:
            raise RuntimeError(
                f"All API endpoints failed for model {self.llm_model} after {len(attempted_indices)} attempts. "
                f"Last error: {last_error}"
            ) from last_error
        if last_error is not None:
            raise last_error
        raise RuntimeError("No API transport is available")

    def _get_llm_suggestions(self, bounds: List[Tuple[float, float]], gen: int) -> List[np.ndarray]:
        k = self.config.llm_suggestions_k
        dim = len(bounds)
        sorted_idxs = np.argsort(self.fitness)
        best_3 = self.population[sorted_idxs[:3]].tolist()
        pop_std = np.std(self.population, axis=0)
        diversity = np.mean(pop_std)
        progress = self.evaluations / self.config.max_evaluations

        self.llm_call_points.append(gen)
        sols: List[np.ndarray] = []
        event = self._new_admission_event(gen=gen, progress=progress)

        if self.proposal_trace_mode == "replay":
            return self._replay_llm_suggestions(event, dim)

        if not (self.use_api and self.client):
            self._close_admission_event("transport_unavailable")
            return sols

        call_success = False
        state_desc = (
            f"Progress: {progress*100:.1f}%. "
            f"Stagnation: {self.stagnation_counter} generations without improvement. "
            f"Diversity: {diversity:.2e} (Lower value means population is converging or trapped). "
            f"Landscape Risk Score: {self.landscape_risk_score:.2f} (>=1.6 means high risk of local optimum trap)."
        )
        if progress < 0.3:
            phase = "Early Exploration"
        elif progress < 0.7:
            phase = "Mid-stage Optimization"
        else:
            phase = "Late-stage Convergence"

        system_prompt = f"""You are an advanced Evolutionary Algorithm Orchestrator.
Your goal is to analyze the optimization state and select the optimal strategy mode.
Available Modes:
1. "{STRATEGY_EXPLORATION}": High diversity, large mutation steps. Use when stuck in local optima or diversity is too low early on.
2. "{STRATEGY_BALANCED}": Standard adaptive evolution (SHADE). Use for general progress.
3. "{STRATEGY_PRECISION}": Local search (L-BFGS-B) and micro-mutation. Use when close to global optimum (late stage) or landscape is unimodal.
4. "{STRATEGY_EMERGENCY}": Drastic perturbation. Use when severely stagnated for long periods.

Return strictly JSON format:
{{
  "solutions": [[x1, x2...], ...],
  "parameters": {{
      "F_base": 0.5,
      "CR_base": 0.9,
      "mode": "{STRATEGY_BALANCED}",
      "reasoning": "Explain why you chose this mode.",
      "confidence": 0.9
  }}
}}
"""
        user_prompt = f"""Task: Minimize {self.config.problem_description} in {dim}D.
Bounds: {bounds}
Top 3 Solutions: {best_3}
State: {state_desc}
Phase: {phase}
"""
        if self.successful_llm_strategies:
            user_prompt += f"Previously successful strategies (Mode, F, CR): {self.successful_llm_strategies}\n"

        user_prompt += f"Request: Suggest {k} new solutions and select the Strategy Mode. "
        if self.landscape_risk_score >= 1.6:
            user_prompt += "Notice: The population is severely stagnated or trapped. Please prioritize 'exploration' or 'emergency' mode and provide parameters that promote strong perturbation (e.g., high F and CR)."

        payload = {
            "model": self.llm_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.2,
            "max_tokens": 4096,
        }
        event["system_prompt"] = system_prompt
        event["user_prompt"] = user_prompt

        try:
            print(f"DEBUG: LLM Payload Model: {payload.get('model')}")
            print(f"DEBUG: LLM System Prompt Length: {len(payload['messages'][0]['content'])}")
            print(f"DEBUG: LLM User Prompt Length: {len(payload['messages'][1]['content'])}")

            resp = self._request_chat_completion(payload, timeout=60)
            self._update_usage_cost(resp, payload)

            print(f"DEBUG: LLM API Usage: {getattr(resp, 'usage', 'N/A')}")
            content = str(resp.choices[0].message.content or "")
            event["raw_response"] = content
            print(f"DEBUG: LLM Response Length: {len(content)}")
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0]
            content = content.strip()
            if not content:
                event["empty_response_calls"] += 1

            parsed = json.loads(content)
            params = parsed.get("parameters", {})
            raw_sols = parsed.get("solutions", [])
            if not isinstance(raw_sols, list):
                raw_sols = []

            event["raw_candidates"] = int(len(raw_sols))
            if not raw_sols:
                event["empty_response_calls"] += 1

            valid_count = 0
            invalid_count = 0
            for candidate in raw_sols:
                if not isinstance(candidate, (list, tuple, np.ndarray)) or len(candidate) != dim:
                    invalid_count += 1
                    continue
                try:
                    sols.append(np.asarray(candidate, dtype=float))
                    valid_count += 1
                except Exception:
                    invalid_count += 1

            event["dimension_valid_candidates"] = int(valid_count)
            event["invalid_dimension_candidates"] = int(invalid_count)
            event["returned_candidates"] = int(len(sols))
            event["returned_candidate_vectors"] = [candidate.tolist() for candidate in sols]

            confidence_score = float(params.get("confidence", 1.0))
            if confidence_score < 0.7:
                print(f"[LLM Validator] Rejecting low-confidence LLM suggestion (score: {confidence_score:.2f})")
                event["low_confidence_rejections"] += 1
                sols = []
                self._register_llm_failure()
                event["call_outcome"] = "low_confidence_rejected"
            else:
                new_mode = params.get("mode", self.commander_mode)
                new_f_base = float(params.get("F_base", self.current_F_base))
                new_cr_base = float(params.get("CR_base", self.current_CR_base))
                if self.config.allow_llm_control_updates:
                    self._apply_llm_parameters(new_mode, new_f_base, new_cr_base)
                    self.last_llm_params = (
                        self.commander_mode, self.current_F_base, self.current_CR_base
                    )
                    event["parameters_applied"] = {
                        "mode": str(self.commander_mode),
                        "F_base": float(self.current_F_base),
                        "CR_base": float(self.current_CR_base),
                        "confidence": float(confidence_score),
                    }
                if self.llm_usage_details:
                    event["usage"] = dict(self.llm_usage_details[-1])

                print(f"[LLM Orchestrator] Mode: {self.commander_mode} | F: {self.current_F_base:.2f} | CR: {self.current_CR_base:.2f}")
                print(f"  Reasoning: {params.get('reasoning', 'N/A')}")

                self.api_success_calls += 1
                call_success = True
                event["call_outcome"] = "api_success"

        except json.JSONDecodeError as e:
            print(f"[LLM Error] {e}")
            event["parse_failures"] += 1
            event["call_outcome"] = "parse_failure"
            self._register_llm_failure()
        except Exception as e:
            error_msg = str(e).lower()
            print(f"[LLM Error] {e}")
            event["api_failures"] += 1

            if "401" in error_msg or "unauthorized" in error_msg:
                print("[LLM Fatal] API Key authentication failed. Disabling LLM to prevent further errors.")
                self.use_api = False
                self.runtime_llm_enabled = False
                event["call_outcome"] = "auth_failure"
            elif "timeout" in error_msg:
                print("[LLM Warning] API request timed out. Will backoff and retry later.")
                event["call_outcome"] = "timeout"
                self._register_llm_failure()
            else:
                event["call_outcome"] = "api_failure"
                self._register_llm_failure()

        if call_success or self.config.mark_call_on_failure:
            if self.stage == 1:
                self.stage1_calls += 1
            else:
                self.stage2_calls += 1

        if not sols:
            outcome = str(event.get("call_outcome", "no_candidates_returned"))
            if outcome == "api_success":
                if event["raw_candidates"] > 0 and event["dimension_valid_candidates"] == 0:
                    outcome = "all_invalid_dimensions"
                else:
                    outcome = "api_success_empty"
            self._close_admission_event(outcome)
        return sols

    def _integrate_suggestions(self, suggestions: List[np.ndarray], objective_function: Callable):
        event = self._pending_admission_event
        if not suggestions:
            self._close_admission_event("no_candidates_returned")
            return

        original_count = len(suggestions)
        dedup = {}
        for s in suggestions:
            arr = np.asarray(s, dtype=float)
            key = tuple(np.round(arr, 8).tolist())
            dedup[key] = arr
        suggestions = list(dedup.values())
        if event is not None:
            event["duplicate_removed"] += int(max(0, original_count - len(suggestions)))
            event["dedup_retained_candidates"] = int(len(suggestions))
        if not suggestions:
            self._close_admission_event("dedup_removed_all")
            return

        to_evaluate = []
        fw = self.fusion_weight_current
        progress = self.evaluations / max(1, self.config.max_evaluations)
        if self.low_dim_budget_mode:
            fw = max(fw, 0.82 if self.commander_mode == STRATEGY_PRECISION else 0.78)
        if self.best_idx is not None and suggestions:
            best = self.population[self.best_idx]
            projection_alpha = float(self.config.semantic_projection_alpha)
            if projection_alpha > 0.0:
                projected = []
                for s in suggestions:
                    projected_candidate = best + projection_alpha * (s - best)
                    projected.append(projected_candidate)
                    if event is not None:
                        event["candidate_lineage"][candidate_id(projected_candidate)] = [
                            candidate_id(s)
                        ]
                suggestions = projected
                if event is not None:
                    event["fused_candidates_added"] += int(len(projected))
            elif not (
                self.config.dimension >= 20 and self.rugged_mode and progress < 0.75
            ):
                fused = []
                for s in suggestions:
                    fused_candidate = fw * best + (1.0 - fw) * s
                    fused.append(fused_candidate)
                    if event is not None:
                        event["candidate_lineage"][candidate_id(fused_candidate)] = [
                            candidate_id(s)
                        ]
                suggestions = suggestions + fused
                if event is not None:
                    event["fused_candidates_added"] += int(len(fused))
            if event is not None:
                event["post_fusion_candidates"] = int(len(suggestions))
            if self.commander_mode != STRATEGY_EXPLORATION and progress >= 0.25:
                ranges = np.array([max(1e-12, hi - lo) for lo, hi in self.bounds], dtype=float)
                filtered = []
                for s in suggestions:
                    radius = float(np.mean(np.abs((s - best) / ranges)))
                    radius_limit = 0.55
                    if self.low_dim_budget_mode:
                        radius_limit = 0.28 if self.commander_mode == STRATEGY_PRECISION else 0.38
                    if radius <= radius_limit:
                        filtered.append(s)
                if filtered:
                    if event is not None:
                        event["local_filter_removed"] += int(max(0, len(suggestions) - len(filtered)))
                    suggestions = filtered
        if event is not None:
            if event["post_fusion_candidates"] <= 0:
                event["post_fusion_candidates"] = int(len(suggestions))
            event["local_filter_retained_candidates"] = int(len(suggestions))
            event["validated_candidates"] = int(len(suggestions))
            event["validated_candidate_vectors"] = [
                np.asarray(candidate, dtype=float).tolist() for candidate in suggestions
            ]

        eval_budget = self._llm_eval_budget()
        if self.low_dim_budget_mode:
            eval_budget = min(eval_budget, 1 if self.commander_mode != STRATEGY_PRECISION else 2)
        selected_candidates = []

        if self.config.weak_admission_control:
            if len(suggestions) <= eval_budget:
                to_evaluate = list(suggestions)
            else:
                sampled_idx = np.random.choice(len(suggestions), size=eval_budget, replace=False)
                to_evaluate = [suggestions[int(idx)] for idx in sampled_idx]
            selected_candidates = list(to_evaluate)
        elif self.config.surrogate_admission_policy == "response_order":
            selected_candidates = list(suggestions)
            to_evaluate = selected_candidates[:eval_budget]
            if event is not None:
                event["surrogate_selected_indices"] = list(range(len(selected_candidates)))
                event["surrogate_fallback_reason"] = "not_applicable_response_order"
        elif self._surrogate_guidance_ready():
            try:
                X_sugg = np.array(suggestions)
                mu, sigma = self.gp.predict(X_sugg, return_std=True)
                kappa = 0.0
                if self.commander_mode == STRATEGY_EXPLORATION:
                    kappa = 2.0
                elif self.commander_mode == STRATEGY_BALANCED:
                    kappa = 0.5

                lcb = mu - kappa * sigma

                required_improvement = float(self.config.surrogate_acceptance_margin) * max(
                    1.0, abs(float(self.best_fit))
                )
                admission_threshold = float(self.best_fit) - required_improvement
                policy = str(self.config.surrogate_admission_policy)
                incumbent_scale = max(1.0, abs(float(self.best_fit)))
                regret_mu = (mu - float(self.best_fit)) / incumbent_scale
                regret_sigma = (
                    float(self.config.regret_uncertainty_scale) * sigma / incumbent_scale
                )
                if policy == "calibrated_regret" and self.config.regret_model_coefficients:
                    if self.best_idx is None:
                        raise RuntimeError("calibrated regret model requires an incumbent vector")
                    feature_matrix = regret_feature_matrix(
                        mu,
                        sigma,
                        suggestions,
                        self.population[self.best_idx],
                        self.bounds,
                        progress,
                    )
                    regret_model = {
                        "feature_mean": list(self.config.regret_model_feature_mean),
                        "feature_scale": list(self.config.regret_model_feature_scale),
                        "coefficients": list(self.config.regret_model_coefficients),
                    }
                    regret_mu = apply_regret_linear_model(feature_matrix, regret_model)
                    regret_sigma = np.full(
                        len(suggestions),
                        float(self.config.regret_model_residual_radius),
                        dtype=float,
                    )
                benefit_probability = np.zeros(len(suggestions), dtype=float)
                if policy == "backbone_risk":
                    if not self.config.risk_model_coefficients or self.best_idx is None:
                        raise RuntimeError("backbone risk model is unavailable")
                    risk_features = backbone_risk_feature_matrix(
                        mu, sigma, suggestions, self.population[self.best_idx],
                        float(self.best_fit), self.bounds, progress,
                    )
                    benefit_probability = apply_backbone_risk_model(
                        risk_features,
                        {
                            "feature_mean": list(self.config.risk_model_feature_mean),
                            "feature_scale": list(self.config.risk_model_feature_scale),
                            "coefficients": list(self.config.risk_model_coefficients),
                            "intercept": float(self.config.risk_model_intercept),
                        },
                    )
                regret_upper = regret_mu + (
                    float(self.config.regret_risk_multiplier) * regret_sigma
                )
                if policy == "mean_top":
                    ranking_values = mu
                elif policy == "calibrated_regret":
                    ranking_values = regret_upper
                elif policy == "backbone_risk":
                    ranking_values = -benefit_probability
                else:
                    ranking_values = lcb
                sorted_idx = np.argsort(ranking_values)
                if event is not None:
                    event["incumbent_at_admission"] = float(self.best_fit)
                    if self.best_idx is not None:
                        event["incumbent_vector_at_admission"] = np.asarray(
                            self.population[self.best_idx], dtype=float
                        ).tolist()
                    event["surrogate_mu"] = [float(value) for value in mu]
                    event["surrogate_sigma"] = [float(value) for value in sigma]
                    event["surrogate_lcb"] = [float(value) for value in lcb]
                    event["regret_predicted_mean"] = [float(value) for value in regret_mu]
                    event["regret_predicted_sigma"] = [float(value) for value in regret_sigma]
                    event["regret_risk_upper"] = [float(value) for value in regret_upper]
                    event["benefit_probability"] = [
                        float(value) for value in benefit_probability
                    ]
                    event["surrogate_admission_threshold"] = float(admission_threshold)
                selected_indices = []
                for i in range(len(suggestions)):
                    idx = sorted_idx[i]
                    if self.rugged_mode and self.commander_mode != STRATEGY_EXPLORATION and i > 1:
                        continue
                    if policy in {"rank_top", "mean_top"}:
                        admitted = True
                    elif policy == "calibrated_regret":
                        admitted = bool(
                            regret_upper[idx]
                            <= float(self.config.regret_abstention_threshold)
                        )
                    elif policy == "backbone_risk":
                        admitted = bool(
                            benefit_probability[idx]
                            >= float(self.config.risk_probability_threshold)
                        )
                    else:
                        admitted = bool(lcb[idx] < admission_threshold)
                    if admitted:
                        selected_indices.append(int(idx))
                selected_candidates = [suggestions[idx] for idx in selected_indices]
                to_evaluate = selected_candidates[:eval_budget]
                if event is not None:
                    event["regret_abstained"] = bool(
                        policy in {"calibrated_regret", "backbone_risk"}
                        and len(selected_indices) == 0
                    )
                    event["surrogate_selected_indices"] = selected_indices
                    event["surrogate_screened_out"] += int(
                        max(0, len(suggestions) - len(selected_candidates))
                    )
            except Exception:
                if self.config.surrogate_unavailable_policy == "reject":
                    selected_candidates = []
                    to_evaluate = []
                    if event is not None:
                        event["surrogate_fallback_reason"] = "prediction_failed_rejected"
                        event["surrogate_screened_out"] += int(len(suggestions))
                else:
                    selected_candidates = list(suggestions)
                    to_evaluate = selected_candidates[:eval_budget]
                    if event is not None:
                        event["surrogate_fallback_reason"] = "prediction_failed_response_order"
        else:
            if self.config.surrogate_unavailable_policy == "reject":
                selected_candidates = []
                to_evaluate = []
                if event is not None:
                    event["surrogate_fallback_reason"] = "surrogate_not_ready_rejected"
                    event["surrogate_screened_out"] += int(len(suggestions))
            else:
                selected_candidates = list(suggestions)
                to_evaluate = selected_candidates[:eval_budget]
                if event is not None:
                    event["surrogate_fallback_reason"] = "surrogate_not_ready_response_order"

        if event is not None:
            event["selected_candidates"] = int(len(selected_candidates))
            event["admitted_candidates"] = int(len(to_evaluate))
            event["evaluation_budget_skipped"] += int(
                max(0, len(selected_candidates) - len(to_evaluate))
            )
            event["selected_candidate_vectors"] = [
                np.asarray(candidate, dtype=float).tolist()
                for candidate in selected_candidates
            ]
            event["admitted_candidate_vectors"] = [
                np.asarray(candidate, dtype=float).tolist() for candidate in to_evaluate
            ]

        population_accepted_count = 0
        best_update_count = 0
        evaluated_count = 0
        evaluated_vectors = []
        evaluated_objectives = []
        evaluated_regrets = []
        population_accepted_vectors = []
        best_update_vectors = []
        frozen_incumbent = float(self.best_fit)
        frozen_incumbent_scale = max(1.0, abs(frozen_incumbent))
        if event is not None and event.get("incumbent_at_admission") is None:
            event["incumbent_at_admission"] = frozen_incumbent
        if event is not None and not event.get("incumbent_vector_at_admission"):
            if self.best_idx is not None:
                event["incumbent_vector_at_admission"] = np.asarray(
                    self.population[self.best_idx], dtype=float
                ).tolist()
        event_id = (
            f"generation-{int(event.get('generation', -1))}-"
            f"event-{max(0, len(self.admission_events) - 1)}"
            if event is not None
            else f"untraced-event-{int(self.evaluations)}"
        )
        gate_policy = (
            "weak_sampling"
            if self.config.weak_admission_control
            else str(self.config.surrogate_admission_policy)
        )
        runtime_gate = RuntimeAdmissionGate(
            selected_candidates,
            to_evaluate,
            remaining_budget=max(0, self.config.max_evaluations - self.evaluations),
            event_id=event_id,
            policy=gate_policy,
        )
        for s in to_evaluate:
            if self.evaluations >= self.config.max_evaluations:
                break

            unclipped_id = candidate_id(s)
            s = np.clip(s, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            if candidate_id(s) != unclipped_id:
                raise AuthorizationError("candidate identity changed after host admission")

            prev_best = float(self.best_fit)
            authorization = runtime_gate.issue(s)
            if event is not None:
                event["evaluation_authorizations"].append(authorization.as_dict())
            f = float(runtime_gate.evaluate(authorization, s, objective_function))
            self.evaluations += 1
            evaluated_count += 1
            evaluated_vectors.append(s.copy())
            evaluated_objectives.append(float(f))
            evaluated_regrets.append(
                float((f - frozen_incumbent) / frozen_incumbent_scale)
            )
            self.X_history.append(s.copy())
            self.y_history.append(f)

            worst_idx = np.argmax(self.fitness)
            if self.config.semantic_commit_policy == "incumbent_improvement":
                accept_threshold = prev_best
            else:
                accept_threshold = self.fitness[worst_idx]
                if self.rugged_mode and progress >= 0.25:
                    accept_threshold = min(
                        accept_threshold, float(np.percentile(self.fitness, 80))
                    )
            if f < accept_threshold:
                population_accepted_count += 1
                population_accepted_vectors.append(s.copy())
                self.population[worst_idx] = s
                self.fitness[worst_idx] = f
                if f < prev_best:
                    best_update_count += 1
                    best_update_vectors.append(s.copy())
                    self.best_fit = f
                    self.best_idx = worst_idx
                    self.last_generation_best = self.best_fit
                    self.no_improve_generations = 0

        if event is not None:
            event["evaluated_candidates"] += int(evaluated_count)
            event["population_accepted_candidates"] += int(population_accepted_count)
            event["best_update_candidates"] += int(best_update_count)
            event["evaluated_candidate_vectors"] = [
                np.asarray(candidate, dtype=float).tolist() for candidate in evaluated_vectors
            ]
            event["evaluated_candidate_objectives"] = [
                float(value) for value in evaluated_objectives
            ]
            event["evaluated_candidate_regrets"] = [
                float(value) for value in evaluated_regrets
            ]
            event["population_accepted_candidate_vectors"] = [
                np.asarray(candidate, dtype=float).tolist()
                for candidate in population_accepted_vectors
            ]
            event["best_update_candidate_vectors"] = [
                np.asarray(candidate, dtype=float).tolist() for candidate in best_update_vectors
            ]

        if len(to_evaluate) == 0:
            if eval_budget <= 0:
                self._close_admission_event("evaluation_budget_zero")
            elif event is not None and event["surrogate_screened_out"] > 0:
                self._close_admission_event("surrogate_screened_all")
            else:
                self._close_admission_event("selection_empty")
        elif evaluated_count == 0:
            self._close_admission_event("evaluation_budget_exhausted")
        elif best_update_count > 0:
            self._close_admission_event("best_updated")
        elif population_accepted_count > 0:
            self._close_admission_event("population_accepted_without_best_update")
        else:
            self._close_admission_event("evaluated_not_population_accepted")

    def _micro_refine(self, objective_function: Callable, bounds: List[Tuple[float, float]]):
        if self.best_idx is None:
            return
        remaining = self.config.max_evaluations - self.evaluations
        if remaining <= 0:
            return
        steps = min(self.config.micro_refine_steps, remaining)
        x_best = self.population[self.best_idx].copy()
        ranges = np.array([b[1] - b[0] for b in bounds])
        sigma = self.config.micro_refine_sigma
        for _ in range(steps):
            if self.evaluations >= self.config.max_evaluations:
                break
            perturb = np.random.normal(0.0, sigma, size=len(bounds)) * ranges
            cand = np.clip(x_best + perturb, [b[0] for b in bounds], [b[1] for b in bounds])
            f = float(objective_function(cand))
            self.evaluations += 1
            self.X_history.append(cand.copy())
            self.y_history.append(f)
            if f < self.best_fit:
                self.best_fit = f
                self.population[self.best_idx] = cand
                self.fitness[self.best_idx] = f
                x_best = cand

    def _local_refine(
        self,
        x0: np.ndarray,
        objective_function: Callable,
        bounds: List[Tuple[float, float]] = None,
        x0_value: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        remaining = self.config.max_evaluations - self.evaluations
        if remaining < 5:
            baseline = float(x0_value) if x0_value is not None else float(objective_function(x0))
            return x0, baseline
        
        # Adjust intensity based on mode
        if self.commander_mode == STRATEGY_PRECISION:
            max_iter = min(200, remaining)
            tol = 1e-15
        else:
            max_iter = min(50, remaining)
            tol = 1e-9
        
        best_x = np.array(x0, copy=True)
        best_f = float(x0_value) if x0_value is not None else float("inf")

        def wrapped_obj(x):
            nonlocal best_x, best_f
            if self.evaluations >= self.config.max_evaluations: raise StopIteration
            val = objective_function(x)
            self.evaluations += 1
            self.X_history.append(x.copy())
            self.y_history.append(val)
            if val < best_f:
                best_x = np.array(x, copy=True)
                best_f = float(val)
            return val
            
        try:
            # Use L-BFGS-B with bounds if provided
            if bounds:
                res = minimize(wrapped_obj, x0, method='L-BFGS-B', bounds=bounds, options={'maxfun': max_iter, 'ftol': tol, 'gtol': tol, 'disp': False})
            else:
                res = minimize(wrapped_obj, x0, method='L-BFGS-B', options={'maxfun': max_iter, 'ftol': tol, 'gtol': tol, 'disp': False})

            if res.fun == 0.0:
                print(f"DEBUG: minimize found 0.0 at x={res.x}")
                val = objective_function(res.x)
                print(f"DEBUG: Re-eval objective(x) = {val}")

            if float(res.fun) < best_f:
                best_x = np.array(res.x, copy=True)
                best_f = float(res.fun)
            return best_x, best_f
        except StopIteration:
            if np.isfinite(best_f):
                return best_x, best_f
            if x0_value is not None:
                return x0, float(x0_value)
            return x0, float("inf")
        except Exception as exc:
            print(f"DEBUG: _local_refine failed with {exc!r}")
            if np.isfinite(best_f):
                return best_x, best_f
            if x0_value is not None:
                return x0, float(x0_value)
            return x0, float("inf")

    def configure_one_shot_directional_injection(
        self,
        direction_delta: np.ndarray,
        alpha: float,
        diversity_floor: float = 0.95,
        authorized_evaluator: Optional[Callable] = None,
    ) -> None:
        delta = np.asarray(direction_delta, dtype=float)
        if delta.shape != (self.config.dimension,) or not np.all(np.isfinite(delta)):
            raise ValueError("directional injection delta must be finite and match the problem dimension")
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("directional injection alpha must be in (0, 1]")
        if not 0.0 < float(diversity_floor) <= 1.0:
            raise ValueError("directional diversity floor must be in (0, 1]")
        self.pending_directional_delta = delta.copy()
        self.pending_directional_alpha = float(alpha)
        self.pending_directional_diversity_floor = float(diversity_floor)
        self.directional_injection_evaluator = authorized_evaluator
        self.last_directional_injection = {"configured": True, "applied": False, "accepted": False}

    def _evolve(self, bounds: List[Tuple[float, float]], objective_function: Callable):
        dim = len(bounds)
        
        # Linear Population Size Reduction (LPSR)
        current_evals = min(self.evaluations, self.config.max_evaluations)
        new_pop_size = round((self.pop_size_min - self.pop_size_initial) * (current_evals / self.config.max_evaluations) + self.pop_size_initial)
        new_pop_size = max(self.pop_size_min, new_pop_size)
        
        if new_pop_size < len(self.population):
            sorted_indices = np.argsort(self.fitness)
            keep_indices = sorted_indices[:new_pop_size]
            self.population = self.population[keep_indices]
            self.fitness = self.fitness[keep_indices]
            self.best_idx = int(np.argmin(self.fitness))
            # Maintain archive size roughly
            if len(self.archive) > new_pop_size:
                random.shuffle(self.archive)
                self.archive = self.archive[:new_pop_size]

        pop_size = len(self.population)
        
        # Strategy Execution
        
        # 1. PRECISION MODE: Run L-BFGS-B on best, then tight DE on rest
        if self.commander_mode == STRATEGY_PRECISION:
            should_refine = (
                self.runtime_sniper_enabled
                and
                self._allow_sniper_refine(self.evaluations / max(1, self.config.max_evaluations))
                and (self.evaluations - self.last_precision_refine_eval) >= max(10, pop_size // 2)
            )
            if should_refine:
                refined_x, refined_f = self._local_refine(
                    self.population[self.best_idx],
                    objective_function,
                    bounds,
                    x0_value=self.best_fit,
                )
                self.last_precision_refine_eval = int(self.evaluations)
                if refined_f < self.best_fit:
                    self.population[self.best_idx] = refined_x
                    self.fitness[self.best_idx] = refined_f
                    self.best_fit = refined_f
            
            # For the rest, use very small mutation (exploitation)
            self.current_F_base = float(self.config.precision_target_f)
            self.current_CR_base = float(self.config.precision_target_cr)
        
        # 2. EMERGENCY MODE: Restart part of population
        elif self.commander_mode == STRATEGY_EMERGENCY and self.stagnation_counter > 0:
            # Aggressive Escape: Replace bottom 50% (was 30%) with random solutions
            # If very severe (stagnation > 15), replace 80% (Global Restart)
            replace_ratio = 0.12
            if self.stagnation_counter > 20:
                replace_ratio = 0.30
                print(f"[Field Commander] SEVERE STAGNATION ({self.stagnation_counter} gen) - GLOBAL RESTART ({int(replace_ratio*100)}%)")
            else:
                print(f"[Field Commander] Emergency Restart Triggered ({int(replace_ratio*100)}%)")

            sorted_indices = np.argsort(self.fitness)
            num_replace = int(replace_ratio * pop_size)
            
            for i in range(num_replace):
                idx = sorted_indices[-(i+1)]
                # Randomize within bounds
                self.population[idx] = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
                # Evaluate immediately
                self.fitness[idx] = float(objective_function(self.population[idx]))
                self.evaluations += 1
                if self.evaluations >= self.config.max_evaluations: break
            
            self.last_emergency_eval = int(self.evaluations)
            self.stagnation_counter = 0 # Reset after action
            self.commander_mode = STRATEGY_BALANCED # Reset mode to prevent infinite emergency loops

        # Standard DE/SHADE Loop
        S_CR, S_F, S_diff = [], [], []
        new_pop = self.population.copy()
        new_fitness = self.fitness.copy()
        sorted_indices = np.argsort(self.fitness)
        injection_delta = self.pending_directional_delta
        injection_target = int(np.argmin(self.fitness)) if injection_delta is not None else None
        
        # --- DOMINANCE GUARD ---
        # The last 15% of the population (minimum 4) are completely isolated from LLM's mode interference.
        # They run pure, unadulterated SHADE. This structurally guarantees we do not perform worse 
        # than standard mathematical baseline due to bad LLM parameter exploration.
        math_only_threshold = pop_size - max(4, int(pop_size * 0.15))
        
        for i in range(pop_size):
            if self.evaluations >= self.config.max_evaluations: break
            
            is_math_only = (i >= math_only_threshold)
            
            # Parameter Adaptation
            r_idx = np.random.randint(0, self.H)
            mu_f = self.M_F[r_idx]
            mu_cr = self.M_CR[r_idx]
            
            # Field Commander Influence (Adaptive Bias)
            if not is_math_only:
                if self.commander_mode == STRATEGY_EXPLORATION:
                    target_f, target_cr, alpha = 0.8, 0.2, 0.5
                elif self.commander_mode == STRATEGY_PRECISION:
                    target_f, target_cr, alpha = 0.1, 0.95, 0.8
                elif self.commander_mode == STRATEGY_EMERGENCY:
                    target_f, target_cr, alpha = 0.85, 0.25, 0.6
                else: # Balanced
                    target_f, target_cr, alpha = mu_f, mu_cr, 0.0 # No bias
                
                mu_f = (1-alpha)*mu_f + alpha*target_f
                mu_cr = (1-alpha)*mu_cr + alpha*target_cr
            
            # Generate F, CR
            cr = np.clip(np.random.normal(mu_cr, 0.1), 0, 1)
            f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
            while f <= 0: f = mu_f + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
            f = min(f, 1.0)
            
            # Mutation Strategy
            if not is_math_only and self.commander_mode == STRATEGY_EXPLORATION:
                # rand/1
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + f * (b - c)
            else:
                # current-to-pbest/1 (SHADE standard)
                p_best_idx = sorted_indices[np.random.randint(0, max(2, int(pop_size * 0.1)))] # Top 10%
                x_pbest = self.population[p_best_idx]
                r1, r2 = np.random.choice([idx for idx in range(pop_size) if idx != i], 2, replace=False)
                x_r1 = self.population[r1]
                
                # Use Archive for r2
                pop_archive = np.vstack((self.population, np.array(self.archive))) if len(self.archive) > 0 else self.population
                x_r2 = pop_archive[np.random.randint(0, len(pop_archive))]
                
                mutant = self.population[i] + f * (x_pbest - self.population[i]) + f * (x_r1 - x_r2)
            
            # Crossover
            trial = np.copy(self.population[i])
            jrand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < cr or j == jrand:
                    trial[j] = mutant[j]
                trial[j] = np.clip(trial[j], bounds[j][0], bounds[j][1])

            injection_applied = False
            if injection_delta is not None and i == injection_target:
                proposed = trial + self.pending_directional_alpha * injection_delta
                proposed = np.asarray([
                    np.clip(proposed[j], bounds[j][0], bounds[j][1]) for j in range(dim)
                ])
                ranges = np.asarray([max(1e-12, hi - lo) for lo, hi in bounds], dtype=float)
                diversity_before = float(np.mean(np.std(self.population, axis=0) / ranges))
                prospective = self.population.copy()
                prospective[i] = proposed
                diversity_after = float(np.mean(np.std(prospective, axis=0) / ranges))
                diversity_ratio = diversity_after / max(diversity_before, 1e-12)
                injection_applied = diversity_ratio >= self.pending_directional_diversity_floor
                if injection_applied:
                    trial = proposed
                self.last_directional_injection = {
                    "configured": True,
                    "applied": bool(injection_applied),
                    "accepted": False,
                    "alpha": float(self.pending_directional_alpha),
                    "diversity_before": diversity_before,
                    "diversity_after": diversity_after,
                    "diversity_ratio": float(diversity_ratio),
                    "target_index": int(i),
                    "evaluated_vector": proposed.copy().tolist(),
                }
                self.pending_directional_delta = None
                
            # Selection
            if injection_applied and self.directional_injection_evaluator is not None:
                f_trial = float(self.directional_injection_evaluator(trial, objective_function))
                self.directional_injection_evaluator = None
            else:
                f_trial = float(objective_function(trial))
                if injection_delta is not None and i == injection_target:
                    self.directional_injection_evaluator = None
            self.evaluations += 1
            self.X_history.append(trial.copy())
            self.y_history.append(f_trial)
            
            if f_trial < self.fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
                S_CR.append(cr)
                S_F.append(f)
                S_diff.append(abs(f_trial - self.fitness[i]))
                self.archive.append(self.population[i].copy())
                if len(self.archive) > pop_size: self.archive.pop(random.randint(0, len(self.archive)-1))
                if injection_applied:
                    self.last_directional_injection["accepted"] = True
            else:
                new_pop[i] = self.population[i]
                new_fitness[i] = self.fitness[i]
                
            if f_trial < self.best_fit:
                self.best_fit = f_trial
                self.best_idx = i
                
        self.population = new_pop
        self.fitness = new_fitness
        
        # Update Memory (SHADE)
        if S_CR:
            w = np.array(S_diff) / (np.sum(S_diff) + 1e-15)
            m_cr = np.sum(w * np.array(S_CR))
            m_f = np.sum(w * (np.array(S_F)**2)) / (np.sum(w * np.array(S_F)) + 1e-15)
            self.M_CR[self.k] = m_cr
            self.M_F[self.k] = m_f
            self.k = (self.k + 1) % self.H
