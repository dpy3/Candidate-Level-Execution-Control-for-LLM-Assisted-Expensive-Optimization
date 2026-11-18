import os
import json
import time
from typing import List, Dict

from safe_llm_saea.paper_ready_safe_llm_saea import SafeLLMSAEA, ExperimentConfig
from safe_llm_saea.cec2022 import get_function, get_bounds


def run_ablation(fid_list: List[int], dims: List[int], runs: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for dim in dims:
        for fid in fid_list:
            f = get_function(fid, dim)
            lo, hi = get_bounds(dim)
            bounds = [(lo, hi)] * dim
            variants = {
                "no_llm": dict(llm_enhancement=False),
                "no_gating": dict(gating_enabled=False),
                "no_fusion": dict(fusion_weight=0.0),
                "no_diversity": dict(diversity_weight=0.0),
            }
            key = f"F{fid:02d}_{dim}D"
            out[key] = {}
            for name, patch in variants.items():
                cfg = ExperimentConfig(
                    algorithm_name="Safe-LLM-SAEA",
                    problem_name=key,
                    dimension=dim,
                    max_evaluations=20 * (500 + 1),
                    population_size=20,
                    runs=runs,
                    llm_enhancement=True,
                    gating_enabled=True,
                    max_llm_calls_per_run=5,
                    max_llm_cost_usd_per_run=0.5,
                    fusion_weight=0.75 if dim == 10 else 0.7,
                    de_F=0.5,
                    de_CR=0.9,
                    min_remaining_budget=50,
                    min_stagnation_generations=2,
                    min_uncertainty_threshold=0.005,
                    min_call_interval=10,
                    min_confidence=0.5,
                    enable_stage2=True,
                    stage2_stagnation_generations=10,
                    stage2_eval_ratio_threshold=0.5,
                    stage2_fusion_weight_boost=0.05,
                    stage2_call_interval_factor=0.7,
                    stage2_F_factor=0.9,
                    stage2_CR_factor=1.05,
                    consistency_calls=2,
                    consistency_threshold=0.15,
                    schema_strict=True,
                    mark_call_on_failure=True,
                    force_initial_llm_call=True,
                    max_llm_calls_stage1=3,
                    max_llm_calls_stage2=2,
                    llm_suggestions_k=3,
                    diversity_weight=0.15 if dim == 10 else 0.2,
                )
                for k, v in patch.items():
                    setattr(cfg, k, v)
                runs_res = []
                for _ in range(runs):
                    algo = SafeLLMSAEA(cfg, use_api=False)
                    res = algo.optimize(f, bounds)
                    runs_res.append(res)
                out[key][name] = runs_res
    return out


def run_sensitivity(fid_list: List[int], dims: List[int], runs: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    fw_list = [0.6, 0.7, 0.75]
    interval_list = [8, 10, 12]
    cons_calls_list = [2, 3]
    sug_k_list = [2, 3, 4]
    div_w_list = [0.0, 0.1, 0.2]
    for dim in dims:
        for fid in fid_list:
            f = get_function(fid, dim)
            lo, hi = get_bounds(dim)
            bounds = [(lo, hi)] * dim
            key = f"F{fid:02d}_{dim}D"
            out[key] = {}
            for fw in fw_list:
                for ci in interval_list:
                    for cc in cons_calls_list:
                        for sk in sug_k_list:
                            for dw in div_w_list:
                                cfg = ExperimentConfig(
                                    algorithm_name="Safe-LLM-SAEA",
                                    problem_name=key,
                                    dimension=dim,
                                    max_evaluations=20 * (500 + 1),
                                    population_size=20,
                                    runs=runs,
                                    llm_enhancement=True,
                                    gating_enabled=True,
                                    max_llm_calls_per_run=5,
                                    max_llm_cost_usd_per_run=0.5,
                                    fusion_weight=fw,
                                    de_F=0.5,
                                    de_CR=0.9,
                                    min_remaining_budget=50,
                                    min_stagnation_generations=2,
                                    min_uncertainty_threshold=0.005,
                                    min_call_interval=ci,
                                    min_confidence=0.5,
                                    enable_stage2=True,
                                    stage2_stagnation_generations=10,
                                    stage2_eval_ratio_threshold=0.5,
                                    stage2_fusion_weight_boost=0.05,
                                    stage2_call_interval_factor=0.7,
                                    stage2_F_factor=0.9,
                                    stage2_CR_factor=1.05,
                                    consistency_calls=cc,
                                    consistency_threshold=0.15,
                                    schema_strict=True,
                                    mark_call_on_failure=True,
                                    force_initial_llm_call=True,
                                    max_llm_calls_stage1=3,
                                    max_llm_calls_stage2=2,
                                    llm_suggestions_k=sk,
                                    diversity_weight=dw,
                                )
                                runs_res = []
                                for _ in range(runs):
                                    algo = SafeLLMSAEA(cfg, use_api=False)
                                    res = algo.optimize(f, bounds)
                                    runs_res.append(res)
                                out[key][f"fw_{fw}_ci_{ci}_cc_{cc}_sk_{sk}_dw_{dw}"] = runs_res
    return out


def main():
    ablation = run_ablation([6, 9, 12], [10, 20], runs=3)
    sens = run_sensitivity([6, 9, 12], [10, 20], runs=2)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("safe_llm_saea", "lightweight_paper_results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"ablation_{ts}.json"), "w", encoding="utf-8") as fp:
        json.dump(ablation, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"sensitivity_{ts}.json"), "w", encoding="utf-8") as fp:
        json.dump(sens, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
