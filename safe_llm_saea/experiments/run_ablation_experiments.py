
import os
import argparse
import multiprocessing
import time
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from safe_llm_saea.paper_ready_safe_llm_saea import SafeLLMSAEA, ExperimentConfig, get_engineering_problem

def run_single_ablation_task(task):
    """
    Runs a single ablation task.
    variant: "SAFE-LLM-SAEA" (Full), "No-GP" (LLM only), "No-LLM" (DE baseline)
    """
    problem_name, dim, run_id, variant, max_evals = task
    
    # Define problem
    problem_data = get_engineering_problem(problem_name)
    if not problem_data:
        return {"status": "error", "error": f"Unknown problem {problem_name}"}
        
    func = problem_data["function"]
    bounds = problem_data["bounds"]
    
    # Safe objective wrapper
    def objective(x):
        try:
            val = float(func(x))
            if np.isnan(val) or np.isinf(val):
                return 1e15
            return val
        except:
            return 1e15

    # Configure based on variant
    llm_enhancement = True
    gating_enabled = True
    algo_name = variant
    
    if variant == "No-GP":
        gating_enabled = False
    elif variant == "No-LLM":
        llm_enhancement = False
        # No-LLM is essentially DE, but running through same codebase for fair comparison
    
    config = ExperimentConfig(
        algorithm_name=algo_name,
        problem_name=problem_name,
        dimension=dim,
        max_evaluations=max_evals,
        population_size=20, # Standard pop size
        runs=1,
        llm_enhancement=llm_enhancement,
        gating_enabled=gating_enabled,
        max_llm_calls_per_run=20, # Cap costs
        max_llm_cost_usd_per_run=1.0,
        fusion_weight=0.8,
        de_F=0.5,
        de_CR=0.9,
        min_remaining_budget=50,
        min_stagnation_generations=5,
        min_uncertainty_threshold=0.3,
        min_call_interval=3,
        min_confidence=0.7,
        enable_stage2=True,
        stage2_stagnation_generations=5,
        stage2_eval_ratio_threshold=0.6,
        stage2_fusion_weight_boost=0.3,
        stage2_call_interval_factor=2.0,
        stage2_F_factor=0.5,
        stage2_CR_factor=0.9,
        consistency_calls=3,
        consistency_threshold=1e-4,
        schema_strict=True,
        mark_call_on_failure=True,
        force_initial_llm_call=True,
        max_llm_calls_stage1=10,
        max_llm_calls_stage2=10
    )
    
    # API Key handling (mock or real)
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    
    start_time = time.time()
    try:
        algo = SafeLLMSAEA(config=config, api_key=api_key)
        result = algo.optimize(objective_function=objective, bounds=bounds)
        duration = time.time() - start_time
        
        # Add metadata
        result["variant"] = variant
        result["problem"] = problem_name
        result["run_id"] = run_id
        result["duration"] = duration
        
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e), "task": task}

def main():
    parser = argparse.ArgumentParser(description='Run Ablation Studies')
    parser.add_argument('--problems', nargs='+', default=["pressure_vessel", "welded_beam", "speed_reducer"], help='Problems to run')
    parser.add_argument('--variants', nargs='+', default=["SAFE-LLM-SAEA", "No-GP", "No-LLM"], help='Variants to run')
    parser.add_argument('--runs', type=int, default=30, help='Runs per problem/variant')
    parser.add_argument('--workers', type=int, default=8, help='Parallel workers')
    args = parser.parse_args()

    max_evals = 3000 # Strict Q1 Standard
    
    tasks = []
    for prob in args.problems:
        # Get dim for this problem
        p_data = get_engineering_problem(prob)
        dim = p_data["dimension"]
        
        for var in args.variants:
            for r in range(args.runs):
                tasks.append((prob, dim, r, var, max_evals))
                
    print(f"🚀 Launching Ablation Study: {len(tasks)} tasks")
    print(f"   Problems: {args.problems}")
    print(f"   Variants: {args.variants}")
    print(f"   Workers: {args.workers}")
    
    results_store = {}
    
    # Output setup
    out_dir = Path(__file__).resolve().parents[1] / "ablation_results"
    os.makedirs(out_dir, exist_ok=True)
    json_path = out_dir / "ablation_results.json"
    
    completed = 0
    with multiprocessing.Pool(args.workers) as pool:
        for res in pool.imap_unordered(run_single_ablation_task, tasks):
            completed += 1
            if res["status"] == "success":
                r = res["result"]
                key = f"{r['problem']}_{r['variant']}"
                if key not in results_store:
                    results_store[key] = []
                results_store[key].append(r)
                print(f"[{completed}/{len(tasks)}] ✅ {r['problem']} ({r['variant']}) Run {r['run_id']} Best: {r['best_fitness']:.2e}")
            else:
                print(f"[{completed}/{len(tasks)}] ❌ Error: {res['error']}")
                
    # Save final results
    import json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(json_path, 'w') as f:
        json.dump(results_store, f, cls=NumpyEncoder, indent=2)
    print(f"🎉 All done! Results saved to {json_path}")

if __name__ == "__main__":
    main()
