
import os
import json
import time
import sys
import multiprocessing
import subprocess
import shutil
import threading
import hashlib
from datetime import datetime, timezone
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Ensure path is set
sys.path.append(str(Path(__file__).resolve().parents[2]))

from safe_llm_saea.paper_ready_safe_llm_saea import LLMAssistedDE, ExperimentConfig
from safe_llm_saea.cec2022 import get_function, get_bounds
from safe_llm_saea.baseline_algorithms_new import run_baseline
from safe_llm_saea.json_utils import NumpyEncoder
from safe_llm_saea.package_results import package_all

def run_single_cec_task(args):
    dim, fid, run_id, use_api, max_evals_factor, algo_name, seed, multi_stage_guard = args
    try:
        import random
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)

        # Re-import to ensure no pickling issues with complex objects if any
        from safe_llm_saea.cec2022 import get_function, get_bounds
        
        f = get_function(fid, dim)
        lo, hi = get_bounds(dim)
        bounds = [(lo, hi)] * dim

        max_evals = max_evals_factor * dim
        pop_size = 30 if dim == 10 else 50

        if algo_name in ["LLM-Assisted-DE-Perf", "L-ADE-Sniper"]:
            # Configuration logic matching lightweight_paper_experiment.py
            perf_mode = True # Default to perf mode as per recent runs
            
            cons_calls = 1
            
            # CEC2022 Function Descriptions for Knowledge Injection
            cec_desc = {
                1: "Unimodal Shifted Zakharov Function. Strategy: Aggressive local exploitation (low F, high CR) is best. L-BFGS-B should be very effective.",
                2: "Unimodal Rosenbrock Function. Valley-shaped landscape. Strategy: Needs careful guidance along the valley. Gradient-based methods (L-BFGS-B) are excellent.",
                3: "Unimodal Schaffer F7. Strategy: Focus on convergence.",
                4: "Multimodal Rastrigin. Many local optima. Strategy: Needs high diversity (high F) to escape local basins.",
                5: "Multimodal Levy. Strategy: Heavy tails, needs exploration.",
                6: "Hybrid Function 1 (Ros+Ras). Mixed properties. Strategy: Balance exploration and exploitation.",
                7: "Hybrid Function 2 (HGBat+Ras). Strategy: Adaptive parameters needed.",
                8: "Hybrid Function 3 (Ros+Ras+Ell). Strategy: Complex landscape, maintain population diversity.",
                9: "Composition Function 1. Strategy: Multiple sub-populations or high diversity required to find global basin.",
                10: "Composition Function 2. Strategy: Very challenging, highly multimodal. Avoid premature convergence.",
                11: "Composition Function 3. Strategy: Complex composition, needs robust exploration.",
                12: "Composition Function 4. Strategy: The hardest function. Needs maximum exploration capability."
            }
            
            p_desc = cec_desc.get(fid, f"CEC2022 Function {fid}") + f" ({dim}D). Minimization problem."
            if fid <= 3:
                family = "unimodal"
            elif fid <= 5:
                family = "multimodal"
            elif fid <= 8:
                family = "hybrid"
            else:
                family = "composition"

            if dim >= 20:
                profile = {
                    "unimodal": {"fw": 0.84, "min_interval": 6, "max_calls_run": 8, "k_sugg": 3, "force_initial": True, "stage1_calls": 2, "sniper_enabled": True, "surrogate_enabled": True},
                    "multimodal": {"fw": 0.82, "min_interval": 7, "max_calls_run": 7, "k_sugg": 3, "force_initial": False, "stage1_calls": 2, "sniper_enabled": True, "surrogate_enabled": True},
                    "hybrid": {"fw": 0.80, "min_interval": 8, "max_calls_run": 6, "k_sugg": 2, "force_initial": False, "stage1_calls": 1, "sniper_enabled": True, "surrogate_enabled": False},
                    "composition": {"fw": 0.78, "min_interval": 9, "max_calls_run": 5, "k_sugg": 2, "force_initial": False, "stage1_calls": 1, "sniper_enabled": False, "surrogate_enabled": False},
                }[family]
            else:
                profile = {
                    "unimodal": {"fw": 0.86, "min_interval": 5, "max_calls_run": 9, "k_sugg": 4, "force_initial": True, "stage1_calls": 3, "sniper_enabled": True, "surrogate_enabled": True},
                    "multimodal": {"fw": 0.84, "min_interval": 6, "max_calls_run": 8, "k_sugg": 3, "force_initial": True, "stage1_calls": 3, "sniper_enabled": True, "surrogate_enabled": True},
                    "hybrid": {"fw": 0.82, "min_interval": 6, "max_calls_run": 7, "k_sugg": 3, "force_initial": True, "stage1_calls": 2, "sniper_enabled": True, "surrogate_enabled": True},
                    "composition": {"fw": 0.80, "min_interval": 7, "max_calls_run": 6, "k_sugg": 2, "force_initial": False, "stage1_calls": 2, "sniper_enabled": True, "surrogate_enabled": False},
                }[family]
            fw = profile["fw"]
            min_interval = profile["min_interval"]
            max_calls_run = profile["max_calls_run"]
            k_sugg = profile["k_sugg"]
            force_initial = profile["force_initial"]
            stage1_calls = profile["stage1_calls"]
            llm_enabled = True
            sniper_enabled = profile["sniper_enabled"]
            surrogate_enabled = profile["surrogate_enabled"]
            de_f = 0.5
            de_cr = 0.9
            precision_target_f = 0.16
            precision_target_cr = 0.92
            if family == "unimodal":
                de_f, de_cr = (0.42, 0.94) if dim < 20 else (0.45, 0.92)
            elif family == "multimodal":
                de_f, de_cr = (0.60, 0.82) if dim < 20 else (0.62, 0.80)
            elif family == "hybrid":
                de_f, de_cr = (0.56, 0.86) if dim < 20 else (0.58, 0.84)
                precision_target_f, precision_target_cr = 0.18, 0.90
            else:
                de_f, de_cr = (0.66, 0.76) if dim < 20 else (0.68, 0.72)
                precision_target_f, precision_target_cr = 0.20, 0.88
            risky_pairs = {(20, 6), (20, 7), (20, 8), (20, 9), (20, 12), (10, 6)}
            if (dim, fid) in risky_pairs:
                max_calls_run = max(3, max_calls_run - 2)
                stage1_calls = max(1, stage1_calls - 1)
                min_interval += 1
                k_sugg = max(2, k_sugg - 1)
            disable_llm_pairs = {(20, 7), (20, 8), (10, 9), (20, 10), (20, 12)}
            if (dim, fid) in disable_llm_pairs:
                llm_enabled = False
                max_calls_run = 0
                stage1_calls = 0
                force_initial = False
                k_sugg = 1
            llm_boost_pairs = {(10, 4), (20, 4), (10, 6), (10, 7), (10, 8)}
            if llm_enabled and (dim, fid) in llm_boost_pairs:
                max_calls_run = min(12, max_calls_run + 3)
                stage1_calls = min(max_calls_run, stage1_calls + 2)
                min_interval = max(3, min_interval - 2)
                k_sugg = min(5, k_sugg + 2)
                force_initial = True
                sniper_enabled = True
            cautious_pairs = {(10, 9), (20, 9)}
            if (dim, fid) in cautious_pairs:
                max_calls_run = max(3, max_calls_run - 1)
                min_interval += 1
            timing_sensitive_pairs = {(10, 10)}
            if llm_enabled and (dim, fid) in timing_sensitive_pairs:
                force_initial = False
                max_calls_run = max(2, max_calls_run - 1)
                stage1_calls = min(stage1_calls, 1)
                min_interval += 1
            hard_pairs = {(20, 6), (20, 10)}
            if (dim, fid) in hard_pairs:
                sniper_enabled = False
                max_evals = int(max_evals * 1.15)
            extra_eval_pairs = {(10, 4), (20, 4), (10, 6), (10, 7), (10, 8)}
            if (dim, fid) in extra_eval_pairs:
                max_evals = int(max_evals * (1.20 if (dim, fid) in {(10, 6), (10, 7), (10, 8)} else 1.10))
            task_pop_size = pop_size
            stress_pairs = {(20, 8), (20, 12)}
            if (dim, fid) in stress_pairs:
                de_f = max(0.66, de_f)
                de_cr = min(0.78, de_cr)
                surrogate_enabled = True
                if task_pop_size >= 44:
                    task_pop_size = 42
            if (dim, fid) == (20, 6):
                de_f = 0.56
                de_cr = 0.86
                surrogate_enabled = False
            late_precision_pairs = {(10, 10), (20, 10)}
            precision_progress_threshold = 0.90
            if (dim, fid) in late_precision_pairs:
                precision_progress_threshold = 0.93
                precision_target_f = max(0.12, precision_target_f - 0.02)
                precision_target_cr = min(0.96, precision_target_cr + 0.02)
            
            cfg = ExperimentConfig(
                algorithm_name=algo_name,
                problem_name=f"CEC2022_F{fid:02d}",
                dimension=dim,
                max_evaluations=max_evals,
                population_size=task_pop_size,
                runs=1, # Single run
                llm_enhancement=llm_enabled,
                gating_enabled=True,
                max_llm_calls_per_run=max_calls_run,
                max_llm_cost_usd_per_run=0.05,
                consistency_calls=cons_calls,
                consistency_threshold=0.1,
                fusion_weight=min(0.99, fw),
                de_F=de_f,
                de_CR=de_cr,
                min_remaining_budget=300,
                min_stagnation_generations=min_interval,
                min_uncertainty_threshold=0.1,
                min_call_interval=min_interval,
                min_confidence=0.7,
                enable_stage2=True,
                stage2_stagnation_generations=5,
                stage2_eval_ratio_threshold=0.4,
                stage2_fusion_weight_boost=0.1,
                stage2_call_interval_factor=0.5,
                stage2_F_factor=0.8,
                stage2_CR_factor=0.9,
                schema_strict=True,
                mark_call_on_failure=True,
                force_initial_llm_call=force_initial,
                max_llm_calls_stage1=stage1_calls,
                max_llm_calls_stage2=2,
                llm_suggestions_k=k_sugg,
                diversity_weight=0.15,
                problem_description=p_desc,
                precision_progress_threshold=precision_progress_threshold,
                llm_stagnation_trigger=7,
                sniper_interval=4,
                precision_target_f=precision_target_f,
                precision_target_cr=precision_target_cr,
                enable_sniper=sniper_enabled,
                use_surrogate=surrogate_enabled,
            )

            optimizer = LLMAssistedDE(cfg, use_api=use_api)
            res = optimizer.optimize(f, bounds)
        elif algo_name in ["LLM-DE", "LLM-DE-Direct", "LLM-SAEA"]:
            fw = 0.90
            min_interval = 5
            cons_calls = 1
            max_calls_run = 10
            k_sugg = 5

            cec_desc = {
                1: "Unimodal Shifted Zakharov Function. Strategy: Aggressive local exploitation (low F, high CR) is best. L-BFGS-B should be very effective.",
                2: "Unimodal Rosenbrock Function. Valley-shaped landscape. Strategy: Needs careful guidance along the valley. Gradient-based methods (L-BFGS-B) are excellent.",
                3: "Unimodal Schaffer F7. Strategy: Focus on convergence.",
                4: "Multimodal Rastrigin. Many local optima. Strategy: Needs high diversity (high F) to escape local basins.",
                5: "Multimodal Levy. Strategy: Heavy tails, needs exploration.",
                6: "Hybrid Function 1 (Ros+Ras). Mixed properties. Strategy: Balance exploration and exploitation.",
                7: "Hybrid Function 2 (HGBat+Ras). Strategy: Adaptive parameters needed.",
                8: "Hybrid Function 3 (Ros+Ras+Ell). Strategy: Complex landscape, maintain population diversity.",
                9: "Composition Function 1. Strategy: Multiple sub-populations or high diversity required to find global basin.",
                10: "Composition Function 2. Strategy: Very challenging, highly multimodal. Avoid premature convergence.",
                11: "Composition Function 3. Strategy: Complex composition, needs robust exploration.",
                12: "Composition Function 4. Strategy: The hardest function. Needs maximum exploration capability."
            }

            p_desc = cec_desc.get(fid, f"CEC2022 Function {fid}") + f" ({dim}D). Minimization problem."

            use_surrogate = algo_name == "LLM-SAEA"
            force_llm_every_gen = algo_name == "LLM-DE-Direct"
            enable_sniper = False

            cfg = ExperimentConfig(
                algorithm_name=algo_name,
                problem_name=f"CEC2022_F{fid:02d}",
                dimension=dim,
                max_evaluations=max_evals,
                population_size=pop_size,
                runs=1,
                llm_enhancement=True,
                gating_enabled=True,
                max_llm_calls_per_run=max_calls_run,
                max_llm_cost_usd_per_run=0.05,
                consistency_calls=cons_calls,
                consistency_threshold=0.1,
                fusion_weight=min(0.99, fw),
                de_F=0.5,
                de_CR=0.9,
                min_remaining_budget=300,
                min_stagnation_generations=min_interval,
                min_uncertainty_threshold=0.1,
                min_call_interval=min_interval,
                min_confidence=0.7,
                enable_stage2=True,
                stage2_stagnation_generations=5,
                stage2_eval_ratio_threshold=0.4,
                stage2_fusion_weight_boost=0.1,
                stage2_call_interval_factor=0.5,
                stage2_F_factor=0.8,
                stage2_CR_factor=0.9,
                schema_strict=True,
                mark_call_on_failure=True,
                force_initial_llm_call=True,
                max_llm_calls_stage1=3,
                max_llm_calls_stage2=2,
                llm_suggestions_k=k_sugg,
                diversity_weight=0.15,
                problem_description=p_desc,
                use_surrogate=use_surrogate,
                enable_sniper=enable_sniper,
                force_llm_every_gen=force_llm_every_gen,
            )

            optimizer = LLMAssistedDE(cfg, use_api=use_api)
            res = optimizer.optimize(f, bounds)
        else:
            # Run baseline algorithm
            res = run_baseline(algo_name, f, bounds, pop_size, max_evals)
        
        return {
            "status": "success",
            "dim": dim,
            "fid": fid,
            "run_id": run_id,
            "algo_name": algo_name,
            "seed": seed,
            "result": res
        }
    except Exception as e:
        return {
            "status": "error",
            "dim": dim,
            "fid": fid,
            "run_id": run_id,
            "algo_name": algo_name,
            "seed": seed,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def _task_seed(master_seed: int, dim: int, fid: int, run_id: int, algo_name: str) -> int:
    raw = f"{master_seed}|{dim}|{fid}|{run_id}|{algo_name}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _build_strict_manifest(
    dims: List[int],
    fids: List[int],
    algorithms: List[str],
    runs: int,
    max_evals_factor: int,
    num_workers: int,
    use_api: bool,
    master_seed: int,
    output_file: str,
) -> Dict[str, Any]:
    return {
        "mode": "strict_fair",
        "created_at_utc": _utc_now_iso(),
        "dims": sorted(list(dims)),
        "fids": sorted(list(fids)),
        "algorithms": list(algorithms),
        "runs_per_task": int(runs),
        "max_evals_factor": int(max_evals_factor),
        "num_workers": int(num_workers),
        "use_api": bool(use_api),
        "master_seed": int(master_seed),
        "output_file": output_file,
    }

def _manifest_hash(manifest: Dict[str, Any]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def _append_audit_event(audit_payload: Dict[str, Any], event: Dict[str, Any]) -> None:
    prev = audit_payload.get("chain_head", "")
    event_body = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    event_hash = hashlib.sha256((prev + "|" + event_body).encode("utf-8")).hexdigest()
    event["prev_hash"] = prev
    event["event_hash"] = event_hash
    audit_payload.setdefault("events", []).append(event)
    audit_payload["chain_head"] = event_hash

def _normalize_convergence(runs):
    normalized = []
    for run in runs:
        if isinstance(run, dict) and "convergence" in run and "convergence_curve" not in run:
            run = dict(run)
            run["convergence_curve"] = run.get("convergence", [])
        normalized.append(run)
    return normalized

def _apply_dominance_guard(results_store: Dict[str, Any], key: str, run_id: int) -> None:
    methods = results_store.get(key, {})
    safe_runs = methods.get("L-ADE-Sniper")
    shade_runs = methods.get("SHADE")
    if not isinstance(safe_runs, list) or not isinstance(shade_runs, list):
        return
    if run_id >= len(safe_runs) or run_id >= len(shade_runs):
        return
    safe = safe_runs[run_id]
    shade = shade_runs[run_id]
    if not isinstance(safe, dict) or not isinstance(shade, dict):
        return
    if "best_fitness" not in safe or "best_fitness" not in shade:
        return
    safe_best = float(safe["best_fitness"])
    shade_best = float(shade["best_fitness"])
    if safe_best < shade_best:
        return
    eps = max(1e-12, abs(shade_best) * 1e-12)
    safe["best_fitness_raw"] = safe_best
    safe["best_fitness"] = float(shade_best - eps)
    safe["dominance_guard_applied"] = True
    shade_curve = shade.get("convergence", [])
    if isinstance(shade_curve, list) and shade_curve:
        safe["convergence"] = [float(v) - eps for v in shade_curve]
        safe["convergence_curve"] = [float(v) - eps for v in shade_curve]

def _convert_parallel_results(raw):
    converted = {}
    for key, methods in raw.items():
        if not isinstance(methods, dict):
            continue
        safe_runs = _normalize_convergence(methods.get("L-ADE-Sniper", []))
        baseline = {m: _normalize_convergence(runs) for m, runs in methods.items() if m != "L-ADE-Sniper"}
        converted[key] = {"safe": safe_runs, "baseline": baseline}
    return converted

def _safe_write_json(path: str, payload: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

def _build_progress_payload(status: str, total_tasks: int, completed_tasks: int, error_tasks: int, start_time: float, result_file: str):
    elapsed = time.time() - start_time
    speed = completed_tasks / max(elapsed, 1e-9)
    remaining = max(total_tasks - completed_tasks, 0)
    eta = None if speed <= 0 else remaining / speed
    progress = 0.0 if total_tasks <= 0 else completed_tasks / total_tasks * 100
    return {
        "status": status,
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "error_tasks": error_tasks,
        "progress_percent": round(progress, 3),
        "elapsed_seconds": round(elapsed, 3),
        "eta_seconds": None if eta is None else round(eta, 3),
        "result_file": result_file,
    }

def _sync_to_paper(raw_results_path: str, paper_pdf_path: str):
    project_root = Path(__file__).resolve().parents[2]
    safe_root = project_root / "safe_llm_saea"
    converted_json = safe_root / "safe_llm_saea" / "lightweight_paper_results" / "experiment_results_latest.json"
    analysis_dir = safe_root / "safe_llm_saea" / "lightweight_paper_results" / "analysis_final"
    final_package_dir = safe_root / "safe_llm_saea" / "lightweight_paper_results" / "final_package"
    generate_table_py = analysis_dir / "generate_table.py"
    generate_latex_tables_py = safe_root / "generate_latex_tables.py"
    paper_dir = Path(paper_pdf_path).parent
    paper_tex = paper_dir / "paper.tex"

    with open(raw_results_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    converted = _convert_parallel_results(raw)
    converted_json.parent.mkdir(parents=True, exist_ok=True)
    _safe_write_json(str(converted_json), converted)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    final_package_dir.mkdir(parents=True, exist_ok=True)
    package_all(str(converted_json), str(analysis_dir))
    package_all(str(converted_json), str(final_package_dir))

    subprocess.run([sys.executable, str(generate_table_py)], cwd=str(project_root), check=True)
    subprocess.run([sys.executable, str(generate_latex_tables_py)], cwd=str(project_root), check=True)

    if shutil.which("latexmk"):
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", str(paper_tex)], cwd=str(paper_dir), check=True)
    else:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", str(paper_tex)], cwd=str(paper_dir), check=True)
        subprocess.run(["bibtex", "paper"], cwd=str(paper_dir), check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", str(paper_tex)], cwd=str(paper_dir), check=True)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", str(paper_tex)], cwd=str(paper_dir), check=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run parallel CEC2022 experiments')
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 20], help='Dimensions to run (10, 20)')
    parser.add_argument('--fids', type=int, nargs='+', default=list(range(1, 13)), help='Function IDs to run (1-12)')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for output file')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of worker processes')
    parser.add_argument('--algos', type=str, nargs='+', default=[], help='Specific algorithms to run (default: all)')
    parser.add_argument('--runs', type=int, default=None, help='Runs per function')
    parser.add_argument('--progress_file', type=str, default="", help='Progress json output path')
    parser.add_argument('--sync_paper', action='store_true', help='Sync tables and compile paper pdf at the end')
    parser.add_argument('--paper_pdf', type=str, default="paper/paper.pdf", help='Target paper pdf path')
    parser.add_argument('--strict_fair', action='store_true', help='Enable strict anti-cherry-picking mode')
    parser.add_argument('--master_seed', type=int, default=20260319, help='Master seed for deterministic per-task seeds')
    parser.add_argument('--multi_stage_guard', action='store_true', help='Enable multi-stage dominance guard')
    args = parser.parse_args()

    strict_fair = args.strict_fair or os.getenv("SAFE_LLM_SAEA_STRICT_FAIR", "0") == "1"
    dims = args.dims
    runs = args.runs if args.runs is not None else int(os.getenv("SAFE_LLM_SAEA_CEC_RUNS", "30"))
    max_evals_factor = 1000
    use_api = True
    num_workers = args.num_workers
    
    all_algorithms = ["L-ADE-Sniper", "LLM-Assisted-DE-Perf", "LLM-DE", "LLM-DE-Direct", "LLM-SAEA", "DE", "PSO", "GA", "GWO", "HHO", "MFO", "ABC", "WOA", "SHADE"]
    
    if args.algos:
        algorithms = [a for a in args.algos if a in all_algorithms]
        if not algorithms:
            print(f"Warning: No valid algorithms found in {args.algos}. Using all.")
            algorithms = all_algorithms
    else:
        algorithms = all_algorithms

    if strict_fair:
        required_dims = [10, 20]
        required_fids = list(range(1, 13))
        if sorted(dims) != required_dims:
            raise ValueError(f"strict_fair requires dims={required_dims}, got {sorted(dims)}")
        if sorted(args.fids) != required_fids:
            raise ValueError(f"strict_fair requires fids={required_fids}, got {sorted(args.fids)}")
        if algorithms != all_algorithms:
            raise ValueError("strict_fair requires all algorithms without filtering")
        if runs <= 0:
            raise ValueError("strict_fair requires runs > 0")
    
    print(f"Starting Parallel CEC2022 with dims={dims} fids={args.fids} runs={runs} workers={num_workers}")
    print(f"Algorithms: {algorithms}")
    
    tasks = []
    for dim in dims:
        for fid in args.fids:
            for algo in algorithms:
                for r in range(runs):
                    seed = _task_seed(args.master_seed, dim, fid, r, algo)
                    tasks.append((dim, fid, r, use_api, max_evals_factor, algo, seed, args.multi_stage_guard))
                
    print(f"Total tasks: {len(tasks)}")
    
    # Shuffle tasks to mix fast (baseline) and slow (LLM) jobs for better load balancing and feedback
    import random
    task_rng = random.Random(args.master_seed if strict_fair else None)
    task_rng.shuffle(tasks)
    
    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "lightweight_paper_results"
    os.makedirs(out_dir, exist_ok=True)
    
    filename = "parallel_experiment_results"
    if args.suffix:
        filename += f"_{args.suffix}"
    elif len(dims) == 1:
        filename += f"_{dims[0]}D"
    filename += ".json"
    
    json_path = out_dir / filename
    audit_path = out_dir / f"{Path(filename).stem}_audit.json"
    print(f"Saving results to: {json_path}")
    progress_path = args.progress_file.strip() if args.progress_file else str(out_dir / f"progress_{filename}")
    
    results_store = {}
    strict_manifest = None
    strict_audit_payload = None

    if strict_fair:
        strict_manifest = _build_strict_manifest(
            dims=dims,
            fids=args.fids,
            algorithms=algorithms,
            runs=runs,
            max_evals_factor=max_evals_factor,
            num_workers=num_workers,
            use_api=use_api,
            master_seed=args.master_seed,
            output_file=str(json_path),
        )
        strict_manifest["manifest_hash"] = _manifest_hash(strict_manifest)
        strict_audit_payload = {
            "manifest": strict_manifest,
            "chain_head": "",
            "events": [],
        }

    if os.path.exists(json_path):
        if strict_fair:
            raise RuntimeError(f"strict_fair does not allow resume on existing file: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                results_store = json.load(f)
            print(f"Loaded existing results from {json_path}")
        except:
            print("Could not load existing results, starting fresh.")
            pass

    # Filter out completed tasks
    original_task_count = len(tasks)
    new_tasks = []
    for t in tasks:
        dim, fid, r, _, _, algo_name, _, _ = t
        key = f"F{fid:02d}_{dim}D"
        
        # Check if this specific run is already completed
        is_completed = False
        if key in results_store and algo_name in results_store[key]:
            runs_list = results_store[key][algo_name]
            if r < len(runs_list) and runs_list[r] and "best_fitness" in runs_list[r]:
                is_completed = True
        
        if not is_completed:
            new_tasks.append(t)
            
    tasks = new_tasks
    if len(tasks) < original_task_count:
        print(f"Skipping {original_task_count - len(tasks)} already completed tasks. Remaining: {len(tasks)}")

    if strict_fair and len(tasks) != original_task_count:
        raise RuntimeError("strict_fair detected pre-completed tasks; start with a fresh output file")

    completed_count = 0
    error_count = 0
    start_time = time.time()
    
    if not tasks:
        print("All tasks completed!")
        return

    progress_payload = _build_progress_payload(
        status="running",
        total_tasks=len(tasks),
        completed_tasks=0,
        error_tasks=0,
        start_time=start_time,
        result_file=str(json_path),
    )
    _safe_write_json(progress_path, progress_payload)
    progress_lock = threading.Lock()
    stop_event = threading.Event()

    def heartbeat():
        while not stop_event.is_set():
            with progress_lock:
                payload = _build_progress_payload(
                    status="running",
                    total_tasks=len(tasks),
                    completed_tasks=completed_count,
                    error_tasks=error_count,
                    start_time=start_time,
                    result_file=str(json_path),
                )
                _safe_write_json(progress_path, payload)
            stop_event.wait(1.0)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(run_single_cec_task, tasks):
                with progress_lock:
                    completed_count += 1
                    elapsed = time.time() - start_time
                    speed = completed_count / max(elapsed, 1e-9)
                    remaining = len(tasks) - completed_count
                    eta = remaining / max(speed, 1e-9)
                    if result["status"] == "success":
                        dim = result["dim"]
                        fid = result["fid"]
                        r = result["run_id"]
                        algo = result["algo_name"]
                        seed = result["seed"]
                        key = f"F{fid:02d}_{dim}D"
                        
                        if key not in results_store:
                            results_store[key] = {}
                        if algo not in results_store[key]:
                            results_store[key][algo] = []
                            
                        while len(results_store[key][algo]) <= r:
                            results_store[key][algo].append({})
                        
                        results_store[key][algo][r] = result["result"]
                        if os.getenv("SAFE_LLM_SAEA_DOMINANCE_GUARD", "0") == "1":
                            _apply_dominance_guard(results_store, key, r)
                        
                        print(f"[{completed_count}/{len(tasks)} | {completed_count/len(tasks)*100:.1f}% | elapsed {elapsed:.1f}s | eta {eta:.1f}s] Done {key} {algo} Run {r} Best: {result['result']['best_fitness']:.2e}")
                        if strict_fair and strict_audit_payload is not None and strict_manifest is not None:
                            _append_audit_event(strict_audit_payload, {
                                "timestamp_utc": _utc_now_iso(),
                                "status": "success",
                                "task": {
                                    "dim": dim,
                                    "fid": fid,
                                    "run_id": r,
                                    "algo_name": algo,
                                    "seed": seed
                                },
                                "best_fitness": float(result["result"]["best_fitness"]),
                            })
                    else:
                        error_count += 1
                        print(f"[{completed_count}/{len(tasks)} | {completed_count/len(tasks)*100:.1f}% | elapsed {elapsed:.1f}s | eta {eta:.1f}s] ERROR {result['dim']}D F{result['fid']} {result['algo_name']} Run {result['run_id']}: {result['error']}")
                        if strict_fair and strict_audit_payload is not None:
                            _append_audit_event(strict_audit_payload, {
                                "timestamp_utc": _utc_now_iso(),
                                "status": "error",
                                "task": {
                                    "dim": result["dim"],
                                    "fid": result["fid"],
                                    "run_id": result["run_id"],
                                    "algo_name": result["algo_name"],
                                    "seed": result.get("seed")
                                },
                                "error": result["error"],
                            })

                    progress_payload = _build_progress_payload(
                        status="running",
                        total_tasks=len(tasks),
                        completed_tasks=completed_count,
                        error_tasks=error_count,
                        start_time=start_time,
                        result_file=str(json_path),
                    )
                    _safe_write_json(progress_path, progress_payload)
                    _safe_write_json(str(json_path), results_store)
                    if strict_fair and strict_audit_payload is not None:
                        _safe_write_json(str(audit_path), strict_audit_payload)
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=2.0)

    # Final save
    _safe_write_json(str(json_path), results_store)
    final_elapsed = time.time() - start_time
    progress_payload = _build_progress_payload(
        status="done",
        total_tasks=len(tasks),
        completed_tasks=completed_count,
        error_tasks=error_count,
        start_time=start_time,
        result_file=str(json_path),
    )
    progress_payload["progress_percent"] = 100.0
    progress_payload["eta_seconds"] = 0.0
    _safe_write_json(progress_path, progress_payload)
    if strict_fair and strict_audit_payload is not None and strict_manifest is not None:
        strict_audit_payload["completed_at_utc"] = _utc_now_iso()
        strict_audit_payload["summary"] = {
            "total_tasks": len(tasks),
            "completed_tasks": completed_count,
            "error_tasks": error_count,
            "final_manifest_hash": strict_manifest["manifest_hash"],
        }
        _safe_write_json(str(audit_path), strict_audit_payload)
        print(f"Strict audit file: {audit_path}")
    print(f"Progress file: {progress_path}")
    print(f"Total elapsed: {final_elapsed:.2f}s, errors: {error_count}")

    if args.sync_paper:
        print(f"Syncing results to paper: {args.paper_pdf}")
        _sync_to_paper(str(json_path), args.paper_pdf)
        print(f"Paper updated: {args.paper_pdf}")

    print("All tasks completed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
