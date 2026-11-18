from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cma
import numpy as np

from safe_llm_saea.cec2022 import get_bounds, get_function
from safe_llm_saea.csa_protocol import CSAContract, audit_csa_compliance, candidate_id, make_trace_event
from safe_llm_saea.paper_ready_safe_llm_saea import ExperimentConfig, LLMAssistedDE


FIDS = (1, 6, 9)
DIMENSION = 10
BOUNDS = [(-100.0, 100.0)] * DIMENSION
CONTRACT = CSAContract(
    proposal_source="cached LLM proposal stream",
    validator="dimension and bound validation",
    numerical_ranker="normalized distance to the CMA-ES distribution mean",
    admission_gate="best-ranked valid proposal replaces one native candidate once",
    evaluation_authority="CEC objective evaluator",
    update_authority="CMA-ES tell plus numerically evaluated incumbent",
)


def _save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_path(root: Path, fid: int, run_id: int) -> Path:
    return root / "proposal_cache" / f"cec_f{fid:02d}_10d_r{run_id:02d}.json"


def capture_proposals(root: Path, runs: int, seed_base: int) -> None:
    if not (os.getenv("ds_api") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise ValueError("An API key is required only for proposal capture")
    for fid in FIDS:
        objective = get_function(fid, DIMENSION)
        for run_id in range(runs):
            path = _capture_path(root, fid, run_id)
            if path.exists():
                continue
            seed = seed_base + fid * 100 + run_id
            np.random.seed(seed)
            config = ExperimentConfig(
                dimension=DIMENSION,
                population_size=10,
                max_evaluations=80,
                llm_enhancement=True,
                max_llm_calls_per_run=1,
                max_llm_calls_stage1=1,
                max_llm_calls_stage2=0,
                force_initial_llm_call=True,
                min_remaining_budget=0,
                min_call_interval=1,
                use_surrogate=True,
                enable_sniper=False,
                proposal_trace_mode="capture",
                problem_name=f"CEC2022_F{fid:02d}",
                problem_description=f"CEC 2022 function F{fid:02d}, {DIMENSION}D, bounded minimization",
            )
            optimizer = LLMAssistedDE(config=config, use_api=True)
            result = optimizer.optimize(objective, BOUNDS)
            events = result.get("diagnostics", {}).get("proposal_trace", [])
            vectors = [vector for event in events for vector in event.get("returned_candidate_vectors", [])]
            if not vectors:
                raise RuntimeError(f"No proposal vectors captured for F{fid:02d} run {run_id}")
            _save({
                "fid": fid, "dimension": DIMENSION, "run_id": run_id, "seed": seed,
                "source": "bounded LLM capture through LLMAssistedDE",
                "model": optimizer.llm_model,
                "proposal_vectors": vectors,
                "source_trace": events,
            }, path)


def _trajectory_hash(curve: Sequence[float], best_solution: Sequence[float]) -> str:
    payload = json.dumps({
        "curve": [round(float(value), 12) for value in curve],
        "best_solution": [round(float(value), 12) for value in best_solution],
    }, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_proposals(vectors: Sequence[Sequence[float]]) -> List[np.ndarray]:
    valid = []
    for vector in vectors:
        arr = np.asarray(vector, dtype=float)
        if arr.shape != (DIMENSION,) or not np.all(np.isfinite(arr)):
            continue
        if all(low <= arr[index] <= high for index, (low, high) in enumerate(BOUNDS)):
            valid.append(arr)
    return valid


def run_cmaes(
    objective: Any,
    seed: int,
    max_fes: int,
    proposal_vectors: Sequence[Sequence[float]],
    strict_zero: bool,
    wrapper_enabled: bool,
) -> Dict[str, Any]:
    lb = np.asarray([item[0] for item in BOUNDS], dtype=float)
    ub = np.asarray([item[1] for item in BOUNDS], dtype=float)
    x0 = (lb + ub) / 2.0
    sigma0 = float(np.mean(ub - lb) / 3.0)
    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, {
        "bounds": [lb.tolist(), ub.tolist()], "popsize": 10, "maxfevals": max_fes,
        "seed": int(seed), "verbose": -9, "verb_disp": 0, "verb_log": 0,
    })
    evaluations = 0
    generation = 0
    best_fitness = float("inf")
    best_solution = x0.copy()
    curve: List[float] = []
    trace: List[Dict[str, Any]] = []
    proposal_consumed = False

    while not es.stop() and evaluations < max_fes:
        solutions = [np.asarray(value, dtype=float) for value in es.ask()]
        event: Dict[str, Any] | None = None
        injected_id = ""
        if wrapper_enabled and not strict_zero and not proposal_consumed and generation == 1:
            valid = _valid_proposals(proposal_vectors)
            if valid:
                scale = np.maximum(1e-12, ub - lb)
                scores = [float(np.linalg.norm((value - np.asarray(es.mean)) / scale)) for value in valid]
                selected = valid[int(np.argmin(scores))]
                solutions[-1] = selected.copy()
                injected_id = candidate_id(selected)
                event = make_trace_event(
                    returned_vectors=proposal_vectors,
                    validated_vectors=valid,
                    selected_vectors=[selected],
                    admitted_vectors=[selected],
                    evaluated_vectors=[],
                    generation=generation,
                    numerical_scores=scores,
                )
                proposal_consumed = True

        fitnesses = []
        proposal_improved = False
        previous_best = best_fitness
        for solution in solutions:
            value = float(objective(solution))
            evaluations += 1
            fitnesses.append(value)
            if value < best_fitness:
                best_fitness = value
                best_solution = solution.copy()
            if injected_id and candidate_id(solution) == injected_id and value < previous_best:
                proposal_improved = True
        if event is not None:
            selected_vector = solutions[-1]
            event["evaluated"] = [{"candidate_id": injected_id, "vector": selected_vector.tolist()}]
            event["incumbent_update_ids"] = [injected_id] if proposal_improved else []
            trace.append(event)
        es.tell([value.tolist() for value in solutions], fitnesses)
        curve.append(best_fitness)
        generation += 1

    payload = {
        "algorithm": "CSA-wrapped CMA-ES" if wrapper_enabled else "native CMA-ES",
        "seed": int(seed),
        "max_fes": int(max_fes),
        "strict_zero_call": bool(strict_zero),
        "csa_contract": CONTRACT.as_dict() if wrapper_enabled else None,
        "evaluations_used": evaluations,
        "best_fitness": best_fitness,
        "best_solution": best_solution.tolist(),
        "convergence_curve": curve,
        "trajectory_sha256": _trajectory_hash(curve, best_solution),
        "csa_trace": trace,
    }
    if wrapper_enabled:
        payload["resource_audit"] = {
            "expected": {"seed": int(seed), "max_fes": int(max_fes), "evaluations_used": int(max_fes)},
            "observed": {"seed": int(seed), "max_fes": int(max_fes), "evaluations_used": int(evaluations)},
        }
        payload["compliance"] = audit_csa_compliance(payload)
    return payload


def execute(root: Path, runs: int, seed_base: int, max_fes: int) -> None:
    summary_rows = []
    for fid in FIDS:
        objective = get_function(fid, DIMENSION)
        for run_id in range(runs):
            cache = json.loads(_capture_path(root, fid, run_id).read_text(encoding="utf-8"))
            proposals = cache["proposal_vectors"]
            seed = seed_base + fid * 100 + run_id
            native = run_cmaes(objective, seed, max_fes, [], strict_zero=False, wrapper_enabled=False)
            strict = run_cmaes(objective, seed, max_fes, proposals, strict_zero=True, wrapper_enabled=True)
            full = run_cmaes(objective, seed, max_fes, proposals, strict_zero=False, wrapper_enabled=True)
            case_root = root / "runs" / f"f{fid:02d}_r{run_id:02d}"
            _save(native, case_root / "native_cmaes.json")
            _save(strict, case_root / "strict_zero_csa_cmaes.json")
            _save(full, case_root / "csa_cmaes.json")
            summary_rows.append({
                "fid": fid, "run_id": run_id, "seed": seed,
                "native_strict_trajectory_equal": native["trajectory_sha256"] == strict["trajectory_sha256"],
                "native_strict_best_equal": native["best_fitness"] == strict["best_fitness"],
                "strict_compliant": strict["compliance"]["csa_compliant"],
                "full_compliant": full["compliance"]["csa_compliant"],
                "strict_zero_closure": strict["compliance"]["zero_call_closure"],
                "full_trace_completeness": full["compliance"]["trace_completeness"],
                "native_fes": native["evaluations_used"], "strict_fes": strict["evaluations_used"],
                "full_fes": full["evaluations_used"],
            })
    report = {
        "protocol": {"fids": list(FIDS), "dimension": DIMENSION, "runs": runs, "max_fes": max_fes},
        "contract": CONTRACT.as_dict(), "rows": summary_rows,
        "aggregate": {
            "runs": len(summary_rows),
            "all_native_strict_trajectories_equal": all(row["native_strict_trajectory_equal"] for row in summary_rows),
            "all_native_strict_best_equal": all(row["native_strict_best_equal"] for row in summary_rows),
            "all_strict_compliant": all(row["strict_compliant"] for row in summary_rows),
            "all_full_compliant": all(row["full_compliant"] for row in summary_rows),
            "all_strict_zero_closed": all(row["strict_zero_closure"] for row in summary_rows),
            "mean_full_trace_completeness": float(np.mean([row["full_trace_completeness"] for row in summary_rows])),
            "all_fes_equal": all(row["native_fes"] == row["strict_fes"] == row["full_fes"] for row in summary_rows),
        },
    }
    _save(report, root / "csa_cmaes_portability_report.json")
    print(json.dumps(report["aggregate"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["capture", "run", "all"], default="run")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=20260712)
    parser.add_argument("--max_fes", type=int, default=200)
    args = parser.parse_args()
    root = Path(args.output_root)
    if args.mode in {"capture", "all"}:
        capture_proposals(root, args.runs, args.seed_base)
    if args.mode in {"run", "all"}:
        execute(root, args.runs, args.seed_base, args.max_fes)


if __name__ == "__main__":
    main()
