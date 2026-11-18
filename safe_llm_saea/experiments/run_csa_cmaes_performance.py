from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import cma
import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from safe_llm_saea.cec2022 import get_function
from safe_llm_saea.csa_protocol import CSAContract, RuntimeAdmissionGate, audit_csa_compliance, candidate_id, make_trace_event


PROTOCOL_ID = "CSA-CMAES-PERFORMANCE-RUNTIME-GATE-2026-07-15"
FIDS = (1, 6, 9)
DIMENSION = 10
BOUNDS = [(-100.0, 100.0)] * DIMENSION
POLICIES = ("native", "strict-zero", "direct-response-order", "random-budget-matched", "distance-top", "gp-mean-top", "gp-lcb-rank-top")
CONTRACT = CSAContract(
    proposal_source="frozen LLM proposal stream",
    validator="finite dimension and bound validation",
    numerical_ranker="policy-specific host rank",
    admission_gate="one proposal replaces one native CMA-ES candidate once",
    evaluation_authority="runtime-gated CEC objective evaluator",
    update_authority="CMA-ES tell plus numerically evaluated incumbent",
)


def _save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cache_path(root: Path, fid: int, run_id: int) -> Path:
    return root / f"cec_f{fid:02d}_10d_r{run_id:02d}.json"


def _valid(vectors: Sequence[Sequence[float]]) -> List[np.ndarray]:
    valid = []
    for vector in vectors:
        candidate = np.asarray(vector, dtype=float)
        if candidate.shape != (DIMENSION,) or not np.all(np.isfinite(candidate)):
            continue
        if all(low <= candidate[index] <= high for index, (low, high) in enumerate(BOUNDS)):
            valid.append(candidate)
    return valid


def _trajectory_hash(curve: Sequence[float], solution: Sequence[float]) -> str:
    body = json.dumps({
        "curve": [round(float(value), 12) for value in curve],
        "solution": [round(float(value), 12) for value in solution],
    }, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _gp_scores(history_x: List[np.ndarray], history_y: List[float], candidates: List[np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed")
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None, random_state=seed)
    scale = np.asarray([high - low for low, high in BOUNDS], dtype=float)
    lower = np.asarray([low for low, _ in BOUNDS], dtype=float)
    train_x = (np.asarray(history_x) - lower) / scale
    test_x = (np.asarray(candidates) - lower) / scale
    model.fit(train_x, np.asarray(history_y, dtype=float))
    return model.predict(test_x, return_std=True)


def _choose(policy: str, valid: List[np.ndarray], mean_vector: np.ndarray, history_x: List[np.ndarray], history_y: List[float], seed: int) -> tuple[np.ndarray, List[float]]:
    if policy == "direct-response-order":
        return valid[0], [float(index) for index in range(len(valid))]
    if policy == "random-budget-matched":
        index = int(np.random.default_rng(seed + 99173).integers(0, len(valid)))
        return valid[index], [0.0 if position == index else 1.0 for position in range(len(valid))]
    if policy == "distance-top":
        scale = np.asarray([high - low for low, high in BOUNDS], dtype=float)
        scores = np.asarray([np.linalg.norm((candidate - mean_vector) / scale) for candidate in valid])
    else:
        mu, sigma = _gp_scores(history_x, history_y, valid, seed)
        scores = mu if policy == "gp-mean-top" else mu - 0.5 * sigma
    index = int(np.argmin(scores))
    return valid[index], [float(value) for value in scores]


def run_cmaes(objective: Any, seed: int, max_fes: int, proposal_vectors: Sequence[Sequence[float]], policy: str) -> Dict[str, Any]:
    lower = np.asarray([low for low, _ in BOUNDS], dtype=float)
    upper = np.asarray([high for _, high in BOUNDS], dtype=float)
    x0 = (lower + upper) / 2.0
    es = cma.CMAEvolutionStrategy(x0.tolist(), float(np.mean(upper - lower) / 3.0), {
        "bounds": [lower.tolist(), upper.tolist()], "popsize": 10, "maxfevals": max_fes,
        "seed": int(seed), "verbose": -9, "verb_disp": 0, "verb_log": 0,
    })
    evaluations = 0
    generation = 0
    best_fitness = float("inf")
    best_solution = x0.copy()
    curve: List[float] = []
    trace: List[Dict[str, Any]] = []
    history_x: List[np.ndarray] = []
    history_y: List[float] = []
    injected = False

    while not es.stop() and evaluations < max_fes:
        solutions = [np.asarray(vector, dtype=float) for vector in es.ask()]
        selected: np.ndarray | None = None
        numerical_scores: List[float] = []
        if policy not in {"native", "strict-zero"} and not injected and generation == 1:
            valid = _valid(proposal_vectors)
            if valid:
                selected, numerical_scores = _choose(policy, valid, np.asarray(es.mean), history_x, history_y, seed)
                solutions[-1] = selected.copy()
                injected = True

        fitnesses: List[float] = []
        authorization = None
        selected_value = None
        previous_best = best_fitness
        gate = None
        if selected is not None:
            gate = RuntimeAdmissionGate([selected], [selected], remaining_budget=max_fes - evaluations, event_id=f"generation-{generation}", policy=policy)
            authorization = gate.issue(selected)

        for solution in solutions:
            is_selected = selected is not None and candidate_id(solution) == candidate_id(selected)
            if is_selected:
                value = float(gate.evaluate(authorization, solution, objective))
                selected_value = value
            else:
                value = float(objective(solution))
            evaluations += 1
            fitnesses.append(value)
            history_x.append(solution.copy())
            history_y.append(value)
            if value < best_fitness:
                best_fitness = value
                best_solution = solution.copy()

        if selected is not None and authorization is not None:
            improved = bool(selected_value is not None and selected_value < previous_best)
            event = make_trace_event(
                proposal_vectors,
                _valid(proposal_vectors),
                [selected],
                [selected],
                [selected],
                population_accepted_vectors=[selected],
                best_update_vectors=[selected] if improved else [],
                generation=generation,
                selection_policy=policy,
                numerical_scores=numerical_scores,
                runtime_authorization_enforced=True,
                evaluation_authorizations=[authorization.as_dict()],
            )
            trace.append(event)

        es.tell([solution.tolist() for solution in solutions], fitnesses)
        curve.append(best_fitness)
        generation += 1

    strict_zero = policy == "strict-zero"
    trace_manifest = {
        "event_count": len(trace),
        "semantic_evaluations": sum(len(event["evaluated"]) for event in trace),
        "runtime_authorization_events": sum(bool(event.get("runtime_authorization_enforced")) for event in trace),
    }
    payload = {
        "protocol_id": PROTOCOL_ID,
        "algorithm": "native CMA-ES" if policy == "native" else "CSA-wrapped CMA-ES",
        "policy": policy,
        "seed": seed,
        "max_fes": max_fes,
        "strict_zero_call": strict_zero,
        "csa_contract": None if policy == "native" else CONTRACT.as_dict(),
        "evaluations_used": evaluations,
        "best_fitness": best_fitness,
        "best_solution": best_solution.tolist(),
        "convergence_curve": curve,
        "trajectory_sha256": _trajectory_hash(curve, best_solution),
        "csa_trace": trace,
        "trace_manifest": trace_manifest,
        "resource_audit": {
            "expected": {"seed": seed, "max_fes": max_fes, "evaluations_used": max_fes},
            "observed": {"seed": seed, "max_fes": max_fes, "evaluations_used": evaluations},
        },
    }
    if policy != "native":
        payload["compliance"] = audit_csa_compliance(payload)
    return payload


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.output_root)
    proposal_root = Path(args.proposal_cache)
    rows = []
    by_policy: Dict[str, List[Dict[str, Any]]] = {policy: [] for policy in POLICIES}
    for fid in FIDS:
        objective = get_function(fid, DIMENSION)
        for run_id in range(args.runs):
            cache = json.loads(_cache_path(proposal_root, fid, run_id).read_text(encoding="utf-8"))
            proposals = cache["proposal_vectors"]
            seed = args.seed_base + fid * 100 + run_id
            run_payloads = {}
            for policy in POLICIES:
                path = root / "runs" / f"f{fid:02d}_r{run_id:02d}" / f"{policy}.json"
                if path.exists() and not args.overwrite:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                else:
                    payload = run_cmaes(objective, seed, args.max_fes, proposals, policy)
                    _save(payload, path)
                by_policy[policy].append(payload)
                run_payloads[policy] = payload
            row = {
                "fid": fid, "run_id": run_id, "seed": seed,
                "native_strict_trajectory_equal": run_payloads["native"]["trajectory_sha256"] == run_payloads["strict-zero"]["trajectory_sha256"],
                "native_strict_best_equal": run_payloads["native"]["best_fitness"] == run_payloads["strict-zero"]["best_fitness"],
                "all_fes_equal": len({run_payloads[policy]["evaluations_used"] for policy in POLICIES}) == 1,
                "all_wrappers_compliant": all(run_payloads[policy]["compliance"]["csa_compliant"] for policy in POLICIES if policy != "native"),
                "best_fitness": {policy: run_payloads[policy]["best_fitness"] for policy in POLICIES},
            }
            rows.append(row)

    rank_lists = {policy: [] for policy in POLICIES}
    for row in rows:
        values = [row["best_fitness"][policy] for policy in POLICIES]
        for policy, rank in zip(POLICIES, rankdata(values, method="average")):
            rank_lists[policy].append(float(rank))

    summaries = {}
    native_values = np.asarray([item["best_fitness"] for item in by_policy["native"]])
    for policy, payloads in by_policy.items():
        values = np.asarray([item["best_fitness"] for item in payloads])
        differences = values - native_values
        wins = int(np.sum(differences < -1e-12))
        losses = int(np.sum(differences > 1e-12))
        ties = len(values) - wins - losses
        summary = {
            "runs": len(payloads),
            "mean_pair_rank": mean(rank_lists[policy]),
            "wins_vs_native": wins,
            "ties_vs_native": ties,
            "losses_vs_native": losses,
            "mean_relative_change_vs_native": float(np.mean(differences / np.maximum(1.0, np.abs(native_values)))),
            "runtime_authorizations": sum(len(event.get("evaluation_authorizations", [])) for item in payloads for event in item.get("csa_trace", [])),
        }
        if policy not in {"native", "strict-zero"} and np.any(np.abs(differences) > 1e-12):
            result = wilcoxon(values, native_values, alternative="less")
            summary["wilcoxon_better_than_native"] = {"statistic": float(result.statistic), "pvalue": float(result.pvalue), "unit": "paired run"}
        summaries[policy] = summary

    function_means = {
        str(fid): {
            policy: mean(row["best_fitness"][policy] for row in rows if row["fid"] == fid)
            for policy in POLICIES
        }
        for fid in FIDS
    }
    report = {
        "protocol_id": PROTOCOL_ID,
        "protocol": {"fids": list(FIDS), "dimension": DIMENSION, "runs_per_function": args.runs, "max_fes": args.max_fes, "injection_generation": 1, "semantic_evaluations_per_nonzero_run": 1},
        "proposal_source": str(proposal_root.resolve()),
        "policies": list(POLICIES),
        "rows": rows,
        "summaries": summaries,
        "function_means": function_means,
        "aggregate": {
            "runs": len(rows),
            "all_native_strict_trajectories_equal": all(row["native_strict_trajectory_equal"] for row in rows),
            "all_native_strict_best_equal": all(row["native_strict_best_equal"] for row in rows),
            "all_fes_equal": all(row["all_fes_equal"] for row in rows),
            "all_wrappers_compliant": all(row["all_wrappers_compliant"] for row in rows),
        },
    }
    _save(report, root / "csa_cmaes_performance_report.json")
    print(json.dumps({"aggregate": report["aggregate"], "summaries": summaries}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal_cache", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=20260712)
    parser.add_argument("--max_fes", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
