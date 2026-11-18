from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np
from scipy.stats import rankdata, wilcoxon

from safe_llm_saea.application_benchmark.common.io_utils import load_json, make_run_filename, save_json
from safe_llm_saea.application_benchmark.run_cached_proposal_replay import CAPTURE_LABEL, _budget, _llm_config, _validate_trace
from safe_llm_saea.application_benchmark.run_lade_proof_experiments import _proof_lade_variant
from safe_llm_saea.application_benchmark.uav.run_uav_benchmark import run_single_uav_case
from safe_llm_saea.csa_protocol import audit_csa_compliance


PROTOCOL_ID = "CSA-UAV-FAIR-FAMILY-BASELINES-2026-07-15"
POLICIES = {
    "direct-response-order": {
        "label": "CSA-fair-direct-response-order",
        "surrogate_admission_policy": "response_order",
        "weak_admission_control": False,
        "role": "protocol-matched direct semantic-injection baseline",
    },
    "random-budget-matched": {
        "label": "CSA-fair-random-budget-matched",
        "surrogate_admission_policy": "rank_top",
        "weak_admission_control": True,
        "role": "budget-matched weak admission control",
    },
    "gp-mean-top": {
        "label": "CSA-fair-gp-mean-top",
        "surrogate_admission_policy": "mean_top",
        "weak_admission_control": False,
        "role": "protocol-matched surrogate mean-ranking baseline",
    },
    "csa-lcb-rank-top": {
        "label": "CSA-fair-lcb-rank-top",
        "surrogate_admission_policy": "rank_top",
        "weak_admission_control": False,
        "role": "fail-closed CSA policy",
    },
    "strict-zero": {
        "label": "CSA-fair-strict-zero",
        "strict_zero_call": True,
        "role": "null-semantic numerical-backbone control",
    },
}


def _source_path(source_dir: Path, case_id: str, run_id: int) -> Path:
    return source_dir / make_run_filename("uav", case_id, CAPTURE_LABEL, run_id)


def _trace_audit(record: Dict[str, Any], strict_zero: bool) -> Dict[str, Any]:
    diagnostics = record.get("diagnostics", {})
    trace = diagnostics.get("csa_trace", [])
    manifest = {
        "event_count": len(trace),
        "semantic_evaluations": sum(len(event.get("evaluated", [])) for event in trace),
        "runtime_authorization_events": sum(bool(event.get("runtime_authorization_enforced")) for event in trace),
    }
    return audit_csa_compliance({
        "strict_zero_call": strict_zero,
        "csa_trace": trace,
        "trace_manifest": manifest,
    })


def _run_one(
    args: argparse.Namespace,
    scenario: Dict[str, Any],
    run_id: int,
    seed: int,
    source_dir: Path,
    raw_dir: Path,
    policy_name: str,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    label = str(policy["label"])
    output = raw_dir / make_run_filename("uav", str(scenario["case_id"]), label, run_id)
    if output.exists() and not args.overwrite:
        return load_json(output)

    if policy_name == "strict-zero":
        variant = _proof_lade_variant(
            "uav",
            str(args.uav_proof_profile),
            strict_zero_call=True,
            llm_enabled=False,
            force_initial_llm_call=False,
            enforce_full_evaluation_budget=True,
            use_api=False,
        )
        llm_config = {**_llm_config(args, use_api=False), "enabled_algorithms": set()}
    else:
        source_path = _source_path(source_dir, str(scenario["case_id"]), run_id)
        source = _validate_trace(source_path)
        if int(source.get("seed", -1)) != int(seed):
            raise ValueError(f"Seed mismatch for {source_path}")
        if dict(source.get("budget", {})) != _budget(args):
            raise ValueError(f"Budget mismatch for {source_path}")
        variant = _proof_lade_variant(
            "uav",
            str(args.uav_proof_profile),
            weak_admission_control=bool(policy.get("weak_admission_control", False)),
            proposal_trace_mode="replay",
            proposal_replay_path=str(source_path.resolve()),
            proposal_replay_strict=True,
            enforce_full_evaluation_budget=True,
            surrogate_min_history=16,
            surrogate_admission_policy=str(policy["surrogate_admission_policy"]),
            surrogate_unavailable_policy="reject",
            use_api=False,
        )
        llm_config = _llm_config(args, use_api=False)

    return run_single_uav_case(
        scenario=scenario,
        algorithm_name="L-ADE-Sniper",
        run_id=run_id,
        seed=seed,
        budget_config=_budget(args),
        llm_config=llm_config,
        output_dir=str(raw_dir),
        algorithm_label=label,
        algorithm_extra_config={"lade_variant": variant},
    )


def _summarize(records: List[Dict[str, Any]], strict_zero: bool) -> Dict[str, Any]:
    costs = [float(record["metrics"]["system_cost"]) for record in records]
    feasible = [bool(record["metrics"].get("success_under_strict_rule", record.get("feasible", False))) for record in records]
    collision_free = [not bool(record["metrics"].get("collision", False)) for record in records]
    semantic_evaluations = 0
    runtime_authorizations = 0
    compliant = 0
    for record in records:
        diagnostics = record.get("diagnostics", {})
        semantic_evaluations += sum(len(event.get("evaluated", [])) for event in diagnostics.get("csa_trace", []))
        runtime_authorizations += sum(len(event.get("evaluation_authorizations", [])) for event in diagnostics.get("csa_trace", []))
        compliant += int(_trace_audit(record, strict_zero)["csa_compliant"])
    return {
        "runs": len(records),
        "mean_system_cost": mean(costs),
        "median_system_cost": float(np.median(costs)),
        "strict_feasible_rate": mean(feasible),
        "collision_free_rate": mean(collision_free),
        "semantic_evaluations": semantic_evaluations,
        "runtime_authorizations": runtime_authorizations,
        "trace_compliant_runs": compliant,
    }


def execute(args: argparse.Namespace) -> Dict[str, Any]:
    manifest = load_json(args.manifest_path)
    scenarios = list(manifest.get("scenarios", []))
    seeds = [int(args.seed_base) + offset for offset in range(int(args.runs))]
    source_dir = Path(args.source_raw_dir)
    root = Path(args.output_root)
    raw_dir = root / "raw"
    by_policy: Dict[str, List[Dict[str, Any]]] = {name: [] for name in POLICIES}
    case_costs: Dict[str, Dict[str, List[float]]] = {name: {} for name in POLICIES}

    for scenario in scenarios:
        case_id = str(scenario["case_id"])
        for run_id, seed in enumerate(seeds):
            for policy_name, policy in POLICIES.items():
                record = _run_one(args, scenario, run_id, seed, source_dir, raw_dir, policy_name, policy)
                by_policy[policy_name].append(record)
                case_costs[policy_name].setdefault(case_id, []).append(float(record["metrics"]["system_cost"]))

    case_means = {
        name: {case: mean(values) for case, values in cases.items()}
        for name, cases in case_costs.items()
    }
    case_ids = [str(scenario["case_id"]) for scenario in scenarios]
    ranks = {name: [] for name in POLICIES}
    for case_id in case_ids:
        values = [case_means[name][case_id] for name in POLICIES]
        for name, rank in zip(POLICIES, rankdata(values, method="average")):
            ranks[name].append(float(rank))

    fairness_mismatches = []
    nonzero_names = [name for name in POLICIES if name != "strict-zero"]
    for index in range(len(by_policy[nonzero_names[0]])):
        reference = by_policy[nonzero_names[0]][index]
        reference_diag = reference.get("diagnostics", {})
        for name in nonzero_names[1:]:
            observed = by_policy[name][index]
            observed_diag = observed.get("diagnostics", {})
            fields = {
                "seed": (reference.get("seed"), observed.get("seed")),
                "budget": (reference.get("budget"), observed.get("budget")),
                "proposal_sha256": (
                    reference_diag.get("proposal_replay_source_sha256"),
                    observed_diag.get("proposal_replay_source_sha256"),
                ),
                "returned_vectors": (
                    [event.get("returned_candidate_vectors", []) for event in reference_diag.get("admission_events", [])],
                    [event.get("returned_candidate_vectors", []) for event in observed_diag.get("admission_events", [])],
                ),
            }
            bad = [field for field, pair in fields.items() if pair[0] != pair[1]]
            if bad:
                fairness_mismatches.append({"pair_index": index, "policy": name, "fields": bad})

    target = np.asarray([case_means["csa-lcb-rank-top"][case] for case in case_ids])
    paired_tests = {}
    for name in nonzero_names:
        if name == "csa-lcb-rank-top":
            continue
        comparator = np.asarray([case_means[name][case] for case in case_ids])
        result = wilcoxon(target, comparator, alternative="less")
        paired_tests[name] = {"statistic": float(result.statistic), "pvalue": float(result.pvalue), "unit": "case mean"}

    report = {
        "protocol_id": PROTOCOL_ID,
        "scope": "protocol-matched functional baselines; not exact reproductions of external published systems",
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "proposal_source": str(source_dir.resolve()),
        "budget": _budget(args),
        "seeds": seeds,
        "policies": POLICIES,
        "summaries": {
            name: {**_summarize(records, name == "strict-zero"), "mean_case_rank": mean(ranks[name])}
            for name, records in by_policy.items()
        },
        "case_means": case_means,
        "paired_wilcoxon_lcb_better": paired_tests,
        "fairness_audit": {
            "paired_runs": len(by_policy[nonzero_names[0]]),
            "mismatches": fairness_mismatches,
            "all_nonzero_resources_and_proposals_matched": not fairness_mismatches,
        },
    }
    save_json(report, root / "fair_family_baseline_report.json")
    print(report["summaries"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--source_raw_dir", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=20260407)
    parser.add_argument("--max_fes", type=int, default=120)
    parser.add_argument("--population_size", type=int, default=12)
    parser.add_argument("--max_iter", type=int, default=8)
    parser.add_argument("--llm_call_budget", type=int, default=2)
    parser.add_argument("--llm_cost_budget_usd", type=float, default=0.05)
    parser.add_argument("--uav_proof_profile", choices=["default", "engaged"], default="default")
    parser.add_argument("--api_key", default="")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--llm_model", default="deepseek-chat")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
