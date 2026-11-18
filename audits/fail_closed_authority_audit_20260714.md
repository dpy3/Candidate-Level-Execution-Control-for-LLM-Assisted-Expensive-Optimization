# Fail-Closed Authority and Provenance Audit

## Protocol

- Eight UAV hard cases (`hard_s00`--`hard_s07`).
- Five fixed seeds per case (`20260407`--`20260411`).
- Exactly 120 real objective evaluations per run.
- Exactly two replayed semantic calls per positive-call run.
- Identical cached proposals, proposal SHA-256, seeds, budgets, calls, and token counts for fail-closed and weak-control pairs.
- Final policy: `rank_top`, `surrogate_min_history=16`, and `surrogate_unavailable_policy=reject`.

## Authority result

- 40/40 fail-closed runs completed.
- 80 admission events were audited.
- 120/120 semantic evaluations contained surrogate-ready numerical scores.
- No evaluated event used a response-order or prediction-failure fallback.
- 40/40 lineage-aware v2 traces passed the checker.
- Mean trace completeness was 1.0.

## Bounded performance result

| Setting | Average rank | Mean system cost | Semantic evaluations |
|---|---:|---:|---:|
| Fail-closed rank-top | 1.50 | 850.34 | 120 |
| Weak control | 2.00 | 895.61 | 120 |
| Strict zero-call | 2.50 | 966.67 | 0 |

Fail-closed rank-top is better than weak control on six of eight original case means. The one-sided case-level Wilcoxon result is `p=0.02734375`. This result is restricted to the reused original audit set. On the separate four-case confirmation set, rank-top remains worse in aggregate than weak sampling (743.00 versus 695.63; `p=0.8125`).

## Checker fault injection

- Five requested fault classes: forged state, skipped admission, budget mismatch, candidate-ID replacement, and missing predecessor.
- 100 deterministic mutations per class.
- Detection: 500/500.
- Expected violation code: 500/500.
- Clean controls: 100/100 passed; false-positive rate 0.

## Reproduction

```powershell
python -m safe_llm_saea.application_benchmark.run_csa_fail_closed_replay `
  --manifest_path tmp\app_benchmarks\lade_proof_20260414_formal_default\uav\manifest.json `
  --source_raw_dir paper\q1\experiment_outputs\cached_proposal_replay_phase3\capture\raw `
  --reference_raw_dir paper\q1\experiment_outputs\cached_proposal_replay_phase3_ranked\replay\raw `
  --output_root paper\q1\experiment_outputs\csa_fail_closed_rerun_v3 `
  --runs 5 --seed_base 20260407 --max_fes 120 --population_size 12 `
  --max_iter 8 --llm_call_budget 2 --surrogate_min_history 16

python -m safe_llm_saea.experiments.run_csa_fault_injection `
  --trials_per_fault 100 --seed 20260714 `
  --output paper\q1\experiment_outputs\csa_checker_fault_injection\fault_injection_report.json
```

## Claim boundary

The audit establishes fail-closed numerical authorization, resource pairing, and record-level provenance for the tested protocol. It does not establish cryptographic trace authenticity, deployment-level safety, or task-independent acquisition superiority.
