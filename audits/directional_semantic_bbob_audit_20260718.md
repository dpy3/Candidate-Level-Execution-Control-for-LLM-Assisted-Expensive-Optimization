# Backbone-specific directional semantic audit (2026-07-18)

## Design change

The semantic source no longer returns absolute candidate coordinates. DE and
PSO receive separate normalized state summaries. The model returns exactly
three bounded directions and local step scales, and host code constructs the
candidate vectors around the current incumbent. The cache records prompt and
state checksums, direction instructions, candidate IDs, and incumbent-source
lineage, but not raw responses or objective labels.

## CEC 2022 development candidate quality

The development set contains seven functions, three seeds, and both
backbones, for 42 backbone-state events and 126 directional candidates.

| Backbone | Source | Candidate improvement rate | Event hit rate | Mean best normalized regret |
|---|---|---:|---:|---:|
| DE | Legacy absolute coordinates | 0.000 | 0.000 | 3,732,360.524 |
| DE | Direction plus host step | 0.476 | 0.810 | -0.059 |
| PSO | Legacy absolute coordinates | 0.000 | 0.000 | 115.349 |
| PSO | Direction plus host step | 0.349 | 0.714 | -0.074 |

All predeclared candidate-quality gates pass. The result establishes that the
representation and backbone-specific prompt materially improve proposal
quality at the matched incumbent states.

## Risk and commit development

One objective evaluation is reserved in every strict/semantic pair. The
strict-zero branch re-evaluates the incumbent; the semantic branch evaluates a
candidate only when its GP upper regret is non-positive. A candidate enters
the population only after its true value improves the pre-evaluation
incumbent. The frozen risk multipliers are 1.0 for DE and 0.5 for PSO.

- DE development: 3/18/0, four semantic evaluations, 80.95% abstention, and
  -0.77% mean relative change.
- PSO development: 4/16/1, ten semantic evaluations, 52.38% abstention, and
  -4.34% mean relative change.

## Frozen BBOB confirmation

- Implementation: IOHprofiler `ioh 0.3.22` under Python 3.12.13.
- Dimension: 10.
- Functions: Sphere (F1), Rosenbrock (F8), rotated Rastrigin (F15), and
  Gallagher 101 peaks (F21).
- Instances: 1--5.
- Protocol SHA-256:
  `c4e0c60f5949ebd8ec2c2d2d2f8b4fdfbc1c54c28b6c69584effbc9f903c8450`.
- Capture: 40 fresh calls, 120 candidates, and 234,060 tokens.
- Result SHA-256:
  `9af58c10d26af54cc05bc64f4d5b2a5c148b8a94e92c9d42a45c75a692b0986a`.

| Backbone | Candidate improvement | Event hit | W/T/L | NTR | Abstention | Mean change (95% bootstrap CI) |
|---|---:|---:|---:|---:|---:|---:|
| DE | 0.450 | 0.750 | 2/11/7 | 0.350 | 0.500 | +19.31% [-1.67%, +51.21%] |
| PSO | 0.467 | 0.800 | 4/14/2 | 0.100 | 0.600 | -3.13% [-13.02%, +3.76%] |

All realized semantic evaluations use one-use runtime authorization, and every
trace passes the CSA checker. DE fails the negative-transfer, W/L, and mean
effect gates. PSO passes W/L and negative-transfer gates but its mean-effect
interval crosses zero. Consequently, no backbone passes every frozen
effectiveness gate and `any_backbone_claim_eligible=false`.

## Pre-frozen BBOB power extension

The power extension retains the same four BBOB functions, dimension, prompts,
risk multipliers, budgets, abstention rule, and improvement-only commit rule.
It uses unseen instances 6--13 and executes all 64 fresh backbone-case calls,
giving 32 paired final-performance cases per backbone. Instances 1--5 are not
reused. The protocol was frozen before capture with SHA-256
`2f244f74055d8f508176fa39ea92428deb7963a5135ce8dffe8d5f6ded95c4ad`.
The calls consumed 322,166 prompt tokens and 56,213 completion tokens, or
378,379 tokens in total.

| Backbone | Candidate improvement | Event hit | W/T/L | NTR | Abstention | Mean change (95% bootstrap CI) |
|---|---:|---:|---:|---:|---:|---:|
| DE | 0.542 | 0.813 | 6/20/6 | 0.188 | 0.469 | +13.27% [-4.93%, +42.15%] |
| PSO | 0.438 | 0.844 | 12/17/3 | 0.094 | 0.406 | -5.18% [-11.78%, +0.34%] |

DE's large unfavorable mean is driven mainly by Rosenbrock, for which the
eight-case mean change is +55%. PSO has favorable means on all four functions
and wins 12 non-tied comparisons against three losses. Nevertheless, its
predeclared mean-effect interval crosses zero by 0.34 percentage points. The
PSO Wilcoxon result (`p=0.0215`) is a supportive diagnostic, not a replacement
for the failed bootstrap gate. All 36 realized semantic evaluations use
one-use authorization, all 27 population commits follow a true incumbent
improvement, and every trace passes the CSA checker.

The result report is
`paper/q1/experiment_outputs/directional_bbob_power_extension_20260718/bbob_confirmation_report.json`
with file SHA-256
`cc36f126734d6d1acfeb721a169f124bda9ed63c493cb95f0cfc2613c3e5cefe`.

## Claim boundary

The evidence supports a new, narrower finding: backbone-specific direction
and step instructions improve semantic candidate quality across CEC 2022 and
BBOB states. It does not establish statistically confirmed final-optimizer
superiority. The host-enforced authority and provenance claim remains valid;
the expanded PSO final-performance result is stronger but remains
inconclusive under the predeclared gate, and the DE result is unfavorable.
