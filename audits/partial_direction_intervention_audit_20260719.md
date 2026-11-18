# Diversity-bounded partial direction intervention audit (2026-07-19)

## Method change

The semantic direction no longer replaces the worst population member. DE
adds a fraction of the selected incumbent-relative direction to one native
best-target trial vector; normal DE selection alone decides whether that trial
enters the population. PSO adds the same fraction to one native global-best
velocity; normal personal-best updating controls acceptance. A prospective
host gate rejects the intervention when the resulting diversity ratio is
below 0.95. Default optimizer execution has no pending direction and follows
the unchanged native branch.

## Development

BBOB instances 1--13 compare mixing coefficients 0.25, 0.50, and 0.75 under
ten paired continuation seeds per state. The development protocol was frozen
before execution with SHA-256
`194c9816eec95a7033943585b026cc16733753f690c8716c0d5dc186f78974c9`.

| Backbone | Alpha | W/T/L | NTR | Mean change (95% bootstrap CI) | Accepted | Gate pass |
|---|---:|---:|---:|---:|---:|---:|
| DE | 0.25 | 6/30/16 | 0.308 | +3.29% [+0.32%, +8.19%] | 54 | No |
| DE | 0.50 | 13/29/10 | 0.192 | +4.88% [-0.15%, +14.61%] | 49 | Yes |
| DE | 0.75 | 9/34/9 | 0.173 | +0.71% [-0.08%, +1.69%] | 32 | No |
| PSO | 0.25 | 9/36/7 | 0.135 | +0.02% [-0.46%, +0.49%] | 24 | Yes |
| PSO | 0.50 | 10/36/6 | 0.115 | -0.39% [-1.10%, +0.20%] | 25 | Yes |
| PSO | 0.75 | 6/36/10 | 0.192 | +0.01% [-0.84%, +0.86%] | 26 | No |

The frozen common selection is alpha 0.50. Its operational gate requires
nonzero accepted interventions, wins exceeding losses, and NTR no greater than
0.25. The gate does not establish a favorable mean effect; this distinction is
retained in the confirmation report.

Development report file SHA-256:
`6a20b826650da27be53373094a3d8c7c05e250f3cc53ed333a48d2d7f8fbe029`.

## Frozen confirmation

The confirmation uses unseen BBOB instances 14--21, four fixed functions, 10D,
alpha 0.50, diversity floor 0.95, and ten paired continuations per state. It
captures 64 fresh backbone-specific calls containing 340,178 tokens. Every
semantic-derived point that reaches the objective receives a one-use runtime
authorization, and all traces pass the CSA checker.

| Backbone | W/T/L | NTR | Mean change (95% bootstrap CI) | Applied | Accepted | Diversity rejects | Operational gate | Scientific effect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DE | 11/15/6 | 0.188 | +1.16% [-0.89%, +4.51%] | 142 | 44 | 38 | Pass | Not supported |
| PSO | 6/20/6 | 0.188 | +4.24% [-0.22%, +12.90%] | 67 | 12 | 143 | Fail | Not supported |

DE passes the three predeclared operational gates, but its mean effect is
unfavorable and the interval crosses zero. PSO does not win more states than it
loses and also has an unfavorable mean. Consequently,
`any_backbone_claim_eligible=true` records the literal operational protocol,
whereas `any_backbone_scientific_effect_supported=false` controls the paper's
performance interpretation.

- Frozen protocol SHA-256:
  `429f420b4bcba125e2bd2f42a4dea4a67f6ef2eb8bcddef462106cc2c129ca4b`.
- Confirmation report file SHA-256:
  `1fe1df8085761c37f429d40a63812fd2d8902580a17e6c516a9004a7363e90c4`.
- Frozen protocol file SHA-256:
  `cb768fcf37efc6e190097f627e37a2dde9803ee84580bd34211545c5f85a8fd6`.

## Decision

Stop acquisition-effect optimization for this paper. The direction-step
interface improves candidate quality, and the host can constrain, authorize,
and trace partial interventions, but neither direct commit nor diversity-
bounded partial intervention establishes favorable final performance across
DE and PSO. The defensible paper position remains a host-enforced authority
and provenance framework with candidate-quality improvement as a bounded
secondary finding.
