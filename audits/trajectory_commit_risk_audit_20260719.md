# Trajectory commit-risk audit (2026-07-19)

## Failure localization

The 32-case-per-backbone BBOB power extension shows that every non-tied final
effect follows a true incumbent-improving population commit. DE has six wins
and six losses among 12 commits; PSO has 12 wins and three losses among 15
commits. All abstained, evaluated-but-not-committed, and non-improving cases
tie their paired native trajectories. Immediate incumbent improvement is
therefore necessary for the current commit rule but does not authorize a claim
of favorable downstream trajectory impact.

## Delayed terminal commit

A post hoc non-interference diagnostic evaluates each originally admitted
candidate under one-use authorization, retains it outside the active
population, preserves the native trajectory, and commits only if it still
beats the native final incumbent. Both backbones produce 0/32/0: no negative
transfer occurs, but no candidate survives as a terminal improvement. The
semantic candidate's observed value comes from changing the subsequent search
trajectory, not from remaining independently competitive at the final budget.

Report SHA-256:
`2443c1f5c2c3b8e4578fc2599bef714098af23ae91d37fc425877f5fa5a48f27`.

## Paired continuation data

The first trajectory-development protocol reuses cached candidates from BBOB
instances 1--5 and runs every candidate under ten paired continuation seeds.
It produces 120 candidate events and 1,200 native/commit trajectory pairs.
The second development protocol applies the same procedure to instances
6--13, producing 192 candidate events and 1,920 pairs. No API calls occur in
either study. Together they provide 312 candidate events and 3,120 paired
continuations.

- Instances 1--5 report SHA-256:
  `df73e672ecf39fc0243201a298e6c7b5b83b60f035bfda2cdeff3df5fc202f04`.
- Instances 6--13 report SHA-256:
  `b4fb87c8fd437c91ad2af183059faa99086162036bbca20d5ca47791d218cc4d`.

## Trajectory-risk models

The compact model uses immediate regret, surrogate mean regret, and surrogate
uncertainty. Instances 1--3 fit separate ridge models, instance 4 calibrates a
one-sided residual radius, and instance 5 is held out. DE rejects every commit.
PSO admits one holdout commit, which has an unfavorable mean trajectory effect.
On the retrospective instances 6--13 check, both models reject every commit.
The model is therefore fail-closed but operationally equivalent to strict-zero.

The geometry model expands development to instances 1--11 and adds step scale,
candidate-to-centroid, nearest-neighbor and replacement distances, the
post-replacement diversity ratio, centroid alignment, and PSO velocity
alignment. Instances 12--13 remain held out. The calibration radii are 0.781
for DE and 0.967 for PSO. Both backbones again reject every holdout commit, so
the required nonzero-commit and wins-exceed-losses gates fail and
`future_confirmation_eligible=false`. Reserved instances 14--21 are not
captured or executed.

- Compact-model report SHA-256:
  `1b4c3b0204c0914d5b7a01678f09b57b135c0965b87765c6d04113515f04ec30`.
- Geometry-model report SHA-256:
  `00be1b03c182dc6b6d5d1aa000b801f36c6cb4913678a76bc3fff9a58f8aa940`.

## Decision

Do not launch another final API batch with either trajectory-risk model. The
next method change must alter how semantic information influences the
optimizer rather than add another scalar admission model. A defensible next
design is diversity-bounded partial intervention: inject the direction into a
trial-vector or velocity update with a host-controlled mixing coefficient,
reject interventions that reduce population or swarm diversity beyond a
predeclared limit, and restore the native state after a short numerical
lookahead when progress is unfavorable. This design requires a new development
study and a separately frozen confirmation; it is not established by the
current evidence.
