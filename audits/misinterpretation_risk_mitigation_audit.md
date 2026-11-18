# Misinterpretation-risk mitigation audit

Updated: 2026-07-20

## Risk 1: ordinary LLM + DE engineering combination

Status: mitigated in the manuscript story.

- Abstract now states that DE is a reference instantiation of an optimizer-independent authority contract, not a new LLM--DE mutation operator.
- Introduction defines the novelty as the enforceable boundary between an upstream proposal source and host-owned evaluation/update operations.
- Contribution 3 now treats archived benchmark/application rankings as descriptive viability context rather than the source of novelty.
- Table 1 describes the LLM as an upstream proposal source with no objective/update authority.
- Figure 1 caption identifies DE as the reference backbone and CSA as the backbone-independent authority layer.
- The returned proposal is defined as a source record followed by six controlled states, eliminating the apparent six-versus-seven-state inconsistency.

Claim: CSA reserves objective invocation and outcome-bearing updates for host code. | Evidence: Definition, Algorithm 1, Figure 1, `csa_protocol.py`, 88/88 final-code authorization. | Status: supported.

Claim: CSA is optimizer independent. | Evidence: DE reference implementation plus CSA-wrapped CMA-ES portability/strict-zero closure. | Status: supported as an interface claim, not as universal performance transfer.

## Risk 2: implied performance superiority despite mixed evidence

Status: mitigated by claim-type separation.

- CEC heading changed to `Archived Benchmark Competitiveness`; its GPT/Claude rankings are explicitly descriptive and non-causal.
- WSN results are described as structured-task viability; the near-null fresh closure is stated in the same paragraph.
- UAV rankings are described as archived ordering based on independent proposal streams and are not attributed to CSA.
- Cover letter removes rank-first and cost-improvement headline bullets and leads with authorization closure, fault detection, and negative performance boundaries.
- Discussion states that archived ordering changes across sources and cannot identify a source-invariant winner.
- Abstract and conclusion retain the unfavorable/inconclusive frozen final-performance boundary.

Claim: direction-step prompts improve candidate quality. | Evidence: DE 0.476 and PSO 0.349 incumbent-improving rates plus unseen BBOB confirmation. | Status: supported as a bounded secondary finding.

Claim: CSA or direction-step prompting improves final optimization performance across tasks/backbones. | Evidence: frozen BBOB power and partial-intervention confidence intervals cross zero; DE/PSO partial-intervention means are unfavorable. | Status: not supported and explicitly prohibited.

## Risk 3: military UAV interpretation or inadequate data sharing

Status: military interpretation mitigated; repository identifiers remain author/account actions.

- Main UAV subsection is renamed `Archived Civilian UAV Evaluation`.
- Main text, Supplementary Appendix D, and cover letter define the simulations as civilian inspection, logistics, and emergency-response route planning.
- The manuscript explicitly excludes military, defense, targeting, surveillance, hostile-agent, and adversarial-operation scenarios.
- Anonymous Data Availability names the 668-file frozen evidence archive and explains offline inspection and credential exclusion.
- Author Data Availability maps the software snapshot and research-data archive separately and states that GitHub is not a persistent archive.
- The external availability statement requires an anonymous reviewer link during review and public software/data identifiers after review.

Claim: the UAV study provides deployment-level or military safety evidence. | Evidence: simulator only. | Status: not supported and explicitly excluded.

Claim: supporting evidence is packaged for review. | Evidence: `open_source_snapshot_20260720.zip`, `eswa_frozen_evidence_bundle_20260720.zip`, artifact index, and SHA-256 manifests. | Status: supported locally; anonymous reviewer link and public DOI remain unresolved.

## Reverse outline

1. Abstract: authority gap -> CSA contract -> closure/fault evidence -> bounded candidate-quality finding -> no final-performance superiority.
2. Introduction: expensive-evaluation authority problem -> prior semantic participation gap -> optimizer-independent contract -> claim-typed evidence -> three contributions.
3. Related Work: numerical budget discipline -> LLM participation roles -> missing candidate-level authority -> property-based comparison.
4. Methodology: authority boundary -> fail-closed admission -> one-use authorization -> six controlled states -> checker and trusted-host limit.
5. Experiments: archived viability -> matched authority tests -> final-code closure -> fault/runtime evidence -> bounded/negative acquisition studies.
6. Discussion/Conclusion: authority contribution established; acquisition, cryptographic provenance, compromised-host protection, and deployment safety remain open.

## Five-dimension adversarial review

- Contribution: pass. The manuscript now identifies a non-compositional authority/provenance contribution.
- Writing clarity: pass with minor density risk. Main narrative is coherent, but the paper remains long and historically layered.
- Experimental strength: pass for enforcement claims; explicitly not passed for universal performance claims.
- Evaluation completeness: pass for frozen controls, second backbone, fault injection, runtime stress, and negative boundaries.
- Method design soundness: pass within the declared trusted-host boundary; malicious-host and cryptographic guarantees remain excluded.

## Remaining submission blockers

- Corresponding-author telephone number.
- Anonymous view-only repository link for double-anonymized review.
- Public software/data DOI, release tag, commit hash, and data licence after archive publication.
