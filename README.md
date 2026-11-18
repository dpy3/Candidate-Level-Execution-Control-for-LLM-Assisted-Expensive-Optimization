# L-ADE-CSA public code and audit release

This public release accompanies the manuscript *Candidate-Level Execution Control for LLM-Assisted Expensive Optimization* by Jeng-Shyang Pan, Peng-Yu Duan, Shu-Chuan Chu, Zhenyu Meng, and Václav Snášel. Manuscript source and local absolute paths are intentionally excluded.

## Evidence boundary

The supported primary contribution is host-enforced authority and auditable provenance for semantic-derived evaluations. The snapshot also contains bounded evidence that backbone-specific direction-step prompts improve candidate quality. It does **not** claim task-independent acquisition superiority or final optimizer dominance.

No new acquisition experiment is launched by this snapshot. Frozen proposal caches can be used for offline replay, but live proposal generation requires a separately configured third-party model service and user-supplied credentials.

## Layout

- `safe_llm_saea/`: curated optimizer, CSA checker, experiment runners, and tests.
- `artifacts/`: frozen protocols and processed reports used by the final claim audit.
- `audits/`: claim-evidence, citation, partial-intervention, and submission audit records.
- `ARTIFACT_INDEX.md`: claim-to-artifact mapping with hashes and API/replay boundaries.
- `MANIFEST_SHA256.txt`: SHA-256 manifest for the snapshot.

## Installation and checks

Python 3.10--3.12 is recommended. The current `ioh` package may require a local C++ toolchain when installed under Python 3.13 because a compatible wheel may be unavailable.

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest safe_llm_saea/experiments/test_csa_protocol.py safe_llm_saea/experiments/test_csa_fault_injection.py safe_llm_saea/experiments/test_directional_semantic.py safe_llm_saea/experiments/test_partial_direction_intervention.py safe_llm_saea/experiments/test_trajectory_commit_risk.py
```

## Public-release scope

API keys, unredacted service responses, local environment files, bytecode, and large raw caches are excluded. Frozen proposal caches and processed audit records support offline replay; live proposal generation requires a separately configured external model service and user-supplied credentials. The frozen performance study does not establish task-independent optimization improvement.

## License

Code is distributed under the MIT License. Manuscript text and third-party benchmark assets may be subject to separate publication or upstream terms.
