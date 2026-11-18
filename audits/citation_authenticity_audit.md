# Citation authenticity audit

Frozen: 2026-07-20. Bibliography contains 34 cited entries and no unused entries.

## Automated multi-source result

- Multi-source verified: 11
- Single-source/manual-review flags: 20
- Not found by the checker: 3
- DOI records directly returned by Crossref in the secondary pass: 21/24

## Resolved high-risk records

- `srinivas2010gaussian`: real ICML 2010 paper; arXiv:0912.3995. The checker miss is an indexing false negative.
- `coello2002theoretical`: real Computers & Structures survey; DOI 10.1016/S0045-7949(01)00323-1. The checker/Crossref endpoint miss is retained as a manual-resolution note.
- `llm4eo2025survey`: arXiv:2509.08269, title and authors manually matched to the cited preprint. It remains a preprint citation.
- `mesbah2018stochastic`: previously chimeric metadata corrected to the 2016 IEEE Control Systems Magazine record, DOI 10.1109/MCS.2016.2602087.
- `liu2024llambo`: arXiv:2402.03921, title/authors manually matched; the automatic 0.83 title match was not accepted without this check.

## Submission interpretation

No cited entry is currently assessed as fabricated. However, database coverage is not complete: the raw JSON deliberately retains 20 single-source flags and 3 checker misses. The unresolved publication-management risk is not authenticity but versioning: preprints should be replaced by formal proceedings/journal records only when the exact accepted metadata is confirmed.
