# Cycle10 literature and novelty boundary

Primary sources were checked on2026-10-16. Historical foundations are retained because they constrain novelty; recent preprints have no inferred venue status. Reading depth is stated explicitly. No snippet-only or policy-excluded source is evidence.

| work | source/status and reading depth | implication and limit |
|---|---|---|
| Jahn,Karnik,Schulman2025 | [CLeaR/PMLR275](https://proceedings.mlr.press/v275/jahn25a.html), abstract/protocol screen | finite unrolled graph segments can characterize latent time-series identification; not a license to ignore graph assumptions; proofs not fully audited here |
| Thams et al2024 | [JMLR25(302)](https://jmlr.org/papers/v25/22-0262.html), primary abstract, PDF and [official code](https://github.com/nikolajthams/its-time/) inspected | nuisance/corrected-past IV already handles temporal-IV pitfalls; requires valid structural/instrument assumptions, not executed in this cycle |
| Geiger et al2015 | [ICML/PMLR37 PDF](https://proceedings.mlr.press/v37/geiger15.pdf), primary theorem-setting screen | hidden-component VAR identification is old; its independent reduced-noise conditions are not automatically met by our instantaneousB construction |
| Reiter et al | [Process Graphs Part I,2305.11561v3](https://arxiv.org/html/2305.11561v3), frontmatter/abstract screen | process-level causal and spectral formulations predate this witness; full proofs not audited |
| CAFE2026 | [2609.32167v1](https://arxiv.org/html/2609.32167v1), assumptions/method/prior-sensitivity sections | amortized posterior counterfactual inference across observationally equivalent acyclic ANMs is established; fully observed independent-noise assumptions differ from hidden temporal setting |
| CSA-PFN2026 | [2605.10590v1](https://arxiv.org/html/2605.10590v1), method/setting screen, priorcycle also checked | static amortized causal sensitivity exists; a dynamic claimed extension needs more than renaming this idea |
| PSG-JEPA2026 | [2608.06799v1](https://arxiv.org/html/2608.06799v1), sections1-3 | directly uses proprioceptive and multi-horizon state-change supervision and explicitly discusses endpoint inverse-action ambiguity; simple state-grounding rescue is covered |
| ACPC2026 | [2608.12939v1](https://arxiv.org/html/2608.12939v1), sections1-3 | frozen shared-action rollout consistency plus separation diagnostic; perturbation bounds are not absolute model-truth certificates |
| CoCo2026 | [2608.04653v1](https://arxiv.org/html/2608.04653v1), abstract/method screen | counterfactual action/observation consistency and same-state multi-action diagnostics already exist; full native training not reproduced |
| Dueling World Models2026 | [2608.06706v1](https://arxiv.org/html/2608.06706v1), method/limitation screen | action-centering removes common-mode effects; this does not by itself establish sufficiency for action-correlated nuisance or full state dynamics |
| Gunn Kim2026 | [Path-Space World Models,2606.28751v1](https://arxiv.org/html/2606.28751v1), sectionsI/II | already studies path-law and irreversibility in a controlled Markov/isotropic regime; explicitly leaves other architectural channels open, so do not attack an unclaimed necessity theorem |
| Ji et al2026 | [Phys Rev Research8,023264](https://journals.aps.org/prresearch/abstract/10.1103/1ncx-6cxk), publisher abstract and metadata | LENS learns irreversible state representations via forward/reverse classification; existing baseline for any proposed arrow-of-time representation. Native code/complete theorem inspection still pending |

The first six delimit the current equivalence audit. The latter six are exploratory pivots or negative signals, not accepted directions. All cited method results are author claims unless explicitly reproduced in the local experiment registry.

## Opportunity map

Temporal structural ambiguity: covered central claim; possible rescue is a realistic assumption-robust identified-set task on which current profile/IV/posterior controls fail. Current Gaussian witness is insufficient. Inverse-action sufficiency: crowded and partly directly covered; adding state supervision or generic conditional scores is weak novelty. Observable stochastic path fidelity: remains a possible mechanism-level discriminator, but LENS, path-space formulations, proper joint likelihood and existing response probes are mandatory strong controls. No experiment or novel method is claimed for this pivot yet.

Excluded internal screening: MDPI/preprints.org and synthetic evaluation-only pages, secondary summary pages, inaccessible snippet-only claims. No inferred acceptance or invented author result is used.
