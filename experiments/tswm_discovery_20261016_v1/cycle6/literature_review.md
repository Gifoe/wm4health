# Literature and innovation boundary

Exploratory search, not exhaustive coverage. Public primary papers are archived
with exact version and SHA256 in outputs/literature/registry.json. A title/abstract
screen is not a reproduction. Older cycles retain earlier verified boundaries.

| Primary work | Boundary relevant to this audit | Verification scope |
|---|---|---|
| [CHiLD,NeurIPS2025](https://arxiv.org/html/2510.18310v1) | Hierarchical temporal latent distribution identifiability needs stated independent-view/operator/variability assumptions. It does not imply exact deterministic hidden-state recovery from a finite context. | Paper, pinned code, native full configuration and component audits; no theorem refutation |
| [Ada-Diffuser,2026](https://arxiv.org/html/2605.16054v1) | Future context is used in identifiable dynamic generation and refinement. Joint posterior/Gibbs refinement can be legitimate; self-generated context alone is not evidence of leakage. | Paper read; public executable code not verified |
| [CaRiNG,ICML2024](https://proceedings.mlr.press/v235/chen24ai.html) | Non-invertible observation maps differ from a non-bijective inverse innovation network. Do not conflate these assumptions. | Primary paper/appendix screening; no execution |
| [SurVAE,NeurIPS2020](https://arxiv.org/abs/2007.02731) | Proper surjective generative modeling and branch accounting are established. | Primary paper archive, scoped mathematical boundary |
| [Exploding inverses,AISTATS2021](https://proceedings.mlr.press/v130/behrmann21a.html) | Numerical inverse instability is established even for bijections; it is distinct from structural folding. | Primary PDF read; no execution |
| [Bayesian self-consistency,ICLR2026,v6](https://arxiv.org/html/2501.13483v6) | Uses prior/likelihood/posterior identities on unlabeled data, including real time series and images. Generic posterior consistency is not new. The earlier v1 toy scope is outdated. | Latest primary version verified; code link located, no execution |
| [Branch-JEPA,v3](https://arxiv.org/html/2607.05238v3) | Finite weighted latent trajectories with full-set Energy Score and decoded distribution scoring; a generic multimodal/proper-score JEPA proposal is covered. v1 MoP-JEPA does not define the current boundary. | Latest primary paper read; no native execution |
| [UWM-JEPA](https://arxiv.org/html/2605.25313v1) | Belief-structured blind rollout with joint unitary-spectrum preservation; paper distinguishes full-state invariants from reduced-state readout and limits long-horizon evidence. Generic belief preservation is not new. | Primary paper read; public GitHub link located, no execution |
| [COMET,2024](https://arxiv.org/html/2404.15109v1) | Reusable mechanisms and composition across dynamics environments predate a new compositional-world-model claim. | Primary paper read; no execution |
| [Mechanistic World Models,2026](https://arxiv.org/abs/2607.12474) | Mechanistic modularity is an existing conceptual program. | Abstract screening and primary PDF archive, not a full implementation audit |
| [World Models in Pieces,2026](https://arxiv.org/html/2606.24842v1) | Local transition certification from agent decisions uses discrete/anchor assumptions; neither generic certification nor unrestricted nonlinear guarantees can be assumed. | Primary method read; no execution |
| [ThermoForce,2026](https://arxiv.org/html/2607.03942v1) | Frozen passive forecast plus physics-structured forced response already targets intervention-sign/magnitude failures in HVAC. | Primary paper read; release states code/data forthcoming, no executed baseline |
| [Rethinking World Models for Biosignals,2026](https://arxiv.org/html/2610.17510v1) | Action latency, external drivers and endogenous dynamics already motivate intervention-validation tests. | Primary perspective read; not empirical method reproduction |

New search leads:ActTime (anonymous repository availability unresolved),
What Must a World Model Distinguish for Planning? (2609.33030), and MDN-VJEPA.
Do not cite secondary search summaries as executed evidence or accepted novelty
boundaries. Several reviewed papers are preprints; venue status is asserted only
where verified. Future-dated journal issue metadata is not proof of availability
at this cutoff. MDPI and preprints.org leads are excluded from evidence selection.
