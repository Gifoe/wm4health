# Exploratory primary-source screening

10primary papers in five groups: physical evidence; temporal control; observation processes; causal treatment sampling; control representations. Scores are provisional judgments of inspected sections, not reproduction scores or venue predictions.

- [Causal Chambers](https://www.nature.com/articles/s42256-024-00964-x) — Nature Machine Intelligence2025; archived arXiv2404.11341v2; method + benchmark; provisional insight/completeness/numeric 4/4/4. Real physical actuators, sensors and validated graphs support protocol-defined interventions. Recorded sensors are not automatically physical latent ground truth.
- [Time-Aware World Model](https://arxiv.org/html/2506.08441v1) — arXiv2506.08441v1,2025; pure method; provisional insight/completeness/numeric 3/3/3. Chosen time-step conditioning and mixed-rate training are existing methods. They do not make a realized unknown future measurement time an available deployment input.
- [EndoClock](https://arxiv.org/html/2608.09266v1) — arXiv2608.09266v1; paper reports MICCAI Medical World Models workshop2026 acceptance; system/tool; provisional insight/completeness/numeric 4/3/N/A tool. Audits where task evidence survives synchronization and endogenous observation clocks. Generic clock-preservation auditing is already covered. It is not a learned controlled-simulator guarantee.
- [HyperNSDE](https://arxiv.org/html/2610.07383v1) — arXiv2610.07383v1,2026October; pure method; provisional insight/completeness/numeric 4/4/3. Jointly models latent-state-dependent observation intensity, with matched-grid analyses and an open implementation. Generic joint time/value generation is not an untouched idea.
- [Stage-wise sensing reliability](https://arxiv.org/html/2609.07126v1) — arXiv2609.07126v1; workshop-style manuscript2026; method + benchmark; provisional insight/completeness/numeric 3/3/3. Separates sensing disturbance propagation through representation, prediction, planner preference and outcome. Internal perturbation size does not certify physical outcome; no broad real-sensor timing guarantee is claimed.
- [CIDER-FM](https://arxiv.org/html/2609.39523v1) — arXiv2609.39523v1,2026September; pure method; provisional insight/completeness/numeric 4/4/4. Pools observational and surrogate experimental regimes with interventional metadata; real light-chamber experiments are included. Its inspected architecture treats samples as sets, not the controlled time-series task executed here.
- [TESAR-CDE](https://proceedings.mlr.press/v202/vanderschueren23a.html) — ICML2023; archival fallback arXiv2306.04255v1; pure method; provisional insight/completeness/numeric 4/4/4. Inverse-intensity weighting for informative treatment-outcome sampling is existing work. Generic informative-sampling bias cannot be claimed as a new world-model problem.
- [TE-CDE](https://proceedings.mlr.press/v162/seedat22b.html) — ICML2022; archival fallback arXiv2206.08311v1; pure method; provisional insight/completeness/numeric 4/4/4. Continuous-time treatment-conditioned counterfactual prediction already uses neural controlled differential equations; its limitations do not make a generic irregular-time extension novel.
- [CF-JEPA](https://arxiv.org/html/2610.00727v1) — arXiv2610.00727v1,2026October; pure method; provisional insight/completeness/numeric 3/3/3. Uses inverse and adversarial dynamics to separate control-related and uncontrollable latents for planning. Relevant to a fresh nonlinear-reachability question, but not executed and not shown insufficient.
- [Preserving Unstable Modes through Inverse Dynamics](https://arxiv.org/html/2610.07540v1) — arXiv2610.07540v1,2026October; pure method; provisional insight/completeness/numeric 4/4/4. Endpoint inverse reconstruction has linear-system reachable-subspace guarantees and nonlinear visual experiments. Nonlinear guarantees are not asserted; no theorem is refuted by this cycle.


## Opportunities and boundaries

Physical evidence cluster: controlled wind/light experiments are an existing resource; the fresh contribution here is only private qualification. Residual: a physical-latent evaluation target would need separate measurement provenance.

Temporal-control cluster: chosen variable-rate control is already modeled by TAWM. Residual: realized sampling-time availability differs from a chosen action duration, but no practical unresolved method gap has been established here.

Observation-process cluster: EndoClock and HyperNSDE address evidence loss and joint observation-time generation. Residual: action/acquisition coupling could differ, but the proposed tachometer-blocking mechanism was falsified before a new model was built.

Causal-treatment cluster: TE-CDE and TESAR-CDE handle continuous-time treatment outcomes and informative observation weighting. A generic informative-observation reformulation is not novel. Precise conditional assumptions matter.

Control-representation cluster: CF-JEPA, EP-IDM and [Controllable-Koopman](https://github.com/KYMiao/Controllable-Koopman) are primary-code/theory anchors for fresh search. Their nonlinear scope and native definitions must be checked; a missing universal guarantee does not prove empirical failure. These papers were not executed in this cycle.
