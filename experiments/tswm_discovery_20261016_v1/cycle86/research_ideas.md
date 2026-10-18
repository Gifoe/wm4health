# One candidate retained for evidence development

Candidate question: how does trajectory-validity selection in synthetic pretraining alter a time-series world model's learned mechanism prior and its robustness to new initial-state/mechanism pairings?

Parent: the unresolved C83 pretraining-population lead. Operations: instrument the actual sampler; isolate selection from numerical failure; break initial-state/mechanism association while preserving mechanism IDs; then cross fixed marginals at evaluation. C85's refuted need for new calibration is not revived.

Potential contribution type: empirical finding plus data/evaluation protocol; a general method would require additional evidence. Selection bias itself and resampling initial conditions are established. The non-obvious interaction to establish is whether common synthetic-data filtering creates a transferable dynamics-learning shortcut in competent existing world models, and whether a practical sampler correction improves held-out simulation under a matched budget.

Current weakness: only one controlled symmetric-linear family and two small predictors; ordinary identification is strong; performance depends on the metric and query population; no released-checkpoint failure is causally explained. Development potential: the causal training-data intervention is more specific than labeling a low score as bad dynamics, and can be extended to nonlinear controlled families using existing native architectures.

Rescue route: if a general predictor improvement does not survive strong baselines, study the narrower benchmark-validity question: can reported few-shot mechanism inference be explained in part by sampler-created initial-condition association? This requires actual trained-model probes and pretraining provenance, not merely a sampler histogram. Do not recommend a paper until that test succeeds and the closest censoring/causal-audit literature has been fully differentiated.

Fallback: finish the existing C71 original-budget experiment and use its independently qualified free-simulation assessment to decide whether a separate modular-generalization failure survives ordinary controls. Its expected completion is not evidence of a gap.
