# Candidate status — final direction not yet accepted

**Research Question:** Can synthetic trajectory filtering create initial-state/mechanism associations that impair a shared time-series world model's ability to infer and simulate the same dynamics under new initial conditions, and how should pretraining preserve useful mechanism coverage while rejecting genuinely invalid numerics?

**Evidence:** C86 executes a pinned public sampler method on exact finite trajectories, runs18 factual-learning fits with matched retained-window/optimization budgets, and evaluates two newly opened mechanism pools. The generated experimental_results.md reports all metrics, ordinary controls, crossed-query attribution and independent checks. These are direct controlled observations in a small family, not evidence about every pretrained world model.

**Existing Methods:** Censored system identification, ordinary local identification, initial-state augmentation and causal generator audits already exist. The candidate distinction is the many-task synthetic-pretraining interaction and mechanism-preserving data intervention. Its novelty and relevance to actual released models remain unproved.

**Proposed Idea:** No new architecture accepted. Mechanism-preserving initial-state resampling and finite-prefix retention are existing data operations used as falsification controls. A new general method is not inferred from their favorable raw-MSE rows.

**Falsification:** Retire the method-necessity claim if competent ordinary approaches suffice. Reject universal improvement if normalized errors or retained-cohort errors worsen. Reject attribution to a released checkpoint without its exact pretraining lineage and a paired causal comparison. If a native pretrained learner remains robust after matched-marginal re-pairing, the strongest shortcut claim fails.

**Feasibility:** Two-thread CPU diagnostic/training is complete; a native nonlinear controlled-model comparison with public code is feasible on the available RTX5090, after checking current jobs and resource costs. Raw arrays/weights/logs stay on the external15TB disk; code and compact verified evidence go to GitHub.

**Next Experiment:** Establish exact training-sampler provenance for a usable native dynamics model, then compare complete-trajectory rejection versus mechanism-preserving finite-prefix generation on at least two nonlinear controlled families, with matched source budgets, strong local identification, held-out mechanism/initial pairings and all error metrics frozen before evaluation. This step is not claimed completed by the current small-model study.

Final accepted directions:0. Candidate status:NEEDS_NATIVE_MODEL_EVIDENCE_AND_NOVELTY_DIFFERENTIATION. Overall objective:ACTIVE / UNACHIEVED.
