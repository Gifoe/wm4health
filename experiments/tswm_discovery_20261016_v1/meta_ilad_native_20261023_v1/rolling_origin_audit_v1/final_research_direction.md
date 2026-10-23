# No accepted final direction

**Research Question:**When a qualified native model forecasts from a later observation window, is an apparent failure caused by dynamics, incorrect forcing time or state reinitialization?

**Evidence:**300 queries/75 environments/five origins. Reset clock produces22–34 MSE at intermediate phases; correct time gives0.23–0.40, ordinary state fitting0.18–0.20 there. Independent risks, selections and native replay pass. Individual state-fit future regressions remain preserved.

**Existing Methods:**The native absolute-time input and ordinary prefix-only state fitting suffice to avoid the broad failure in this regime.

**Proposed Idea:**None accepted. The local inverse-inference tradeoff is not by itself new.

**Falsification:**The anticipated broad late-origin defect does not survive correct timestamps and ordinary controls. Per-sample universal improvement is also false.

**Feasibility:**Original source/checkpoint on the existing5090, no new neural training or data download.

**Next Experiment:**No optimizer-budget tuning on these cases. Any follow-up needs an independent capability/source and a prior-art distinction that survives ordinary inference alternatives, rather than manufacturing a failure by changing the clock. Overall goal ACTIVE/UNACHIEVED.

[Results](experimental_results.md) · [literature](literature_review.md) · [registry](baseline_registry.md) · [negative evidence](negative_evidence.md) · [hypotheses](hypothesis_space.md) · [triage](research_ideas.md).
