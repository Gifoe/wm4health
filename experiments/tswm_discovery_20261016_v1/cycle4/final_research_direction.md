# Current research decision

**No final research direction has passed the required evidence and novelty checks.**
This file is an evidence checkpoint. The autonomous discovery objective is ACTIVE.

**Research Question.** Can compressed probabilistic time-series world states be
tested for retaining action-dependent uncertainty, rather than merely fitting
conditional means and a variance head on an insufficient state?

**Evidence.** Native VJEPA component checks establish the declared gradient and
mean-only planning paths. A separate controlled stochastic-process experiment
shows accurate means can coexist with discarded variance-driver information and
larger risk regret. These do not yet prove failure of a trained native JEPA model.

**Existing Methods.** Ordinary heteroscedastic likelihood, predictive-moment SDR,
probabilistic SSMs, GSAVE and conditional-distribution representations already
address important parts. They must be tested before calling this unsolved.
Towards Financial World Modeling already probes volatility; no generic probe novelty.

**Proposed Idea.** The current toy's information-recovery problem is resolved by
an ordinary moment control, so its new-method route is stopped. Screening a
native-model distributional-sufficiency audit remains possible. No architecture
is endorsed. Any eventual method
would need to address an actual residual estimation/efficiency problem beyond the
existing controls, not rename the current moment or likelihood repair.

**Falsification.** Stop this as a new-method direction if existing probabilistic
representations/SDR repair the issue at comparable information, nuisance burden,
capacity and compute, or if independent data/native models remove the phenomenon.
The current attached-repair pilot already failed its all-three-initialization gate.

**Feasibility.** Current server RTX5090, pinned MotionJEPA/VJEPA code and saved
controlled generators. Small models take seconds per fit; native architecture,
independent cohorts and appropriate established controls remain required.

**Next Experiment.** First verify an unaddressed native-model/protocol claim against
existing distributional-representation and SDR work. Only if such a precise claim
survives, freeze independent data cohorts and test the existing moment/joint-density
controls against a native probabilistic representation, using fixed checkpoints,
held-out conditional scores and SCM decision regret. Do not spend more training
on the solved linear-mixing toy. Otherwise switch to another failure mechanism.

This is the only watchlist priority, not a top-conference-ready recommendation.
If the controls resolve it, return to broader discovery; do not force an idea.
