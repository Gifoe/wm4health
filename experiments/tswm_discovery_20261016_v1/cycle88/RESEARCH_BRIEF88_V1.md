# C88: learned state/mechanism belief audit - candidate, not accepted novelty

Overall goal ACTIVE/UNACHIEVED. Final directions accepted0. Previous turn made progress: C87's limited negative study was published at db360550896d50835df4a04018b73ad51cf75e6a. Its numerical failures remain failures; do not restart its closed runs or reinterpret its count as a certified continuous-time comparison.

## Question under review

When an observation stream changes, can a learned time-series world model infer the current state without unnecessarily overwriting reusable dynamics, and update dynamics when the change is persistent? This generic question is old: dual/joint filtering, switching state-space models, APE/MOCA, HiP-RSSM, MTS3 and newer adaptive world models are direct prior art. No new architecture or method claim is accepted from this wording.

The only potential residual question is a measured limitation of *learned* partial-observation dynamics inference under matched source supervision and online compute. It must survive ordinary prior-aware alternatives and correct ambiguity accounting. C31's proposed known-transition-family tracking example is not a substitute for this learned-world-model question.

## Fidelity and information contract

- Main contenders learn from TRAIN observations/actions and allowed next-observation targets only. No physical state, family label, parameter value, event cause or event boundary is an inference feature. Diagnostic probes using such labels are explicitly separated.
- Hand-coded true dynamics, known-parameter IMM/UKF and exact simulator posterior are privileged diagnostic references. Their success cannot by itself close a learned-model research question.
- Ordinary learned comparators must exploit shared source priors, not only fit one short test trajectory. Source-learned Bayesian feature dynamics with a switching/run-length filter and MOCA/ALPaCA-style inference are relevant; they must have the same allowed information and fair tuning budgets.
- Use at least two nonlinear controlled families with genuinely partial observations for any scientific result. A linear/Gaussian fixture is a qualification/reference calculation, not evidence that the full question is solved or unresolved.
- Forecast queries must be scheduled independently of hidden event times. Mechanisms/exogenous shocks are held fixed after the query in the declared simulation task; no unseen future event time is supplied. Known future action plans are shared by every method. Post-event-age strata are analysis labels only, never inputs.
- Both transient state shocks and persistent dynamics changes, plus observation-noise/input-distribution controls, must appear in every learner's TRAIN/VAL data before any held-out claim. Do not create a bad baseline by withholding the relevant event class.
- Causes that produce indistinguishable observable histories require uncertainty, not forced cause classification. Forecast quality/calibration are primary; a hidden-parameter classifier is not required for faithful simulation.

## Before another learner campaign

1. Verify closest work and actual source code, especially native MTS3/HiP-RSSM task/state updates and stronger continual-adaptation papers. Treat paper titles and archive uploads as unverified leads until their content/assumptions are read.
2. Qualify the native implementation/gradients and allowed-mask/forecast interface. Record source commit, runtime, precision and all failures. Use task-local dependencies; do not downgrade the shared Torch environment or enable external experiment logging.
3. Freeze the learned-data task, strong ordinary comparator, source-competence criterion and matched source/online budgets. No new architecture training before this is concrete.
4. Screen a developmental source pool first. Only a reproducible excess forecast error left unexplained by training, inference support, observation ambiguity, capacity and ordinary controls can justify a candidate method experiment.

## Rejection conditions

Retire a scoped new-method claim if existing prior-aware learned filters suffice, if only a deliberately mismatched fixed-context implementation fails, if the difference is just inaccessible information, or if closer work already establishes the same mechanism/solution under the same assumptions. A loss on one dataset is not a research gap. A protocol/source audit is progress toward experiments, not completion of the research objective.
