# Candidate triage

Accepted new directions:0. No recommendation is supported strongly enough for a top-conference claim.

Rejected current formulation: add joint uncertainty to a general time-series world model. The native gate demonstrates a pointwise/joint difference, but ordinary Gaussian calibration repairs much of it and functional/autoregressive/generative approaches already exist. Adding a covariance head or cache would be an incremental implementation choice.

Unresolved question, not an accepted idea: can a reusable dynamical model preserve system behavior under short-context adaptation while retaining useful probabilistic state updates? DPC/KAFFEE already addresses a substantial version. A distinction involving partially observed systems or changing mechanisms is only a possible test setting; ordinary filters, switching SSMs and functional models are strong alternatives. No evidence currently warrants a new architecture.

Next priority is to inspect and, if feasible, reproduce a genuinely generative dynamics baseline such as DynaMix under its native contract before selecting another failure mechanism. Do not require actions or Bayesian posteriors from an autonomous point model. Stop any candidate whose failure disappears under existing controls or a corrected preprocessing protocol.
