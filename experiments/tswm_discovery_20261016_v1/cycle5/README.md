# Cycle 5: native-model attribution and computational falsification

Research objective remains **ACTIVE**. No final direction is accepted.
Server directory: `D:\tswm_discovery_20261016_v1\cycle5`.
All previous cycles are preserved. These are private research records.

Completed evidence: one full released-code KUMA PEMS03 run, three full COFE-model
ETTh1 training/control runs, three native COFE component seeds, three trained
checkpoint objective checks, and 18 controlled Gaussian counterfactual cases.
Interrupted first attempts remain separate and are excluded from completed fits.

- KUMA: test MSE 0.063033101, MAE 0.166339368, 14 epochs,
  validation-selected checkpoint; unmodified released engine, no batch caps.
- COFE: a free noise-estimator head can be zeroed after training without changing
  predictive distribution parameters. This lowers its native objective.
  Correcting a dispersion mapping removes synthetic level-shift sensitivity.
  These establish objective/implementation limitations, not a novel method.
- Counterfactual pilot: ordinary Gaussian conditioning and generic SDP achieve
  the frozen 1% gap criterion in all 18 cases; maximum relative gap
  2.48756e-05. Stop custom solver development in this restricted regime.

Read `experimental_results.md` for numbers and protocol boundaries,
`negative_evidence.md` for stopped claims, and `final_research_direction.md` for
the current decision. No paper-table exact replication or top-conference-ready
research direction is claimed.
