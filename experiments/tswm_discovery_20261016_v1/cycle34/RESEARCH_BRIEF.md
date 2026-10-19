# C34 research intake: reusable time-series world-model mechanisms

Date: 2026-10-17 Asia/Shanghai. Goal remains ACTIVE_UNACHIEVED, accepted final directions0. C33 actual native controlled experiment is completed and closed as a final-method motivation, not a reason to declare the field solved.

User scope: a reusable time-series world-model framework comparable in research level to WorldTS. Per-dataset training is allowed. Do not impose universal zero-shot/shared-checkpoint, action/control, causal intervention, physical-state semantics, Loop, LLM or online-switch requirements. A dataset-specific optimization is insufficient. A useful finite-data/compute interaction of known components can qualify; it need not be impossible for every classical method with unlimited resources.

## Frozen questions before search

1. Among actual recent numerical TS world-model frameworks, what predictive/state/dynamics capabilities are claimed and empirically supported, under which input and training contracts? Distinguish source facts, reported results and untested inference.
2. Which cross-task capability tension or failure remains worth measuring after task-matched ordinary inference/training/representation controls? A source bug, mismatched inputs, one bad score or theoretical impossibility under unavailable information does not establish a new architecture problem.
3. What concrete mechanism interaction, closest existing methods, new independent public protocol and small decisive RTX5090 experiment would distinguish a defensible candidate? Supply a constructive minimum viable question or rescue route before recommending a pivot; component prior art alone is not a rejection.

## Three independent clean-context perspectives

- Mainstream: actual framework task contracts and native evidence, excluding forced physical/causal/zero-shot definitions.
- Critics/methodology: empirical tension and discriminating protocols, including competent strong controls and what a negative result would mean.
- Adjacent constructive mechanisms: finite-budget combinations with a concrete non-obvious interaction, closest-overlap challenge, and executable baseline/benchmark.

Primary search window 2026-04-17..2026-10-17, with necessary earlier background labeled. Primary author papers/code/proceedings only. Verify source identity and the actual claim/full text separately. Unconfirmed references remain unused. No publication, remote training or large data download in this intake. Each perspective writes its own evidence note; no agent edits others' files or chooses an architecture from test scores.

## Current evidence and boundaries to avoid repetition

C33 used fixed original WorldTS stage1 seed2021 on NP/PJM H24:4 ordinary decoders×3 decoder seeds×2 tasks=24fits;20epochs,22820updates/fit. All training/selection sealed before56TEST records. Two independent audits passed112score checks,120sample-order schedules, raw-data/weight/cache/selection/replay/provenance. Source revision269426ea6217d7152a244da0e209d922acd46d4f.

Removing reconstruction gives primary MAE NP1.557%,PJM0.219%; native-objective selection reduces that to0.513%/0.491%,PJMseed signs mixed. Ordinary global MLP improves MAE vsnative prediction-only5.600%/1.402%,butPJM MSEworsens3.809%,13/24fits still improve atcap; only decoderseed replication and two old related electricity tasks. Its global aggregation/extra params/explicit existing RevIN statistics are bundled,not uniquely attributed. No latent-information-insufficiency gap or novel method accepted; no further NP/PJM tuning on this route.

Earlier negative boundaries (do not blindly revive): generic action memory, shared-seed counterfactual nonidentification, covariance/particle belief repairs, causal-flow velocity masking, ordinary controlled system-ID, frozen-state adaptation and readout-aware/epoch1 selection. They were evaluated on particular tasks,not proof that all related finite-budget combinations are solved. Inspect cited cycle reports if an exact overlap matters; never read/execute cycle6 secret launcher. No new generator merely chosen to force an existing model to fail.
