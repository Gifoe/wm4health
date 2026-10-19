# General time-series world models: concrete candidate development

Mode: exploratory; assumed venue family: general CCF-A machine learning. No paper-ready novelty or accepted method is claimed. The human requires a general modeling contribution at the WorldTS level; a large shared pretrained model is optional. Prior completed experiments remain sealed.

## Source-grounded cards

1. WorldTS2609.31162v1 and pinned executable core: causal patch states,per-window RevIN statistics,conditional fixed-horizon prediction and separate decoding. Source fact: identical raw overlapping patches can be expressed using different window statistics. Inference: reusing cached states may require coordinate handling. Forecasting failure or practical overhead has not been measured for this lead.
2. LatentTSF2602.00297: frozen pretrained AE provides predictive coordinates; public checkpoints/code are available. The exact AE variant and normalization contract must be checked rather than assuming all variants use RevIN.
3. OSSM2609.33566: observation correction versus autonomous propagation is already explicit. Classical observers and controlled state-space models are required controls,not a new mechanism to rename.
4. Pythia2610.05240v1: shared contextual predictive pretraining and probabilistic readout exist. Its default target space is preserved from a numerical foundation model; public code/model availability is not established. A point predictive state and a distribution over uncertain system mechanisms are distinct objects,not interchangeable labels.
5. Does Normalization Choice Matter for Causal Large Time-Series Models?2606.09954: causal/initial-statistic normalization and caching already have direct recent precedent. This overlap removes generic “causal normalization enables streaming” as a novelty claim.

## Candidate A: coordinate handling for reusable predictive states

Parent: WorldTS plus causal-normalization work. Operator: refine. Task: update a persistent latent state as the observation window and reference statistics change. Root challenge: cached states may use a different coordinate reference from newly encoded states. Possible mechanism: an explicit low-cost state transport,with reconstruction/re-encoding and fixed/causal statistics as existing controls. Contribution type: conditional inference/system method,only if a real accuracy-cost residual is established across systems.

Overlap challenge: causal normalization already supports caching and state reuse. Revised question: does exact compatibility with a moving-window predictive representation create a measurable unresolved cost or accuracy tradeoff beyond these alternatives? Evidence challenge: the native WorldTS encoder may be so cheap that exact re-encoding is sufficient. First test that cost and the representation change on historical trained weights. Do not develop transport merely because coordinate drift is mathematically possible. This is a needs-evidence lead,not a new-normalization claim.

## Candidate B: reusable uncertainty over system mechanisms

Parent: WorldTS/Pythia predictive states plus meta-dynamics literature. Operator: transfer. Task: maintain uncertainty over a system's persistent dynamics from short histories and reuse it across a set of future conditional queries. Root challenge: a probabilistic output head can model future values without necessarily preserving a coherent persistent mechanism hypothesis across updates/branches. Existing latent NeuralProcesses,Bayesian system identification,HiP-MDPs,RP-GSSM and belief-JEPA models are strong overlap risks. A classical posterior sampler is a control,not a proposed invention. A new claim would need finite-resource evidence showing useful reusable uncertainty beyond these methods. This is a structurally different fallback marked needs-search/needs-mechanism. It is not the old unidentified-noise-coupling counterfactual problem.

## Candidate C: conditional transition reuse beyond direct horizon heads

Parent: WorldTS plus OSSM/KUMA-style state propagation. Operator: combine. Task: one learned transition mechanism should support distinct known driver sequences and update schedules with a consistent observation interface. Possible interaction: separating observation correction from controlled propagation while retaining parallel predictive training. Generic controlled SSMs,GRUs,NeuralCDEs and action-conditioned JEPAs already cover much of this. Earlier ordinary FIR/GRU repairs and action-memory diagnostics prevent repeating the basic future-response claim. A residual must concern an explicit accuracy/data/compute tradeoff that existing causal recurrent models have not resolved on a compatible task. Mark needs-evidence;do not name a new architecture from the component list.

## Current selection

Test A first because its native source,trained checkpoints and a cheap falsifier exist. Retain B as the different fallback. C is currently a protocol/method interaction lead with substantial overlap risk. None is a final recommendation. Ordinary exact re-encoding being cheap would substantially weaken A;it would not prove every streaming world-model problem solved. An unrelated classical universal method or a similar title does not by itself settle a precise finite-resource residual.
