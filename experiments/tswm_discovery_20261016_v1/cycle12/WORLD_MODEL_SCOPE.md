# Research scope: general time-series world models

## Objective
Discover and validate a general modeling contribution for time-series predictive states and conditional evolution. The research question must survive changing the evaluation dataset. It need not require a large pretrained foundation model, multimodality, actions, or a Loop Transformer unless evidence makes that mechanism necessary.

## Verified anchors
- WorldTS,arXiv2609.31162v1: https://arxiv.org/html/2609.31162v1 ; official code https://github.com/decisionintelligence/WorldTS . A two-stage latent predictive framework with covariate conditioning, evaluated across21datasets. It is a per-task forecasting framework; it does not demonstrate universal causal simulation or one shared pretrained model.
- Pythia,arXiv2610.05240v1: https://arxiv.org/html/2610.05240v1 . Shared multimodal predictive pretraining and a separate probabilistic readout. The primary paper lists code/model “To be released”; full native reproduction is not currently established.
- TSWM mechanism consistency,arXiv2610.01842v1: https://arxiv.org/html/2610.01842v1 ; official source https://github.com/HaochenZhang717/TSWM-Official-Repo . Forecast accuracy versus supplied directional action relations is already studied; this broad distinction is not a new research problem.
- OSSM,arXiv2609.33566: https://arxiv.org/abs/2609.33566 . Separating state propagation and observation correction already has a concrete recent proposal. Generic correction-versus-prediction separation cannot be renamed as a new mechanism.

## Evidence rules
One common method/protocol should address a capability across distinct systems. Datasets test this claim; success on one dataset does not define it. Include ordinary recurrent/state-estimation/probabilistic controls as appropriate, but the mere existence of a universal estimator is not proof that a specific finite-data residual has been solved. First identify the residual and realistic information/budget setting. Distinguish original-paper reproduction, a controlled transfer, and a new experiment. Do not impose causal identifiability or full stochastic simulation on a paper that only claims observational forecasting.

## Immediate research discriminator
Investigate whether a learned predictive state supports coherent conditional evolution and reuse when the observation or conditioning protocol changes, and whether any repeatable deficiency survives ordinary state-estimation/history/capacity controls. This is a search hypothesis, not an accepted novel gap or a committed architecture. The closest observer, meta-dynamics, probabilistic JEPA and foundation-model work must be checked before choosing a specific test. No method or hyperparameter is selected using test outcomes.

## Current disposition
Cycle11's12healthy neural fits reject its practical new-repair motivation. QG is optional infrastructure. No new model has been accepted. Concrete idea selection remains delegated by the human to this agent; no user choice is required for ordinary research decisions.
