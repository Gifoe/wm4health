# Closest work and the novelty boundary

This is a targeted primary-source audit of the current candidate, not a new general survey. Author-reported performance is not presented as reproduced performance.

| Work | Decision-relevant evidence | Consequence for our candidate |
|---|---|---|
| [ILA, RSS 2022](https://invariance-through-latent-alignment.github.io/) | Unpaired target encoder adaptation uses latent distribution matching and dynamics consistency. Actual source and its configurable dynamics-update path were inspected. | Frozen-dynamics observation adaptation is established. |
| [Cross-Embodiment Transfer, 2024](https://arxiv.org/html/2406.01968) | The source agent is frozen; target observation maps are learned using cycle, adversarial and forward/inverse consistency losses. | Neither unpaired alignment nor combining these losses is a contribution. |
| [MoVie, NeurIPS 2023](https://arxiv.org/abs/2307.00972) | The paper studies new-view policy adaptation; the inspected evaluation entry point calls an existing self-supervised adaptation method. | New-view reuse already has a directly relevant public baseline. Only its entry/config contract was inspected here. |
| [CSR, ICLR 2025](https://arxiv.org/html/2407.20651v3) | Observation, reward and transition change factors are explicit, with self-adaptive model changes. The formal linear transition and identifiability assumptions restrict the theory. | Generic change-factor separation and selective model adaptation are already occupied. Our unconstrained conjugacy qualification does not refute its restricted theorem. |
| [AdaJEPA, June 2026](https://arxiv.org/html/2606.32026v1) | Adaptation occurs within MPC using newly executed transitions; encoder and predictor subsets are compared. The released trainer, buffer integration and defaults were inspected. | Updating the appropriate small component after a shift is not an untouched problem. |
| [Sandwich-Residuals, September 2026](https://arxiv.org/html/2609.21740v1) | The paper tests lightweight corrections around a frozen predictor, including compound shifts, and distinguishes useful compensation from true system identification. | A generic input/output residual repair is insufficient differentiation. Paper inspected; implementation was not pinned or executed in C84. |
| [Zweig and Bruna, UAI 2020](https://proceedings.mlr.press/v124/zweig20a.html) | Isomorphic-MDP transfer already treats observational correspondence and structural symmetry as central issues. | Coordinate ambiguity is established prior art, not our discovery. |

The remaining inference is narrower: component attribution requires a stated physical anchoring contract, and an action-prediction objective need not require that attribution at all. No claim is made that the above methods already solve every partially observed or nonlinear transfer problem. Conversely, their untested settings are not evidence of a new gap.
