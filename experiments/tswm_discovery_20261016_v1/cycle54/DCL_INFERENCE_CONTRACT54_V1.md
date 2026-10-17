# DCL: identification and forward-simulation input contracts

This is a source/paper intake, not an executed reproduction or a new model failure result. Source pin: [2cd46d13ae9e27feeefafe5745f26003929f9c6e](https://github.com/dynamical-inference/dcl/tree/2cd46d13ae9e27feeefafe5745f26003929f9c6e). Paper version inspected: [arXiv2410.14673v2, June2025](https://arxiv.org/html/2410.14673v2).

The paper studies latent system identification, adds a marginal-density correction to its theory, and explicitly examines unseen trials, ten-step dynamics metrics and delayed observations in Appendices F/G/H. It acknowledges limited nonlinear Lorenz dynamics recovery. Therefore, claims that it never tests heldout data, ignores long horizons, or requires only instantaneous injective observations would be inaccurate. These paper experiments were read, not reproduced here.

The released switching source uses a retrospective input contract. `SLDSContrastiveLearningSolver.predictions` in `dcl/solver/contrastive_solver.py:332-353` passes the next observed embedding as `x_next`. Its `dynamics_step:294-308` does so during learning as well. `MSESwitchingModel.logits` in `dcl/models/dynamics/switching_dynamics.py:258-268` uses that target to score every linear mode. `GumbelSLDS.forward` in `dcl/models/dynamics/slds.py:58-107` then weights/selects those modes. This can support identification of an already observed transition. It is not by itself a forward-simulation interface that has no future observations.

The metric helper `predict_with_dynamics` in `dcl/metrics/identifiability.py:189-235` also supplies next embeddings for switching systems. `DynamicsR2` is a correspondence-of-dynamics diagnostic, not automatically causal forecast accuracy. A good reported value should retain its original interpretation. This input difference is not evidence that the authors claimed a causal forecast result or committed data leakage in their stated identification task.

Competent existing controls are already visible. `MarkovSwitchingModel` and fixed/symmetric variants exist in `switching_dynamics.py:162-233`; `GumbelSLDS.to_gt_dynamics` explicitly replaces Markov switching with target-based selection for identification. Existing HMM/SLDS filtering, recurrent predictors and the C42 structured-HMM learning comparison also preclude presenting a post-hoc Markov gate as a new invention. Before any cross-task forecast comparison, specify exactly how future mode probabilities are learned and propagated from allowed history, and compare an ordinary causal gate with the same learned bank. Do not supply future embeddings or the true future mode sequence to the learned forecaster. No such new DCL learning or forecast comparison has run here.

Paper/runtime availability boundary: the Table1 sweep explicitly does not split the packaged data, while the paper separately reports generated heldout trials. The source-sweep behavior does not negate that paper experiment. Browser retrieval of the linked public data record failed; server metadata availability is being probed separately without downloading signals. A future adaptation must preserve source/data versions and disclose any newly defined split.

Recorded source Git blob IDs:

| Source | Git blob SHA1 |
|---|---|
| dcl/models/dynamics/slds.py | 025c5de80ae5911c96ea9cd383d3dbfac179e608 |
| dcl/models/dynamics/switching_dynamics.py | 28f7df82f663d877bc767120f9670dda1e28e65a |
| dcl/solver/contrastive_solver.py | 943acf0b118bc70b551d18c721689cb68dc62a7e |
| dcl/metrics/dynamics.py | 49efd0f768b437da9e865d0be83631e0fb7d4f77 |
| dcl/datasets/timeseries.py | ce526fd3234573b7f56b76c82c2719e3ee5ce2eb |

Decision: retain DCL as a relevant decoder-free dynamics-learning control. Do not confuse its switching identification interface with deployable simulation, and do not manufacture novelty by overlooking the available Markov models or the paper's own extended evaluations. The immediate C54 decision remains the ongoing matched native RP-GSSM EM-option comparison.
