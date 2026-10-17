# C60 prior-art and protocol boundary

The source neural-state components come from Forgione, Mejari and Piga, Learning neural state-space models: do we need a state estimator? https://arxiv.org/abs/2206.12928 and https://github.com/forgi86/sysid-neural-estimator . Pinned commit 7473be56be61b53f4e035db6c41cf4efbdc8cff3 and the MIT license remain intact. The current post-update output timing, derived input increment and cross-data recipe are explicit adaptations; the paper's complete training or benchmark table is not reproduced.

MetaLIC is a direct meta-state-space comparator, https://github.com/merlresearch/MetaLIC , but its released benchmark evaluation receives true output history in every test window and uses encoder/readout prediction. Its released loss does not train the computed x_plus term. That route must not be advertised as a free latent simulator. An adapted autoregressive rollout can be separately defined, but poor one-step-to-rollout migration would not itself refute the paper's original task.

MagNetX and PI-HNO already target history-conditioned material response; GRU/SSM state estimation and long-horizon training are established modeling mechanisms. C58 records direct stochastic, trajectory-coherence and memory prior art. None of those components can be renamed into a contribution here.
