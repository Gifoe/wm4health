# 竞争解释和反证

| Explanation | Evidence / falsifier | Current status |
|---|---|---|
| Driver absent/unidentifiable | Invertible observation construction,randomized support,fullobs positive controls | Rejected for these tasks |
| Entire representation collapsed | Effective rank,current q,p decoding,inverse accuracy | Rejected for joint_idm |
| Weak linear probe | Nonlinear MLP and post-pilot ExtraTrees; frozen successor probes | Strongly weakened within tested readout families |
| Only a bad transition head | Refit successor predictor on frozen encoding | Not sufficient to explain primary joint_idm result |
| Intrinsic4D information bottleneck | True causal state has three factors; a suitable4D map exists | Dimension alone is not an information-theoretic impossibility; learned selection still matters |
| More capacity solves it |12D control and full raw observation | Supported; prevents a universal new-architecture claim |
| Any stochastic predictor fixes it | SimplifiedGaussian/mix versus proper normalized EMA/router/SSM | Simplified controls insufficient; source-informed control pending |
| Grounded target selection is the missing ingredient | Same teacher:full bank,contrast bank,projected bank,random subspace,extra-budget reconstruction | cycle21 fixed independent test underway |
| A new neural compressor is necessary | Classical active-subspace+action-output target succeeds in9/9 cases | Rejected on these toys; target-selection novelty remains unresolved |

Future source-native/realistic checks must distinguish a new usable state-selection mechanism from an old sensitivity method, extra teacher capacity, auxiliary-loss weighting or an artificially favorable observation map. Do not choose a new setting merely to hide an already-successful simple baseline.
