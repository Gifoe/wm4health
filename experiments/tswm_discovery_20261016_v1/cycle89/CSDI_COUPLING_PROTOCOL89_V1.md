# Learned stochastic time-series qualification, frozen before fitting

The finite mechanism-ID critic stress does not establish a failure of a learned world model. This experiment first checks a native stochastic time-series generator on controlled nonlinear dynamics. It is a cross-task CSDI transfer, not a reproduction of its published tables and not a new method. The candidate concerns using forecasts for comparing hypothetical actions without changing the original per-action probabilities. OneWorld already addresses shared mechanisms; marginal-constrained coupling, Bayesian mixtures and optimal transport are established controls.

## Native source and interface

Use ermongroup/CSDI commit 7f24a436f08d98853a6b43d4f7f04e5a65ecdf27, unmodified main_model.py and diff_models.py. Invoke CSDI_base.get_side_info, calc_loss and impute with a fixed forecasting mask. The optional linear_attention_transformer import may be replaced by a sentinel module which raises if constructed; config is_linear=False selects the native PyTorch Transformer path. This is recorded explicitly, not a reproduction of the original dependency environment.

Four features are position/state x, velocity/second state z, future exogenous control u, and observed family indicator (-1 Duffing, +1 reservoir). The first eight state observations are conditioned; all controls and family indicators are conditioned. Future state values are targets only. Native impute generates all features internally; only its two target-state feature outputs are scored. No hidden mechanism IDs, simulator parameters, future target values or oracle trajectories enter the model. The model sees planned controls, not realized future outcomes.

## Controlled world distribution and splits

Reuse the unchanged cycle88 float64 RK4 discrete simulator, dt=.1 with four substeps. All trajectories start at [0,0,0], heating is zero. Parameters are k in [.4,.6,.8,1.,1.2,1.4,1.6,1.8,2.0], d=.35,g=1,n=.5,tau=3. Mechanism prior is [.4,.2,.1,.08,.07,.05,.04,.035,.025], fixed throughout a rollout. Both nonlinear families are trained by the same generator. Eight context positions have zero state and zero control; 64 future positions follow eight independently uniform[-1,1] piecewise-constant control segments of eight simulator steps each. At position t, the control feature is the action generating that position from its predecessor. There is no observation noise. Identical zero history intentionally leaves the persistent mechanism uncertain; this is a controlled feasibility setting, not evidence of broad real-world generalization.

Generate 4096 TRAIN trajectories per family with seed20268910, 128 CAL per family seed20268911. Normalize only state features using TRAIN mean/std, floor.05. EVAL seed20268912 generates eight independent query groups per family, four different action plans per group, exact nine-world trajectory banks and their known probabilities; it is not used for training or checkpoint selection. All payloads stay on the server data volume.

## Pilot fit and validation

One pilot optimizer seed20268913, exactly3000 Adam updates, batch64, lr1e-3, scheduler multipliers .1 at2250 and2700. Four native residual layers,64channels,eight heads,diffusion embedding128,time embedding128,feature embedding16,50 quadratic diffusion steps beta1e-4 to.5. Float32, TF32 disabled process-locally, threads2. Select among steps0/500/1000/2000/3000 by fixed-RNG CAL denoising MSE averaged over all50 diffusion times on the first64 CAL trajectories per family. No EVAL or coupling metric selects hyperparameters or checkpoints. This first seed qualifies feasibility; do not claim seed robustness.

Before fitting, execute forward/backward and native impute. Verify finite gradients/samples and that replacing all masked future targets with different dummy values does not change sampled targets when RNG is reset. Record native file hashes and runtime. If an implementation fault occurs, version the adapter/repair and record the failed attempt.

## Generator competence before coupling claims

After selection, draw128 independent native samples per EVAL action query by tiling conditions and invoking impute(n_samples=1) in batches64. Evaluate normalized trajectory energy score using exact nine-world reference weights; compare an ordinary deterministic exact posterior-mean forecast, independent exact mixture sampling with the same128-sample budget, and the exact population mixture. Estimate sampling uncertainty by query group, not by treating correlated trajectory positions as replicates. Also report nearest-oracle trajectory residual relative to between-world spread, mean error, terminal scalar CRPS, mechanism nearest-neighbor frequencies with an explicit warning that off-manifold samples cannot establish mechanism recovery.

If CSDI is worse than the deterministic posterior mean on energy score in either family or has large off-manifold residual, it is an unqualified generator for this coupling test. Do not label its behavior a new world-model failure or extend tuning based on EVAL. A later stronger generator requires a new prospective protocol. Only a qualified generator permits downstream consistency/calibration/planning claims.

## Ordinary marginal-preserving control

Separately replay all nine previously selected finite-ID critic checkpoints, verify their SHA256 and population-BCE selection, and compute log-domain iterative proportional fitting of their terminal Gibbs joint law to the fixed per-branch priors. Use all tuples K2/3/4, gamma1, tolerance1e-10, at most10000 full coordinate sweeps. Report both convergence and physics agreement after projection. This is an established baseline, not innovation. The finite oracle stress is solved by ordinary controls if they preserve probabilities without losing mechanism agreement; keep that negative evidence even if the learned-generator transfer fails.

Primary sources: [CSDI](https://arxiv.org/abs/2107.03502), [native code](https://github.com/ermongroup/CSDI/tree/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27), [OneWorld](https://arxiv.org/abs/2609.30946), [EDDY](https://arxiv.org/abs/2605.06553), [OSCC](https://arxiv.org/abs/2609.33543). No novelty is accepted by this protocol.
