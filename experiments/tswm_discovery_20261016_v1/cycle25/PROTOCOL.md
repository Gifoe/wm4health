# Native nonlinear multi-mechanism belief audit

Prospective. No new cycle25 model outcomes at writing. General question: does a learned time-series world model update only those physical future mechanisms for which new evidence is informative? This is not a proposed new architecture or benchmark performance tuning. Old Bayesian/particle methods must be competent controls.

## Source/interface gate

Unchanged Explore-the-Context corrected sourcef0baf6262523d9c5abaf8c0381b4db97818f7c34 and six published16D-latent `pendulum_quadrantactionfactorar2bd` checkpoints, constrained/unconstrained variants, original three seeds each. Original validation-selected final-budget checkpoints from cycle23, no new selection or training. Published variants may differ in standard-deviation rescaling as well as positive weights; do not call the comparison an isolated causal ablation.

Verify native action-repeat2, base dt.05, coarse dt.1, fully observed[cos(theta),sin(theta),angular_velocity], no observation/process noise, fixed parameters across resets. Unknown parameters are four quadrant action gains. Before data, audit the actual native prior sampler (including its RNG behavior), rather than assuming that nominal Box bounds equal its generated distribution. Do not fix the native prior and then mislabel an out-of-distribution failure as reproduction.

Prove and runtime-check: with known fixed reset state, u=0 is independent of every unknown gain. At query/probe centers theta=[3pi/4,pi/4,-pi/4,-3pi/4], angular velocity0, u=1.5, verify both substeps stay in the original quadrant with no velocity clipping over the full gain bounds. Then each one-step query depends only on its corresponding gain; a parameter-sensitive probe at quadrant1 cannot reveal other independent gains at empty context. Only make independence claims after auditing the actual prior. Do not assume every environment's u=0 is uninformative.

## Fixed data and model interfaces

64 new native context seeds starting20262501; independent initial-trajectory seeds starting21262501 and control RNGseed22262501.32-step native trajectories with random actionsUniform[-2,2], complete prefixes n=0,8,32 (supported max context50; add one<=33). Preserve gain/context IDs, true unwrapped state for audit only, observations and controls. Learners receive only the observed state/action prefix. All methods score the same outcomes. Reset probe/query state explicitly to known centers with the original environment still holding the SAME parameter context; this is a new controlled audit, not a reproduction of the paper's complete active-calibration protocol.

Probe quadrant1 initialtheta=pi/4, velocity0, u=0 or1.5. Query all four centers at u=1.5 for one native coarse step. u=1.5 is called parameter-sensitive, not necessarily conditionally informative after a long noiseless history that may already identify that parameter. At n=0 the other three gains are unchanged by this probe if prior independence holds. New physical systems, all six checkpoint seeds and variants share the same frozen cohort. Original training parameter-ID deduplication is not claimed.

## Evaluation and competitors

Primary distribution scores: angular-velocity marginal CRPS using Gaussian absolute-moment identity and independent latent-pair draws; joint energy score for the four scaled velocities(v/8); variogram score for the same joint vector; per-query mean MSE and marginal95% PIT coverage. These scores are well-defined for deterministic/manifold-supported physical truth. Do not use full3D Lebesgue-density NLL of redundant cos/sin observations as the scientific criterion. Keep the same latent draw across four queries from a system; conditional decoder noise is independent between reset trials.

Frozen competitors: no update; native context re-encoding; standard statistical linear regression/ensemble Kalman conditioning with full latent covariance; defensive prior/native-posterior mixture importance sampling, using its weighted particles and either diagonal or full covariance Gaussian moment projection. Particle budgets1024/4096 fixed, common particles for projection comparison; no budget chosen from scores. Three evaluation random seeds, common base random draws across comparable methods. Posterior covariance PSD stabilization is numerical and must be recorded. Retain ESS and Monte Carlo resolution diagnostics; sample degeneracy is not evidence that all existing inference methods fail.

Record prior and post-evidence cross-query means/covariances, especially unchanged mechanisms at n=0; physical scores can reveal that an internally coherent learned likelihood is nevertheless misspecified. Distinguish (1) amortized updater mismatch, (2) learned decoder's false coupling, (3) old inference approximation/particle degeneracy, and (4) numerical errors. A source/interface gate failure must be resolved before executing the scientific comparison. Write executable details before new model scores. If ordinary methods suffice, stop the new-architecture argument; do not force an idea.

Keep original data/source/weights, all failures and logs. No public publishing. All final claims remain provisional until independent verification and nearest-work review; final accepted directions currently0.
