# Frozen protocol: native controlled-latent EM positive control

Previous goal turn cycle26=PROGRESS: verified new counterevidence and source/runtime preparation. General reusable time-series world models remain the objective. Final accepted directions0. This protocol is frozen before any cycle27 fitting or held-out scoring.

## Purpose and source fidelity

Execute an existing model that already learns controlled hidden-state dynamics, rather than diagnose an untrained or inappropriate baseline as a Research Gap. Source samotto1/KoopmanGeneratorEM,commit ef830a6d4a6cff699492d54230fc2ad1729cab20,unmodified numeric module and released toy_model_dim_sweep.ipynb archived/hashed in cycle26. This is a native-algorithm/source-settings positive control plus a new independent evaluation extension;it is not the paper's complete dimension sweep or a reproduction of all its reported figures.

Native dynamics: x1dot=-x1+u;x2dot=5*(x1**3-x2);observed y=x2+N(0,.01variance). dt=.01;50 source-distribution training trajectories,501 observations,inputs held50 steps with N(0,5variance) amplitudes,independent standard-normal physical initial state. scipy.odeint on each constant-input segment with native default tolerances. Only first250 observations/249 transitions from each trajectory fit the model. Final251 observations never train or select models. Underlying physical state saved for verification/privileged diagnostics only.

Native augmented lift dimension5 including constant=4 stochastic latent coordinates;C[0,0]=1 as dimension-sweep notebook. Same reflected-eigenvalue generator initialization and .1 initial noise covariances,zero prior mean,identity prior covariance. Three declared initialization seeds20262711/12/13;single data seed20262701. No best-seed selection. Native module functions,constructor's implicit initialization and explicit-Euler EM updates preserved. Train-data RNG follows the notebook's per-trajectory sequence,with a declared seed. Scipy/NumPy/Numba versions recorded;task-local cycle26 vendor,CPUthreads1;no environment upgrades.

## Training budget and integration repairs

Use the native likelihood diagnostic and source .01 average-per-iteration convergence tolerance. Initial likelihood explicitly requested:the bare toy notebook otherwise reads undefined self.L. Import/omit unused save helper correctly;no change to optimization equations. Skip standalone1000x1000 eigenvalue plotting diagnostic.

Save initialization and iteration100;train at least100 EM updates. Thereafter continue each seed while the most recent ten-step likelihood increase/10>.01,up to1000 updates. Compute likelihood at iteration1,10,20,...,1000. A cap with continuing improvement is explicitly censored/underconverged,not evidence of a model defect. Retain full training trace and any failed seed;do not silently resample a successful fit. Final likelihood selection is training-only. No held-out predictions examined until all declared fits finish. This training-only extension is intentionally stronger than a bare100-step run and labeled as such. Checkpoint100 and final both evaluated,never choose using test data.

## Independent evaluation extension

Generate64 new trajectories each for control dwell50 (same distribution),5 (fast switching),100 (slow switching),seeds20262702/3/4 respectively. These cohorts are independent across regimes,not paired causal policy contrasts. They share physical/noise definitions with training,never initial states/noise realizations. They are fixed-parameter physical systems with many trajectories,not64different plants. Evaluation history lengths25,100,250;forecast251 observations under the actually scheduled future inputs. Models receive only prefix y,past u and planned future u;no future y or hidden physical state.

Two native prediction conventions retained:
1. Native smoothed-initial-state convention:infer latent posterior at time0 from available prefix,then free-run over complete input schedule and score only the future. It is allowed to condition on prefix observations;it is not a filtered posterior propagation through the already observed interval.
2. Standard filtered-prefix convention:start free forecast at posterior of last prefix sample. Same fitted model/weights,ordinary Kalman state estimation. Any gain here is an existing inference repair,not a new architecture.

Also score the native50 training trajectories' future continuation,clearly labeled within-trajectory continuation. Generalization claims use only independent64-trajectory cohorts. Persistence and linear ARX are ordinary same-observation controls. ARX uses ten lags of y/u,standardization and ridge lambda from[1e-4,.01,1,100],selected on held-out **training-prefix** times200–249 after fitting targets through199,then refit all250 training samples. No separate strong-model novelty claim depends on beating ARX. Known-physics exact5D exponential rollout from true current physical state is a privileged information/control-numerics diagnostic,not a fair observation-only comparator.

All scored settings retained. Metrics:global noisy-observation MSE,clean-state-output MSE,relative L2 to clean output,and per-trajectory errors. Native Gaussian outputs:pointwise marginal NLL and95%coverage,not joint trajectory calibration. No test-driven stopping/hyperparameters. Aggregation across initialization seeds does not create independent physical cohorts. Bootstrap trajectories only after averaging model-seed losses within each cohort;report seed spread separately.

## Verification and conclusions

Verify source Git blobs/SHA256,exact5D matrix-exponential physical rollout versus odeint,training-prefix boundary,no future observation use,and model invariants. Independently replay native fitted parameters with NumPy Kalman filter/smoother and dynamics;recompute scores from saved predictions. Save numeric weights,latent moments,input schedules,raw traces,configs,source/dependency hashes and runtime. No pickled model from external sources.

If this mature baseline works,reduce priority of broad claims that partial observation plus input-dependent dynamics necessitates a new framework. If it fails,first test training convergence,numerics,initialization,capacity/model class and state inference. Neither its failure nor success alone establishes the proposed cross-system pretrained-transfer problem. Native slow-manifold dynamics do not directly test actuator-gated observability;that candidate remains untested here. Do not complete the overall research goal on this positive control.
