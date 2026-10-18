# Common-history simulation: frozen development protocol

This protocol is authored before any free-forecast accuracy result. It is baseline adequacy and failure attribution, not a new method, unseen-composition benchmark, or paper reproduction. Scope: the existing source-based Modular NRI subset experiment. Later framework claims still need independent mechanisms, observation interfaces and real tasks.

## Information and execution gate

Preserve the running training protocol and its native reset-block VAL checkpoint selection. Begin neural development forecasting only after all three 2000-epoch fits and their campaign have terminal successful receipts. Strict-load each fit's selected immutable checkpoint and verify all parameters/source/data hashes. No retraining, checkpoint reselection, or future-dependent graph inference. Reserved EVAL rows 256:512 remain closed.

Use all 256 development trajectories (source test rows 0:256). Every arm sees exactly observations 0:49 and produces times 49:99, without intermediate true-state resets. Native arms additionally benefit from their shared module/proposal training on 2048 source trajectories; the classical controls have that TRAIN normalization/calibration but fit transition coefficients from each presented history. This is a clearly stated information and model-class difference, not a claim of matched parameter count or compute.

## Complete model arms and attribution

1. Native library with the trained state encoder's argmax edge assignments from the 49-step history.
2. The same frozen model with 100 native simulated-annealing proposal/acceptance steps, using only the history. Use the already qualified context_inference and rollout functions unchanged.
3. The same model with 1000 such steps. Restart from the same per-fit/per-batch seed as the 100-step arm. This is an explicit inference-compute control, not a new learned model. Batch size 32, fixed row order. No weight update. Record inference times and history-fit losses.
4. Affine residual VAR(1) fitted to that trajectory's observed history.
5. Affine residual VAR(2), likewise fitted from history.
6. Persistence of the last observed state.

All forecasts use the source's stored 20 state coordinates without physical-unit reinterpretation. Neither graph labels nor future observations enter these six arms. For neural arms, report graph assignment accuracy as a diagnostic only, including the ambiguity of a global permutation of the two learned edge modules. Prediction accuracy is the primary measurement.

A separate privileged diagnostic uses the recorded true graph assignments with the learned module library. Fix the binary module-to-label orientation by minimizing native ten-step-reset reconstruction error on TRAIN rows 0:128, considering only the two global permutations. Then apply that one orientation to VAL graph labels. Report this arm separately: it is an attribution oracle, not an ordinary forecast baseline or a repair method. It helps distinguish graph-inference errors from library/rollout errors.

## Classical control selection, before VAL scoring

Compute coordinate mean and standard deviation from all 2048 TRAIN histories, using float64 and a 1e-8 scale floor. At each lag p=1,2, regress normalized state increments on the p most recent states plus an intercept. Minimize mean squared one-step regression error plus alpha times the squared non-intercept coefficient norm. The intercept is unpenalized.

Select alpha separately per lag from [1e-8,1e-6,1e-4,1e-2,1,100]. For every TRAIN trajectory fit on its first 24 observations and freely predict its remaining 25; choose the lowest aggregate MSE in the original stored coordinates (first grid value on ties). Store every grid score, per-trajectory errors, normalization and hashes. Do not use VAL/EVAL for this selection. At VAL inference refit the selected lag/alpha on all 49 observed steps; use float64, no clipping, stabilization or oracle state constraints.

Qualification of the classical implementation uses one known stable affine system and an independent augmented least-squares solution. This verifies lag ordering, intercept, normalization and recursive prediction. It is an engineering qualification, not research evidence. Do not repeat smoke tests after a passing critical-path check unless code changes.

## Measurements and interpretation

Save per-trajectory predictions and graph assignments on server G storage. At horizons 1,10,25,50 report prefix MSE/MAE, endpoint MSE and per-coordinate errors. Show each fit seed and mean/range across the three fits. If reporting a trajectory bootstrap, label it conditional variation over these fixed source trajectories, not uncertainty over independent mechanisms. No significance claim from three training seeds alone.

Count every nonfinite or divergent trajectory; never silently discard it, clip it, or average only surviving predictions. Keep failed arms explicit. Report model size, source revision, checkpoint selection, graph-search budget and actual runtime. Native reset-block VAL remains a separate metric and must not be plotted as common-history simulation error.

If longer graph search or the privileged assignments repair a deficit, attribute it to inference, not a new representation/dynamics defect. If the ordinary history-based controls suffice, do not claim a new mechanism is necessary. If all competent baselines fail, first examine library training, simulator implementation, data coverage and observability. A residual on this spring subset is not yet a general time-series world-model gap.

The next research decision is whether a specific remaining capability warrants an independent structural-composition experiment. No new framework is accepted by this protocol.
