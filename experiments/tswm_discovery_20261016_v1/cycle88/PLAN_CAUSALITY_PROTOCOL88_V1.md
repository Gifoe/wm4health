# C88 decisive developmental falsification: temporal action abstraction

Previous goal turn classification: NO_PROGRESS (status clarification only). Goal remains ACTIVE and accepted novel directions remain zero.

## Decision, before results

This is one controlled cross-task experiment, not an original-paper reproduction or a new method proposal. The generic state/mechanism adaptation question in RESEARCH_BRIEF88_V1.md is not advanced by this test. Its narrower, source-led question is whether whole-next-window action aggregation in pinned native MTS3 causes materially anticipatory forecasts after learning, and whether a one-line ordinary causal reindexing suffices. Do not call that reindexing a novel architecture.

Native source: ALRhub/MTS3 commit ab971c0ec43df7664c4b727ca69d5b8dd0cab5da. Preserve all pinned files. Stock MTS3 and lagged-index MTS3 have exactly the same parameters, initialization, training batches, optimizer, model dimensions, loss and selection rule. The latter changes only the manager action slice used to advance the next slow-state prior from the next window to the just-completed window; retain the original last-window skip. Native acRKN is a causal existing comparator with its official latent dimension30 and widths. Its parameter count is reported, not assumed matched. No state/physical parameter/family ID is an inference feature.

## Data and information

Two controlled nonlinear fast/slow families: a heated Duffing oscillator and a bounded coupled reservoir. State dimension3; the slow thermal state is hidden. Sixteen nonlinear sensor channels observe only the two fast coordinates through the same fixed mapping. A scalar mechanical control and a scalar heating control are both observed. Both families have varying physical parameters, and future controls are randomized independently of parameters/history: supplied plans are exogenous interventions, not behavior-policy observations.

Define the simulator by its explicit float64 four-substep RK4 map with macro dt=.1. It is a discrete simulation benchmark; no certified continuous-time solver accuracy claim is made. TRAIN has32 independent parameter systems/family, eight independent trajectories/system; developmental VAL has eight new systems/family, four trajectories/system. Each trajectory has192 transitions. Sensor noise=.01 before TRAIN-only per-channel standardization. All initial conditions and parameter draws are retained without trajectory-based rejection. Nonfinite simulation is an error, not silently resampled. The full hidden metadata stays remote and is used only by simulator/sanity analysis.

Random TRAIN windows have96 transitions; model H=32. Observed prefix length65 means o[0:65] is available, and output positions64:96 predict o[65:97], a32-step free rollout. The first query occurs inside a macro window to avoid a special boundary-only test. Model observations after index64 are zeroed as well as explicitly marked invalid. Native next-observation output alignment is checked. No future observation or normalization statistic is available at inference.

## Pilot budget and selection

One developmental optimizer seed2026884 for each of stock/lagged/acRKN; identical NumPy minibatch/crop stream. Adam lr=.001, batch16,1000 updates, gradient clipping1, float32 highest precision and TF32 disabled. Forecast-only Gaussian NLL is the training loss. Validate factual forecast MSE at0/250/500/750/1000; select only by lowest unedited VAL MSE. No prefix or edited-plan metric participates in training/checkpoint selection. Save final optimizer/RNG states and selected/final weights. Equal updates are explicit; unequal elapsed cost is reported.

Qualification is developmental, not a universal convergence guarantee. A model is source-competent only if its selected factual MSE is below0.9 times observation persistence in **each** family. If no model passes, the pilot cannot establish a scientific failure and no method claim is promoted. Training insufficiency remains a competing explanation.

## Decisive measurements after selection

Use VAL windows starting at48, with the fixed65-observation prefix. For each source-competent model and each suffix onset80 or88, set only mechanical controls from that onset onward to +.8 or -.8, within the source action support; retain prefix/history/heating. Re-simulate each edited suffix from the exact same state with the same observation-noise realization. True sensor targets before the edit first takes effect must be identical, exactly up to stored floating arithmetic. Report model early-prefix RMS/max change, paired future-response error, factual/edited MSE, persistence ratios and family breakdown. Positive response accuracy is required: zero response everywhere is not faithful simulation. Test a changed masked-observation tensor too, to rule out future-observation leakage.

Numerical nonanticipation threshold: maximum standardized-sensor prediction difference<=1e-6; material pre-action sensitivity descriptive threshold RMS>=.01 TRAIN sensor standard deviations. A sensitivity below that level can still be a functional violation but does not by itself justify a research direction. Exact float32 same-batch replay is required before causal comparisons.

## Research consequence

If the ordinary lagged-index control is competent, nonanticipatory, and its per-family factual MSE is <=1.05 times stock MTS3, reject a new-method claim based on this defect. If acRKN also suffices, that strengthens rejection. If the ordinary correction loses substantial forecast/response accuracy, investigate source competence, optimization and capacity before interpreting any tradeoff. One simulator pool/one seed is discovery evidence only; positive research claims additionally require independent held-out pools, seeds, stronger closest-work comparison, and a real public action-conditioned benchmark. Do not open a held-out EVAL to tune this pilot. Do not add architecture complexity merely to salvage a weak candidate.
