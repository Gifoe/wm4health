# Completed remote experiments

## Protocol and scope

This is a **new controlled mechanism experiment**, not an original-paper Minecraft reproduction, not a real-data forecasting result and not a demonstrated new method. Unchanged native class/function bodies are loaded by AST; the model is deliberately reduced, and the physical-state encoder is fixed to identity. The learned encoder, pixel information bottleneck and native SIGReg gradients are outside scope. No existing source or earlier sealed cycle was modified.

Three 4D deterministic plants use x_next=.8*x+[0,.04,-.03,.02]+force(a)*e0. Force is zero, .35*a, or .35*sign(a)*relu(abs(a)-.5). Actions are randomized independently of state, with 30% exact zero and 70% uniform[-1,1], including during training. Train/validation/test counts are 8192/1024/2048. Cohort seeds are 15001/15002/15003 plus 100*plant_index. Three initialization seeds share these cohorts: they measure initialization variability, not independent dataset replication.

The native ARPredictor is reduced to width32, depth1, heads2, context1 and horizon3; 15,312 trainable parameters. All arms instantiate the same 267,777-parameter frozen random native inverse readout; only full uses its loss. Native-default AdamW1e-4, wd1e-3, clipping1 and 1,500 updates of batch128 are held fixed. Arms are prediction only; prediction+hinge(weight.5,margin.3); and prediction+hinge+frozen-IDM(weight1). Fixed-step final is primary; minimum factual-validation MSE checkpoints are secondary, with no test selection. Native prediction loss is independently matched to the evaluator before every fit. Native alternate-action rollout retains the first context action and zeroes later actions; its first-step hinge is therefore a constant, disclosed rather than silently corrected.

Cosine separation is evaluated in the fixed physical coordinates; no claim of representation-coordinate invariance is made. Null-plan sanity checks assert exact equality of physical trajectories for action*.4 in null/deadzone plants. Candidate regret compares the chosen plan with the true best among a fixed bank of 64 plans at 256 held-out initial states. This is offline candidate ranking, not closed-loop MPC; the physical oracle is never used for training or checkpoint selection.

## Fixed-step held-out results

| Dynamics | Loss arm | Factual MSE mean ± SD | Response MSE | Predicted / true cosine gap | Null response RMS | Candidate regret |
|---|---|---:|---:|---:|---:|---:|
| null | prediction | 0.000316252 ± 8.24e-05 | 9.53281e-06 | 2.86645e-05 / 6.76179e-09 | 0.000758866 | 0.000149361 |
| null | hinge | 0.0233748 ± 0.0018 | 0.0242111 | 0.141708 / 6.76179e-09 | 0.0546414 | 0.00538377 |
| null | full | 0.028421 ± 0.00294 | 0.0279313 | 0.175941 / 6.76179e-09 | 0.0663417 | 0.0053134 |
| linear | prediction | 0.000373943 ± 8.25e-05 | 6.97419e-05 | 0.0410212 / 0.0404066 | N/A | 0.000441625 |
| linear | hinge | 0.014483 ± 0.00213 | 0.00995197 | 0.160222 / 0.0404066 | N/A | 0.0188188 |
| linear | full | 0.0156628 ± 0.00268 | 0.0123094 | 0.165444 / 0.0404066 | N/A | 0.0192829 |
| deadzone | prediction | 0.000633648 ± 7.74e-05 | 0.000316646 | 0.00494845 / 0.00625266 | 0.0129826 | 0.0039685 |
| deadzone | hinge | 0.0217342 ± 0.00137 | 0.0247663 | 0.174625 / 0.00625266 | 0.0600896 | 0.0184421 |
| deadzone | full | 0.0242672 ± 0.00205 | 0.0258583 | 0.169255 / 0.00625266 | 0.064351 | 0.0217997 |


Mean ± sample SD across three initialization seeds; no independent-cohort confidence interval or population significance claim. Raw per-fit metrics include secondary validation-selected checkpoints in outputs/action_equivalence_pilot/results.json. Each fit retains best.pt, final.pt and six raw training-log records. Total training time 821.04 seconds; process orchestration and evaluation are additional.

## Finite objective witness

The exact known all-null plant is augmented by delta*a*e1. A fixed121-point grid [-.6,.6] selects delta by the native TRAIN objective on4096 fresh samples; validation2048/test4096 use fresh seeds61002/61003. Seed61001 generates training. This is a privileged diagnostic, not a learned baseline, theorem or population certificate. The same cohorts are reused across frozen readout seeds.

| Frozen readout seed | Arm | Train-selected spurious force | Exact predictor test objective | Wrong predictor test objective | Wrong predictor test factual MSE |
|---|---|---:|---:|---:|---:|
| 31 | hinge | 0.3 | 0.15 | 0.1480895 | 0.007952113 |
| 31 | full | -0.33 | 0.6058617 | 0.6022434 | 0.009622058 |
| 32 | hinge | 0.3 | 0.15 | 0.1480895 | 0.007952113 |
| 32 | full | 0.32 | 0.6101724 | 0.6070155 | 0.009047737 |
| 33 | hinge | 0.3 | 0.15 | 0.1480895 | 0.007952113 |
| 33 | full | 0.38 | 0.6043073 | 0.5974758 | 0.01275872 |


All six held-out witnesses lower the unchanged source-default objective by introducing physically wrong force. This rules out optimization failure and insufficient model capacity as the sole explanation for this objective-level mismatch. It does not show that all validation-tuned weights fail, or that a learned/probabilistic inverse model must fail.

## Non-executed objects

DoTime PFN repository metadata only: selected checkpoint295,027,515 bytes, revision bec6618c8b25f3a5d8cf608d4c597978d5a694ac, SHA256214db2fab16fb5fa17e68244abc0df3b7b0154b9fb2d2ce16b2be3f4d5d93618. Weights were not downloaded and no PFN accuracy result is claimed. Exact metadata is in outputs/dotime_pfn_metadata.json. SLIM, DWM, MiraBench, CAER and FLEX-WAM were read, not run.
