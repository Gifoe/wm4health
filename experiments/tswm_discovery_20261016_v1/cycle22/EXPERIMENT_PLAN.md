# Real conditional-state comparator: Causal Chambers wind tunnel

This fixed new-task experiment follows the cycle20/21 mechanism studies. It is a migration of native WorldTS components to an independent public real time-series task, not a reproduction of its original leaderboard, not causal-effect identification, and not a new proposed model. It asks whether existing role-aware conditional models already provide a competent solution before claiming a new state/target-discovery framework.

## Qualified interface and splits

Use the preserved wt_walks_v1 archive already publisher-MD5/SHA256 checked in cycle8, source0c8a77ea14bd4437b1fdd79adc1aa91ac2342569. New inventory rechecks26CSV hashes against the earlier audit:16 random-walk recordings and10 wave recordings,251016rows,no nonfinite modeled values or nonpositive clock increments. Run1 is shorter(1016rows); retain it without padding or inventing data. Metadata inspection preceded this plan; no new-model test metrics have been viewed.

Historical target channels: current_in,current_out,rpm_in,rpm_out,pressure_upwind,pressure_downwind,pressure_ambient,pressure_intake,mic,signal_1,signal_2. Known scheduled conditions: load_in,load_out,hatch,pot_1,pot_2. The generator writes each actuator command sequence in advance and measures after setting all commands. At origin o the history is rows[o-64:o], future conditions rows[o:o+16], and sensor targets rows[o:o+16]. No future sensor measurement,time interval,flag,counter or sensor setting is an input. Raw timestamps stay float64 for auditing. This is sample-index conditional sensor forecasting under prescribed plans, not a physical continuous-time simulator or identified counterfactual response.

Train random_walk1–10,val11–12,test13–16,whole-recording separation. Window stride16,context64,horizon16,no cross-recording windows. Fit per-column mean/std on all TRAIN raw rows only,ddof0,stdfloor1e-6,apply unchanged to other splits. Model targets float32. Native window RevIN is an additional explicit architectural choice, not an alteration of the global train-only scaler.

Wave recordings are secondary held-out protocol shifts,never used for hyperparameters/selection. Eight match the random-walk sensor settings. fast_run1 and slow_run2 change osr_mic1→8: retain and report as a separate acquisition-shift group,not pure dynamical OOD. The different sampling intervals are described but future realized dt is never provided. No claim of new measurement failure is permitted.

## Models and training fixed before execution

Native WorldTS source at269426ea6217d7152a244da0e209d922acd46d4f remains unchanged;18Python file hashes checked. Use its numerical causal-patch encoder/decoder and PatchTST predictor. Adapt only data/harness/config:state/decoder patch3,condition_dim32(or4 for corresponding arm),predictor patch8/stride4,d_model64,n_heads4,e_layers1,d_ff128,dropout.05,history64,horizon16,11targets+5conditions,no time/modal branch. Native stage1 LatentStateLoss:alignmentMAE weight10,cosine15,legacy variance/covariance regularizer.1,stdweight1,covweight.04,detach_targettrue. Stage2 decoder mix is predicted-target observation MSE+.3 encoded-future-target reconstruction MSE,with encoder/predictor frozen and in eval mode. Model selection uses validation latent loss(stage1) and validation forecast MSE(stage2).

Six neural arms,three initialization seeds2021/2022/2023,18fits:
1.worldts32:above two-stage,latent32,RevINon.
2.worldts4:two-stage,latent4,RevINon.
3.worldts4_fixed:two-stage,latent4,RevINoff(global scaler remains).
4.worldts32_endtoend:same native architecture,all modules jointly trained on forecast MSE;an objective control,not official WorldTS training.
5.gru64:ordinary conditional state-space encoderGRU64 over history16vars,decoderGRU64 driven by future5commands,linear11sensor residual readout added to last observed target,forecast MSE.
6.mlp128:flattened64x16history+16x5plan,two128-wideGELU layers,176residual outputs added to last observed target,forecast MSE.

Adam,lr.001,batch128,gradient clip1,epoch shuffle seeded independently per stage,CUDAfloat32/noAMP/TF32off,deterministic algorithms. Stage1 max50epochs,patience10;stage2 and other supervised arms max100,patience15. ReduceLROnPlateau factor.5,patience5,minlr1e-5 on the corresponding validation criterion. Save all curves and best checkpoints;no test tuning or hidden training extension. These are explicit new-task choices rather than copied per-dataset optimal paper settings. Counts/seconds/actual updates are reported;this is not an exact FLOP-matched method leaderboard.

Simple controls:persistence,ordinary direct ridgeARX(past8lags of16vars+all16future5commands;7ridge values1e-6,1e-4,1e-2,.1,1,10,100 selected only on validation). MLP/GRU access same raw signal values as WorldTS;ARX intentionally uses less historical context as a simple sufficiency check.

No claim that a4D-per-token WorldTS is a strict4-number state bottleneck:it retains64tokens,condition tokens,and RevIN's per-window mean/std sidestream. Record this information interface explicitly when comparing with the earlier current-state toy.

## Test and falsification

All18 fits and ridge selection finish before test evaluation. Primary metrics:equal-channel train-standardized MSE/MAE over all11targets,each horizon,each recording,and target groups(current/RPM/pressure/mic/signals). Report raw-unit MAE per channel,forecast improvement versus persistence,and separate matched-settings versus changed-acquisition wave groups. Record actual update/parameter/time/memory budgets. No paired intervention truth is manufactured from the logs.

Native diagnostics:at the fixed final model,decode predicted latent and privileged encoded future-target latent using the same decoder;save oracle-vs-predicted errors without labeling the privileged result a forecast. Native role separation plus healthy ordinary baselines would weaken or eliminate a general architecture claim from the earlier mixed-observation toy. Failure must be localized among encoder/readout,transition/objective,training convergence and acquisition/task shift before calling it a research gap.

This first real task cannot by itself establish broad generalization or a top-conference contribution. If ordinary models suffice,keep that negative evidence and adjust research scope instead of adding dataset-specific tricks. Preserve every failed attempt,source/config/data hashes,logs and predictions.
