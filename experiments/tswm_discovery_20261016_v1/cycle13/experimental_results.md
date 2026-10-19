# Verified cross-system ordinary controls

New controlled acquisition-policy experiment,not original-paper reproduction. Two real observed targets:ETTh1oil temperature and BikeSharing rental counts. L96,H24;chronological60/20/20;training-only standardization. BikeSharing has17379original rows on a17544hour grid:165naturally absent hours. Windows crossing any original gap were excluded. No future value was imputed into a context. The two datasets have145and123test roots respectively,with nonoverlapping24step target blocks.

There are24fits:2ordinary GRU families x2acquisition recipes x2datasets x3seeds,2000updates each. Same input information and active parameter counts[20376];seasonal skip adds no parameters. Both recipes know causal-MAR/sensor-MNAR type and the configured time-varying policy coefficient. They do not know hidden values or Bernoulli draws. Both families select checkpoints only on the same fixed source validation queries. Seed variation includes initialization/training sampling; physical data are shared.

Values are normalized MAE mean+/-sampleSD over3seeds,not confidence intervals. Negative paired difference favors ordinary augmentation. All conditions are retained; comparing error across different mask laws alone cannot establish regret because the observed information differs.

| Dataset | Family | Query | Source training | Augmented training | Paired difference |
|---|---|---|---:|---:|---:|
| ETTh1 | acquisition_controls | q0_constant | 0.176925 +/- 0.006096 | 0.174041 +/- 0.001924 | -0.002884 +/- 0.004233 |
| ETTh1 | acquisition_controls | q0_single_switch | 0.168613 +/- 0.004837 | 0.163755 +/- 0.000981 | -0.004859 +/- 0.005163 |
| ETTh1 | acquisition_controls | q0_fast_switch | 0.168204 +/- 0.005277 | 0.163844 +/- 0.000707 | -0.004360 +/- 0.004703 |
| ETTh1 | acquisition_controls | q1_constant | 0.169376 +/- 0.006025 | 0.166663 +/- 0.002713 | -0.002712 +/- 0.005526 |
| ETTh1 | acquisition_controls | q1_single_switch | 0.171779 +/- 0.006036 | 0.168166 +/- 0.003092 | -0.003612 +/- 0.005006 |
| ETTh1 | acquisition_controls | q1_fast_switch | 0.163244 +/- 0.005381 | 0.159986 +/- 0.002022 | -0.003258 +/- 0.003923 |
| ETTh1 | acquisition_controls | fully_observed | 0.154491 +/- 0.008162 | 0.150077 +/- 0.003859 | -0.004414 +/- 0.004340 |
| BikeSharing | acquisition_controls | q0_constant | 0.689665 +/- 0.011098 | 0.673090 +/- 0.008417 | -0.016575 +/- 0.014320 |
| BikeSharing | acquisition_controls | q0_single_switch | 0.683677 +/- 0.010040 | 0.669542 +/- 0.005511 | -0.014136 +/- 0.013635 |
| BikeSharing | acquisition_controls | q0_fast_switch | 0.667768 +/- 0.008570 | 0.652815 +/- 0.009746 | -0.014952 +/- 0.015022 |
| BikeSharing | acquisition_controls | q1_constant | 0.674073 +/- 0.014073 | 0.659767 +/- 0.006332 | -0.014305 +/- 0.014987 |
| BikeSharing | acquisition_controls | q1_single_switch | 0.675531 +/- 0.012873 | 0.662847 +/- 0.001971 | -0.012684 +/- 0.014826 |
| BikeSharing | acquisition_controls | q1_fast_switch | 0.670516 +/- 0.010818 | 0.656715 +/- 0.005097 | -0.013800 +/- 0.014279 |
| BikeSharing | acquisition_controls | fully_observed | 0.637167 +/- 0.013566 | 0.619486 +/- 0.020226 | -0.017681 +/- 0.019538 |
| ETTh1 | residual_controls | q0_constant | 0.195223 +/- 0.002151 | 0.193616 +/- 0.001363 | -0.001607 +/- 0.002203 |
| ETTh1 | residual_controls | q0_single_switch | 0.180930 +/- 0.002558 | 0.176622 +/- 0.000517 | -0.004309 +/- 0.002060 |
| ETTh1 | residual_controls | q0_fast_switch | 0.187113 +/- 0.005244 | 0.179160 +/- 0.001536 | -0.007953 +/- 0.005449 |
| ETTh1 | residual_controls | q1_constant | 0.185169 +/- 0.001758 | 0.182046 +/- 0.000759 | -0.003123 +/- 0.002255 |
| ETTh1 | residual_controls | q1_single_switch | 0.182377 +/- 0.002190 | 0.177729 +/- 0.000865 | -0.004648 +/- 0.002920 |
| ETTh1 | residual_controls | q1_fast_switch | 0.180503 +/- 0.003380 | 0.171532 +/- 0.002484 | -0.008971 +/- 0.002663 |
| ETTh1 | residual_controls | fully_observed | 0.170542 +/- 0.001161 | 0.170758 +/- 0.003099 | 0.000216 +/- 0.002248 |
| BikeSharing | residual_controls | q0_constant | 0.828468 +/- 0.016460 | 0.833160 +/- 0.028868 | 0.004692 +/- 0.044377 |
| BikeSharing | residual_controls | q0_single_switch | 0.882188 +/- 0.003777 | 0.888470 +/- 0.024362 | 0.006282 +/- 0.026807 |
| BikeSharing | residual_controls | q0_fast_switch | 0.920168 +/- 0.040659 | 0.898392 +/- 0.028094 | -0.021776 +/- 0.035531 |
| BikeSharing | residual_controls | q1_constant | 0.768480 +/- 0.014967 | 0.767445 +/- 0.009062 | -0.001035 +/- 0.019086 |
| BikeSharing | residual_controls | q1_single_switch | 0.800113 +/- 0.017048 | 0.803532 +/- 0.010304 | 0.003419 +/- 0.023349 |
| BikeSharing | residual_controls | q1_fast_switch | 0.830427 +/- 0.014879 | 0.824753 +/- 0.017061 | -0.005674 +/- 0.024083 |
| BikeSharing | residual_controls | fully_observed | 0.537901 +/- 0.011201 | 0.530781 +/- 0.008333 | -0.007119 +/- 0.002870 |

Full MSE,observation rates,reference persistence/past-day scores,predictions and curves are in the rawJSON/NPZ. The verifier also computes a descriptive training-only hour-of-week median reference; it is not a test-selected model. Training elapsed sum313.225s. This excludes setup and verification.

Independent verification checked40saved query groups,raw timestamps/normalization/labels,causal sampling laws,zeroed unobserved values,forward-fill/gaps,common queries,checkpoint selection and all24saved predictors/metrics. Original GPU replay error is exactly0. CPU versus original cuDNN predictions differs by at most0.0021522045;with cuDNN TF32 disabled,representative group checks across24models agree with CPU within2.3841858e-06. This backend precision difference is retained,not hidden by claiming exact CPU equality. No training was rerun by verification.
