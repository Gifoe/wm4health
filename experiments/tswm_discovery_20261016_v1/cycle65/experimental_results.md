# Frozen-model law results

All models and queries follow PROTOCOL65_V1.md and EVALUATION_SPEC65_V1.md. The primary distance is the exact empirical one-dimensional W1 averaged over128 fixed random unit directions plus8 coordinate axes, divided by the reference's mean projected standard deviation. It is a finite sliced distance, not full multivariate W1/W2. Each generated law has512 samples; each exact do/observational reference has8192 samples. Rejection sampling has a valid unit envelope and its acceptance/proposal counts are retained. No threshold was chosen from these results.

Each learned-method row averages3 independent training seeds x3 MC replications. The nine law values are not nine independent model fits. Per-training-seed means appear below. An iid reference row has only3 MC replications. Scores against do and obs are two target views of the SAME generated law. Native block queries should first be judged against obs, prefix/causal rollouts against do; both columns expose the semantic difference.

| mechanism | plan | method | laws | do distance | obs distance | mean generation seconds |
|---|---|---|---|---|---|---|
| linear | zero | CSDI_native_coverage_block | 9 | 0.337953 | 0.198135 | 0.10961 |
| linear | zero | CSDI_native_coverage_prefix | 9 | 0.222718 | 0.346598 | 0.89204 |
| linear | zero | CSDI_native_random_block | 9 | 0.352006 | 0.129784 | 0.13693 |
| linear | zero | CSDI_native_random_prefix | 9 | 0.272244 | 0.255712 | 0.91789 |
| linear | zero | causal_MDN_causal_rollout | 9 | 0.0820483 | 0.527618 | 0.013429 |
| linear | zero | causal_linear_AR | 9 | 0.0603888 | 0.510921 | 0.00017777 |
| linear | zero | privileged_iid_do | 3 | 0.0529436 | 0.51741 | reference |
| linear | zero | privileged_iid_obs | 3 | 0.327624 | 0.0684768 | reference |
| linear | sine | CSDI_native_coverage_block | 9 | 0.541846 | 0.166991 | 0.10384 |
| linear | sine | CSDI_native_coverage_prefix | 9 | 0.338007 | 0.536856 | 0.87563 |
| linear | sine | CSDI_native_random_block | 9 | 0.555911 | 0.233691 | 0.11724 |
| linear | sine | CSDI_native_random_prefix | 9 | 0.420188 | 0.629946 | 0.90197 |
| linear | sine | causal_MDN_causal_rollout | 9 | 0.0852099 | 0.853337 | 0.0034271 |
| linear | sine | causal_linear_AR | 9 | 0.05793 | 0.837363 | 0.0001587 |
| linear | sine | privileged_iid_do | 3 | 0.0642741 | 0.835445 | reference |
| linear | sine | privileged_iid_obs | 3 | 0.51855 | 0.0537854 | reference |
| nonlinear | zero | CSDI_native_coverage_block | 9 | 0.559854 | 0.214802 | 0.11311 |
| nonlinear | zero | CSDI_native_coverage_prefix | 9 | 0.353374 | 0.608418 | 0.86535 |
| nonlinear | zero | CSDI_native_random_block | 9 | 0.59267 | 0.22391 | 0.12164 |
| nonlinear | zero | CSDI_native_random_prefix | 9 | 0.474681 | 0.508436 | 0.8703 |
| nonlinear | zero | causal_MDN_causal_rollout | 9 | 0.0792185 | 1.06954 | 0.013943 |
| nonlinear | zero | causal_linear_AR | 9 | 0.249427 | 1.19083 | 0.00014406 |
| nonlinear | zero | privileged_iid_do | 3 | 0.0510693 | 1.01396 | reference |
| nonlinear | zero | privileged_iid_obs | 3 | 0.583707 | 0.0544325 | reference |
| nonlinear | sine | CSDI_native_coverage_block | 9 | 0.757593 | 0.239727 | 0.099962 |
| nonlinear | sine | CSDI_native_coverage_prefix | 9 | 0.542755 | 0.593106 | 0.87476 |
| nonlinear | sine | CSDI_native_random_block | 9 | 0.874287 | 0.422866 | 0.10394 |
| nonlinear | sine | CSDI_native_random_prefix | 9 | 0.764992 | 0.633563 | 0.88239 |
| nonlinear | sine | causal_MDN_causal_rollout | 9 | 0.0781615 | 1.1008 | 0.0036649 |
| nonlinear | sine | causal_linear_AR | 9 | 0.545584 | 1.58728 | 0.00016203 |
| nonlinear | sine | privileged_iid_do | 3 | 0.0471201 | 1.07047 | reference |
| nonlinear | sine | privileged_iid_obs | 3 | 0.695539 | 0.0602849 | reference |

The decisive existing-method comparison is below. Coverage-prefix is still2.71-to6.94 times the MDN do-distance across the four aggregate settings. MDN errors are0.078-to0.085 versus exact iid distances0.047-to0.064. This does not establish exact calibration of MDN; it rules out a claim that this study requires a new architecture to approach the target law. Mask coverage improves each aggregate native-prefix result, but not every individual training seed, and native conditional approximation remains a viable explanation. These two systems do not establish a general impossibility of diffusion reuse.

| mechanism | plan | coverage-prefix do | ordinary MDN do | exact iid do | prefix/MDN ratio |
|---|---|---|---|---|---|
| linear | zero | 0.2227177 | 0.08204829 | 0.05294362 | 2.7145 |
| linear | sine | 0.338007 | 0.0852099 | 0.0642741 | 3.9668 |
| nonlinear | zero | 0.3533743 | 0.07921848 | 0.05106927 | 4.4608 |
| nonlinear | sine | 0.5427546 | 0.07816151 | 0.0471201 | 6.944 |

Generation timings exclude saving arrays and metric computation. Each native block batch uses50 denoiser calls; each eight-step prefix batch uses400. The same particle-batch size512 applies. Causal MDN uses8 small transition calls. These are observed operational costs, not an equal-compute new-method comparison.

Per-training-seed means (each averages3 MC replications):

| mechanism | plan | method | target | training seed | mean distance |
|---|---|---|---|---|---|
| linear | zero | CSDI_native_coverage_block | do | 2021 | 0.350016 |
| linear | zero | CSDI_native_coverage_block | do | 2022 | 0.322054 |
| linear | zero | CSDI_native_coverage_block | do | 2023 | 0.34179 |
| linear | zero | CSDI_native_coverage_block | obs | 2021 | 0.116452 |
| linear | zero | CSDI_native_coverage_block | obs | 2022 | 0.363566 |
| linear | zero | CSDI_native_coverage_block | obs | 2023 | 0.114386 |
| linear | zero | CSDI_native_coverage_prefix | do | 2021 | 0.219676 |
| linear | zero | CSDI_native_coverage_prefix | do | 2022 | 0.254165 |
| linear | zero | CSDI_native_coverage_prefix | do | 2023 | 0.194312 |
| linear | zero | CSDI_native_coverage_prefix | obs | 2021 | 0.242762 |
| linear | zero | CSDI_native_coverage_prefix | obs | 2022 | 0.499317 |
| linear | zero | CSDI_native_coverage_prefix | obs | 2023 | 0.297713 |
| linear | zero | CSDI_native_random_block | do | 2021 | 0.349571 |
| linear | zero | CSDI_native_random_block | do | 2022 | 0.368501 |
| linear | zero | CSDI_native_random_block | do | 2023 | 0.337946 |
| linear | zero | CSDI_native_random_block | obs | 2021 | 0.165486 |
| linear | zero | CSDI_native_random_block | obs | 2022 | 0.0994327 |
| linear | zero | CSDI_native_random_block | obs | 2023 | 0.124434 |
| linear | zero | CSDI_native_random_prefix | do | 2021 | 0.303382 |
| linear | zero | CSDI_native_random_prefix | do | 2022 | 0.265759 |
| linear | zero | CSDI_native_random_prefix | do | 2023 | 0.247591 |
| linear | zero | CSDI_native_random_prefix | obs | 2021 | 0.295693 |
| linear | zero | CSDI_native_random_prefix | obs | 2022 | 0.176386 |
| linear | zero | CSDI_native_random_prefix | obs | 2023 | 0.295055 |
| linear | zero | causal_MDN_causal_rollout | do | 2021 | 0.0844103 |
| linear | zero | causal_MDN_causal_rollout | do | 2022 | 0.0998261 |
| linear | zero | causal_MDN_causal_rollout | do | 2023 | 0.0619085 |
| linear | zero | causal_MDN_causal_rollout | obs | 2021 | 0.538231 |
| linear | zero | causal_MDN_causal_rollout | obs | 2022 | 0.553098 |
| linear | zero | causal_MDN_causal_rollout | obs | 2023 | 0.491524 |
| linear | zero | causal_linear_AR | do | 2021 | 0.0508788 |
| linear | zero | causal_linear_AR | do | 2022 | 0.0581435 |
| linear | zero | causal_linear_AR | do | 2023 | 0.0721442 |
| linear | zero | causal_linear_AR | obs | 2021 | 0.526071 |
| linear | zero | causal_linear_AR | obs | 2022 | 0.506626 |
| linear | zero | causal_linear_AR | obs | 2023 | 0.500067 |
| linear | sine | CSDI_native_coverage_block | do | 2021 | 0.550504 |
| linear | sine | CSDI_native_coverage_block | do | 2022 | 0.499837 |
| linear | sine | CSDI_native_coverage_block | do | 2023 | 0.575196 |
| linear | sine | CSDI_native_coverage_block | obs | 2021 | 0.0959323 |
| linear | sine | CSDI_native_coverage_block | obs | 2022 | 0.279941 |
| linear | sine | CSDI_native_coverage_block | obs | 2023 | 0.125101 |
| linear | sine | CSDI_native_coverage_prefix | do | 2021 | 0.366369 |
| linear | sine | CSDI_native_coverage_prefix | do | 2022 | 0.33338 |
| linear | sine | CSDI_native_coverage_prefix | do | 2023 | 0.314272 |
| linear | sine | CSDI_native_coverage_prefix | obs | 2021 | 0.495224 |
| linear | sine | CSDI_native_coverage_prefix | obs | 2022 | 0.589364 |
| linear | sine | CSDI_native_coverage_prefix | obs | 2023 | 0.525981 |
| linear | sine | CSDI_native_random_block | do | 2021 | 0.558326 |
| linear | sine | CSDI_native_random_block | do | 2022 | 0.53873 |
| linear | sine | CSDI_native_random_block | do | 2023 | 0.570676 |
| linear | sine | CSDI_native_random_block | obs | 2021 | 0.292245 |
| linear | sine | CSDI_native_random_block | obs | 2022 | 0.178735 |
| linear | sine | CSDI_native_random_block | obs | 2023 | 0.230095 |
| linear | sine | CSDI_native_random_prefix | do | 2021 | 0.464231 |
| linear | sine | CSDI_native_random_prefix | do | 2022 | 0.392074 |
| linear | sine | CSDI_native_random_prefix | do | 2023 | 0.404258 |
| linear | sine | CSDI_native_random_prefix | obs | 2021 | 0.661175 |
| linear | sine | CSDI_native_random_prefix | obs | 2022 | 0.557829 |
| linear | sine | CSDI_native_random_prefix | obs | 2023 | 0.670832 |
| linear | sine | causal_MDN_causal_rollout | do | 2021 | 0.106705 |
| linear | sine | causal_MDN_causal_rollout | do | 2022 | 0.0841591 |
| linear | sine | causal_MDN_causal_rollout | do | 2023 | 0.064766 |
| linear | sine | causal_MDN_causal_rollout | obs | 2021 | 0.847683 |
| linear | sine | causal_MDN_causal_rollout | obs | 2022 | 0.868806 |
| linear | sine | causal_MDN_causal_rollout | obs | 2023 | 0.843522 |
| linear | sine | causal_linear_AR | do | 2021 | 0.0519488 |
| linear | sine | causal_linear_AR | do | 2022 | 0.055257 |
| linear | sine | causal_linear_AR | do | 2023 | 0.0665841 |
| linear | sine | causal_linear_AR | obs | 2021 | 0.848647 |
| linear | sine | causal_linear_AR | obs | 2022 | 0.831987 |
| linear | sine | causal_linear_AR | obs | 2023 | 0.831454 |
| nonlinear | zero | CSDI_native_coverage_block | do | 2021 | 0.651639 |
| nonlinear | zero | CSDI_native_coverage_block | do | 2022 | 0.505774 |
| nonlinear | zero | CSDI_native_coverage_block | do | 2023 | 0.52215 |
| nonlinear | zero | CSDI_native_coverage_block | obs | 2021 | 0.209928 |
| nonlinear | zero | CSDI_native_coverage_block | obs | 2022 | 0.198035 |
| nonlinear | zero | CSDI_native_coverage_block | obs | 2023 | 0.236444 |
| nonlinear | zero | CSDI_native_coverage_prefix | do | 2021 | 0.498939 |
| nonlinear | zero | CSDI_native_coverage_prefix | do | 2022 | 0.282725 |
| nonlinear | zero | CSDI_native_coverage_prefix | do | 2023 | 0.278459 |
| nonlinear | zero | CSDI_native_coverage_prefix | obs | 2021 | 0.273439 |
| nonlinear | zero | CSDI_native_coverage_prefix | obs | 2022 | 0.567236 |
| nonlinear | zero | CSDI_native_coverage_prefix | obs | 2023 | 0.984578 |
| nonlinear | zero | CSDI_native_random_block | do | 2021 | 0.574846 |
| nonlinear | zero | CSDI_native_random_block | do | 2022 | 0.511952 |
| nonlinear | zero | CSDI_native_random_block | do | 2023 | 0.691211 |
| nonlinear | zero | CSDI_native_random_block | obs | 2021 | 0.160582 |
| nonlinear | zero | CSDI_native_random_block | obs | 2022 | 0.159216 |
| nonlinear | zero | CSDI_native_random_block | obs | 2023 | 0.351933 |
| nonlinear | zero | CSDI_native_random_prefix | do | 2021 | 0.399767 |
| nonlinear | zero | CSDI_native_random_prefix | do | 2022 | 0.381681 |
| nonlinear | zero | CSDI_native_random_prefix | do | 2023 | 0.642594 |
| nonlinear | zero | CSDI_native_random_prefix | obs | 2021 | 0.458869 |
| nonlinear | zero | CSDI_native_random_prefix | obs | 2022 | 0.512356 |
| nonlinear | zero | CSDI_native_random_prefix | obs | 2023 | 0.554082 |
| nonlinear | zero | causal_MDN_causal_rollout | do | 2021 | 0.11064 |
| nonlinear | zero | causal_MDN_causal_rollout | do | 2022 | 0.0627234 |
| nonlinear | zero | causal_MDN_causal_rollout | do | 2023 | 0.0642916 |
| nonlinear | zero | causal_MDN_causal_rollout | obs | 2021 | 1.15161 |
| nonlinear | zero | causal_MDN_causal_rollout | obs | 2022 | 1.03988 |
| nonlinear | zero | causal_MDN_causal_rollout | obs | 2023 | 1.01713 |
| nonlinear | zero | causal_linear_AR | do | 2021 | 0.240304 |
| nonlinear | zero | causal_linear_AR | do | 2022 | 0.264754 |
| nonlinear | zero | causal_linear_AR | do | 2023 | 0.243223 |
| nonlinear | zero | causal_linear_AR | obs | 2021 | 1.18103 |
| nonlinear | zero | causal_linear_AR | obs | 2022 | 1.21841 |
| nonlinear | zero | causal_linear_AR | obs | 2023 | 1.17304 |
| nonlinear | sine | CSDI_native_coverage_block | do | 2021 | 0.89757 |
| nonlinear | sine | CSDI_native_coverage_block | do | 2022 | 0.721633 |
| nonlinear | sine | CSDI_native_coverage_block | do | 2023 | 0.653576 |
| nonlinear | sine | CSDI_native_coverage_block | obs | 2021 | 0.441017 |
| nonlinear | sine | CSDI_native_coverage_block | obs | 2022 | 0.131663 |
| nonlinear | sine | CSDI_native_coverage_block | obs | 2023 | 0.1465 |
| nonlinear | sine | CSDI_native_coverage_prefix | do | 2021 | 0.77257 |
| nonlinear | sine | CSDI_native_coverage_prefix | do | 2022 | 0.54278 |
| nonlinear | sine | CSDI_native_coverage_prefix | do | 2023 | 0.312914 |
| nonlinear | sine | CSDI_native_coverage_prefix | obs | 2021 | 0.538488 |
| nonlinear | sine | CSDI_native_coverage_prefix | obs | 2022 | 0.48054 |
| nonlinear | sine | CSDI_native_coverage_prefix | obs | 2023 | 0.760288 |
| nonlinear | sine | CSDI_native_random_block | do | 2021 | 0.871571 |
| nonlinear | sine | CSDI_native_random_block | do | 2022 | 0.72984 |
| nonlinear | sine | CSDI_native_random_block | do | 2023 | 1.02145 |
| nonlinear | sine | CSDI_native_random_block | obs | 2021 | 0.401774 |
| nonlinear | sine | CSDI_native_random_block | obs | 2022 | 0.22403 |
| nonlinear | sine | CSDI_native_random_block | obs | 2023 | 0.642794 |
| nonlinear | sine | CSDI_native_random_prefix | do | 2021 | 0.650466 |
| nonlinear | sine | CSDI_native_random_prefix | do | 2022 | 0.660158 |
| nonlinear | sine | CSDI_native_random_prefix | do | 2023 | 0.984353 |
| nonlinear | sine | CSDI_native_random_prefix | obs | 2021 | 0.431794 |
| nonlinear | sine | CSDI_native_random_prefix | obs | 2022 | 0.591408 |
| nonlinear | sine | CSDI_native_random_prefix | obs | 2023 | 0.877486 |
| nonlinear | sine | causal_MDN_causal_rollout | do | 2021 | 0.0951069 |
| nonlinear | sine | causal_MDN_causal_rollout | do | 2022 | 0.0665879 |
| nonlinear | sine | causal_MDN_causal_rollout | do | 2023 | 0.0727897 |
| nonlinear | sine | causal_MDN_causal_rollout | obs | 2021 | 1.16947 |
| nonlinear | sine | causal_MDN_causal_rollout | obs | 2022 | 1.05291 |
| nonlinear | sine | causal_MDN_causal_rollout | obs | 2023 | 1.08 |
| nonlinear | sine | causal_linear_AR | do | 2021 | 0.529639 |
| nonlinear | sine | causal_linear_AR | do | 2022 | 0.579841 |
| nonlinear | sine | causal_linear_AR | do | 2023 | 0.527272 |
| nonlinear | sine | causal_linear_AR | obs | 2021 | 1.57691 |
| nonlinear | sine | causal_linear_AR | obs | 2022 | 1.62452 |
| nonlinear | sine | causal_linear_AR | obs | 2023 | 1.56041 |

Validation learning-curve control; both fixed checkpoints retained, selected flag determined only by the mean of block/prefix DSM:

| corpus | native mask regime | updates | block DSM | prefix DSM | selected |
|---|---|---|---|---|---|
| linear_s2021 | CSDI_native_random | 2000 | 0.4053681 | 0.4209123 | False |
| linear_s2021 | CSDI_native_random | 8000 | 0.350008 | 0.3887503 | True |
| linear_s2021 | CSDI_native_coverage | 2000 | 0.4043295 | 0.4206287 | False |
| linear_s2021 | CSDI_native_coverage | 8000 | 0.3439288 | 0.3877821 | True |
| linear_s2022 | CSDI_native_random | 2000 | 0.4044675 | 0.4148438 | False |
| linear_s2022 | CSDI_native_random | 8000 | 0.3475743 | 0.3863158 | True |
| linear_s2022 | CSDI_native_coverage | 2000 | 0.4011083 | 0.4134775 | False |
| linear_s2022 | CSDI_native_coverage | 8000 | 0.3486895 | 0.3861508 | True |
| linear_s2023 | CSDI_native_random | 2000 | 0.400616 | 0.4117996 | False |
| linear_s2023 | CSDI_native_random | 8000 | 0.3464127 | 0.3807464 | True |
| linear_s2023 | CSDI_native_coverage | 2000 | 0.4014484 | 0.4138828 | False |
| linear_s2023 | CSDI_native_coverage | 8000 | 0.3429695 | 0.3810452 | True |
| nonlinear_s2021 | CSDI_native_random | 2000 | 0.3690772 | 0.4040235 | False |
| nonlinear_s2021 | CSDI_native_random | 8000 | 0.3176558 | 0.3692761 | True |
| nonlinear_s2021 | CSDI_native_coverage | 2000 | 0.37612 | 0.4070432 | False |
| nonlinear_s2021 | CSDI_native_coverage | 8000 | 0.3102449 | 0.3693986 | True |
| nonlinear_s2022 | CSDI_native_random | 2000 | 0.383664 | 0.4047494 | False |
| nonlinear_s2022 | CSDI_native_random | 8000 | 0.3147352 | 0.3639304 | True |
| nonlinear_s2022 | CSDI_native_coverage | 2000 | 0.3750953 | 0.402642 | False |
| nonlinear_s2022 | CSDI_native_coverage | 8000 | 0.3105552 | 0.3641467 | True |
| nonlinear_s2023 | CSDI_native_random | 2000 | 0.3666545 | 0.3926617 | False |
| nonlinear_s2023 | CSDI_native_random | 8000 | 0.3149952 | 0.3652404 | True |
| nonlinear_s2023 | CSDI_native_coverage | 2000 | 0.3643556 | 0.3957529 | False |
| nonlinear_s2023 | CSDI_native_coverage | 8000 | 0.3109991 | 0.3651704 | True |

Exact linear Bayes-noise validation references use the same normalization, fixed masks and native seeded noise. The realized and population quantities differ by finite validation noise. This is not a nonlinear oracle and a small DSM excess does not prove arbitrary-query calibration.

| corpus | view | realized Bayes DSM | population irreducible DSM |
|---|---|---|---|
| linear_s2021 | block | 0.335106 | 0.3355933 |
| linear_s2021 | prefix | 0.3783964 | 0.3761723 |
| linear_s2022 | block | 0.3372335 | 0.336161 |
| linear_s2022 | prefix | 0.3767056 | 0.3768188 |
| linear_s2023 | block | 0.3352809 | 0.336362 |
| linear_s2023 | prefix | 0.3741908 | 0.3766102 |

Native50-step sampler with an exact Gaussian conditional noise predictor, propagated analytically and checked using the unchanged impute method on32768 total native draws. The supplemental distance below is Gaussian W2 normalized by true covariance trace and must not be numerically equated to the primary finite sliced W1 above. Even exact denoising leaves about12% variance loss in prefix do simulation for this schedule. This is an existing fixed-reverse-variance/discrete-sampling issue, not evidence requiring a new world-model architecture.

| corpus/plan/operator | operator | Gaussian error vs do | Gaussian error vs obs | variance trace vs do |
|---|---|---|---|---|
| linear_s2021_zero_block | block | 0.5188883 | 0.06439748 | 0.311683 |
| linear_s2021_zero_prefix | prefix | 0.0609562 | 0.7377511 | 0.8818045 |
| linear_s2021_sine_block | block | 0.678521 | 0.06439748 | 0.311683 |
| linear_s2021_sine_prefix | prefix | 0.06095622 | 1.040041 | 0.8818045 |
| linear_s2022_zero_block | block | 0.5188643 | 0.06432322 | 0.3117299 |
| linear_s2022_zero_prefix | prefix | 0.06089999 | 0.7378302 | 0.88191 |
| linear_s2022_sine_block | block | 0.6785026 | 0.06432322 | 0.3117299 |
| linear_s2022_sine_prefix | prefix | 0.06090001 | 1.040097 | 0.88191 |
| linear_s2023_zero_block | block | 0.5188558 | 0.06429702 | 0.3117464 |
| linear_s2023_zero_prefix | prefix | 0.06088221 | 0.7378553 | 0.8819434 |
| linear_s2023_sine_block | block | 0.6784962 | 0.06429703 | 0.3117464 |
| linear_s2023_sine_prefix | prefix | 0.06088223 | 1.040115 | 0.8819434 |

Independent replay PASS: {"MDN_validation_checkpoints": 12, "checkpoint_hashes": 36, "native_validation_views": 48, "projected_W1_values": 65280, "raw_corpus_hashes": 6, "sample_law_hashes": 240, "selected_checkpoints": 18, "target_metric_views": 480, "trained_CPU_GPU_output_checks": 12}. Every raw law is hashed, every projected empirical W1 independently integrated using SciPy, and all36 validation-budget checkpoints and18 selections are checked. Maximum scaled numeric discrepancy=2.697269e-07; metric-specific tolerances and errors remain in INDEPENDENT_REPLAY65_V1.json. Native algorithms and complete configuration identities remain explicit.

Two synthetic systems with independently fitted checkpoints do not demonstrate a shared general-purpose world model, transfer to a new mechanism, a real intervention effect or clinical utility. No method was selected by these fixed query results. No new architecture is proposed or validated in this cycle.
