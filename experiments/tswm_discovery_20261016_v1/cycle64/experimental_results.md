# Executed results

Scope: fully observed synthetic linear Gaussian feedback only. All prespecified learned-law rows, including invalid corrections, are in `outputs/FIT_AND_GAUSSIAN_EVALUATION64_V1.json`; all supplementary same-joint rows are in `outputs/CAUSAL_FACTORIZATION64_V1.json`. Sampler and iid-reference files retain every replication. No real-data or learned neural generator result is claimed.

System: Y_t=.8Y_(t-1)+.7U_t+.5epsilon_t; U_t=-kY_(t-1)+.5eta_t; independent Gaussian innovations; Y0=.25. Logging policies have full support, not a uniform propensity lower bound. Fixed queried plans are zero and sinusoid; queries are never used for fitting/selection.

At H64, the table averages six related evaluations (3 independent training seeds x 2 fixed query plans). Discrepancy is sqrt(W2_squared/trace(true covariance)) between Gaussian laws. Invalid rows are not silently averaged away. The remaining horizon tables are in `outputs/SUMMARY64_V1.json`.

| gain | method | invalid/all | mean normalized discrepancy |
|---|---|---|---|
| 0.0 | observational_block | 0/6 | 0.0309328 |
| 0.0 | causal_AR | 0/6 | 0.00222993 |
| 0.0 | ratio_known_policy | 0/6 | 0.0309328 |
| 0.0 | ratio_estimated_policy | 0/6 | 0.0309068 |
| 0.0 | fixed_eigenvalue_floor | 0/6 | 0.0309068 |
| 0.0 | same_frozen_joint_temporal_causal_factorization | 0/6 | 0.0274592 |
| 0.25 | observational_block | 0/6 | 0.276002 |
| 0.25 | causal_AR | 0/6 | 0.00263787 |
| 0.25 | ratio_known_policy | 0/6 | 0.0418434 |
| 0.25 | ratio_estimated_policy | 0/6 | 0.0411688 |
| 0.25 | fixed_eigenvalue_floor | 0/6 | 0.0411688 |
| 0.25 | same_frozen_joint_temporal_causal_factorization | 0/6 | 0.0286553 |
| 0.75 | observational_block | 0/6 | 0.622425 |
| 0.75 | causal_AR | 0/6 | 0.00467166 |
| 0.75 | ratio_known_policy | 0/6 | 0.124622 |
| 0.75 | ratio_estimated_policy | 0/6 | 0.119993 |
| 0.75 | fixed_eigenvalue_floor | 0/6 | 0.119993 |
| 0.75 | same_frozen_joint_temporal_causal_factorization | 0/6 | 0.0464265 |
| 2.0 | observational_block | 0/6 | 0.834823 |
| 2.0 | causal_AR | 0/6 | 0.00855857 |
| 2.0 | ratio_known_policy | 6/6 | undefined |
| 2.0 | ratio_estimated_policy | 6/6 | undefined |
| 2.0 | fixed_eigenvalue_floor | 0/6 | 37.3287 |
| 2.0 | same_frozen_joint_temporal_causal_factorization | 0/6 | 0.108579 |

H64 independent validation observational conditional NLL per step (3 training-seed mean). The causal AR is first reconditioned on the entire validation action plan, so these columns score the same observational conditional target. Low observed NLL does not itself certify intervention fidelity.

| gain | oracle | block Gaussian | causal AR reconditioned |
|---|---|---|---|
| 0.0 | 0.726177 | 0.729162 | 0.726177 |
| 0.25 | 0.660976 | 0.66401 | 0.66098 |
| 0.75 | 0.422864 | 0.425812 | 0.422866 |
| 2.0 | -0.115237 | -0.112129 | -0.115236 |

Across all horizons/queries, 16/192 policy-ratio laws are invalid (8 known-policy, 8 estimated-policy). Exact policy removal from a finite fitted Gaussian need not remain integrable. This does not invalidate the exact data-generating density-ratio identity. The fixed precision floor guarantees an algebraic Gaussian, but is not a faithful repair. Same-joint temporal factorization is always normalized and gives a strong no-retraining reference; its remaining finite-estimation error is larger than the correctly specified, much more parameter-efficient causal AR in the hardest cell.

Population terminal importance weights, sinusoidal query. A non-PD 2Q_do-Q_obs means infinite second moment, reported population ESS fraction zero; finite empirical ESS does not certify coverage.

| gain | H | finite second moment | population ESS fraction |
|---|---|---|---|
| 0.0 | 1 | True | 1 |
| 0.0 | 4 | True | 1 |
| 0.0 | 16 | True | 1 |
| 0.0 | 64 | True | 1 |
| 0.25 | 1 | True | 1 |
| 0.25 | 4 | True | 0.699627 |
| 0.25 | 16 | False | 0 |
| 0.25 | 64 | False | 0 |
| 0.75 | 1 | True | 1 |
| 0.75 | 4 | False | 0 |
| 0.75 | 16 | False | 0 |
| 0.75 | 64 | False | 0 |
| 2.0 | 1 | True | 1 |
| 2.0 | 4 | False | 0 |
| 2.0 | 16 | False | 0 |
| 2.0 | 64 | False | 0 |

Finite-population references and ordinary SMC (means over three independent MC seeds). Metrics concern Gaussian projection of empirical moments, not exact empirical-law W2. The original operational rule was discrepancy<=.1 and terminal tail error<=.03 in each replicate. It was fixed before initial SMC execution; exact iid references were added after the screen and are disclosed as such. The rule itself rejects all H64/P512 exact iid replications. Therefore threshold failures alone cannot diagnose an algorithmic research gap.

| gain | H | particles | iid discrepancy | SMC discrepancy | iid pass | SMC pass |
|---|---|---|---|---|---|---|
| 0.75 | 16 | 128 | 0.169952 | 0.24259 | 0/3 | 0/3 |
| 0.75 | 16 | 512 | 0.0745503 | 0.143832 | 3/3 | 0/3 |
| 0.75 | 64 | 128 | 0.256075 | 0.439691 | 0/3 | 0/3 |
| 0.75 | 64 | 512 | 0.126358 | 0.321327 | 0/3 | 0/3 |
| 2.0 | 16 | 128 | 0.14839 | 0.406681 | 0/3 | 0/3 |
| 2.0 | 16 | 512 | 0.0727683 | 0.337212 | 3/3 | 0/3 |
| 2.0 | 64 | 128 | 0.248954 | 0.682387 | 0/3 | 0/3 |
| 2.0 | 64 | 512 | 0.118002 | 0.524669 | 0/3 | 0/3 |

SMC still has larger moment discrepancy than the iid reference in these fixed cells; no temperature/sweep/budget search was performed. That leaves efficiency room for this sampler configuration, but does not show failure of full TDS, stronger SMC or causal generators. Tail errors are retained separately in the raw JSON; a small tail error alone cannot establish the full law.

Oracle affine DDIM at gain2/H64. Its unchanged observed, plug-in-corrected and exact-target scores share discretization and initial N(0,I). The exact-target control separates solver error from wrong guidance.

| steps | score | normalized discrepancy | variance trace ratio | terminal tail abs error |
|---|---|---|---|---|
| 64 | observed | 0.878673 | 0.0675582 | 0.0240544 |
| 64 | plugin | 0.773379 | 0.140242 | 0.0239367 |
| 64 | exact_target | 0.0446827 | 0.91265 | 0.00537968 |
| 256 | observed | 0.873029 | 0.0723687 | 0.023994 |
| 256 | plugin | 0.764096 | 0.151286 | 0.0238083 |
| 256 | exact_target | 0.0112821 | 0.977646 | 0.00146146 |
| 1024 | observed | 0.871617 | 0.0736054 | 0.0239742 |
| 1024 | plugin | 0.761772 | 0.154143 | 0.0237675 |
| 1024 | exact_target | 0.00299078 | 0.994354 | 0.0004122 |

The denoised-mean plug-in correction does not equal the noisy log expected density-ratio score. Exact noisy-target scoring substantially repairs this oracle problem. Both facts are known posterior-guidance considerations; no new sampler follows from them alone. Score rows at all fixed noise levels and full 144 DDIM laws are retained.

Independent replay: PASS. 12 raw-corpus hashes, 48 checkpoint hashes, 480 original learned-law rows, 96 same-joint rows, 168 particle-law rows and 144 DDIM final-law moment rows checked. Alternate QR regression, direct sequential covariance regression, structural innovation covariance and SVD Bures arithmetic give maximum scaled discrepancy 4.26279e-10. Independent iid-reference arithmetic max absolute W2 difference 2.27374e-13. Oracle structural-conditioning difference 1.55431e-15; noisy-ratio score finite-difference error 8.8114e-12. This replay does not independently execute a full denoising pipeline.

Fit campaign elapsed 8.683s; sampler campaign elapsed 7.223s, measured on CPU and excluding setup, reference and replay. These are execution receipts, not a matched-compute speed claim. Sampling seeds and shared prefixes are not independent biological systems.
