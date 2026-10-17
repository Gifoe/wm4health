# C54: Jointly learned RP-GSSM linear audit

Status: all 12 fixed-budget fits (RP native; ordinary LGSSM EM with the explicitly qualified initial-covariance normalization correction), training-only state maps, causal heldout scoring and independent numerical checks completed. This is a paper-recipe reconstruction using pinned native components; it is not the original full hyperparameter sweep or visual benchmark reproduction.

| Baseline | Source/version | Executed scope | Status |
|---|---|---|---|
| RP-GSSM | a75225d5edceb972bb872dc9a335608733567dd4 | Learned full A + native linear recognition, constant diagonal site precision, 2 tasks x 3 initializations x 10,000 updates | Complete |
| LGSSM EM | Dynamax 0.1.5 plus initial-moment normalization fix | 2 tasks x 3 initializations x 200 native EM iterations | Complete |
| True-parameter Kalman oracle | Original generator A,Q,C,R,m1,Q1 | Causal conditional mean for the same first 50 observations | Complete; oracle only |
| Official RP visual demos | Same public source | Source/configuration review, image files not attached | Not reproduced |

The ordinary EM baseline uses the same native E-step, regression updates and 200-iteration fit loop, with only initial covariance corrected to sum_Ezz/N - outer(sum_Ez/N,sum_Ez/N). The uncorrected seed0 first M-step was indefinite; two failed 200-iteration attempts were excluded and retained. Independent training-only qualification verified the corrected moments, all other parameters identical, and increasing finite likelihood. The protocol amendment predates heldout/state-readout evaluation. No upstream package/source was edited. See plan54_em_amendment_v2.json and outputs/QUALIFICATION_EM_FIX54_V1.json.

The three seeds vary initialization on one fixed dataset per dimension; they are not independent domains or independent data draws. All state labels enter only post-training diagnostic maps. Full-sequence smoothing uses future observations relative to individual times; causal prediction uses only the first 50 observations. Auxiliary observation maps use training observations alone and are applied equally to both learned models. Long-horizon factual MSE includes irreducible process noise; conditional-mean discrepancy measures deviation from the oracle conditional expectation. No significance or broad generalization claim is made.

All source files retain their pinned SHA256. Independent NumPy Kalman/RTS checks account explicitly for the RP source's fixed 1e-5 I addition to reported smoothed covariances. This does not equate the Gaussian auxiliary surrogate with the proper normalized RPM posterior. All model and readout checkpoint hashes are unchanged after evaluation.

