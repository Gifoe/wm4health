# C46 TRAIN/VAL failure-attribution addendum

Generator replay on every retained TRAIN/VAL trajectory: max observation mismatch0. Independent augmented-SVD ridge refit and raw-score arithmetic max discrepancy1.11e-16. The true states and probe labels are auditor-only; no forecasting model or selection receives them.

| Model | VAL current-state affine probe R2 | VAL future-state probe MSE | VAL q-mean decoder reconstruction MSE |
|---|---:|---:|---:|
| o0d0_seed2026461 | 0.246472 | 0.446349 | 1.070616 |
| o0d0_seed2026462 | 0.218773 | 0.440816 | 1.103710 |
| o0d1_seed2026461 | 0.249482 | 0.439331 | 0.776522 |
| o0d1_seed2026462 | 0.260708 | 0.433753 | 0.829136 |
| o1d0_seed2026461 | 0.227867 | 0.434388 | 0.912838 |
| o1d0_seed2026462 | 0.186321 | 0.438660 | 1.193778 |
| o1d1_seed2026461 | 0.233592 | 0.440019 | 0.858595 |
| o1d1_seed2026462 | 0.304401 | 0.433617 | 0.741268 |
| ordinary_aligned_PCA_LGSSM | 0.910205 | 0.390382 | not comparable |

Zero decoder prediction on these same normalized VAL history frames has MSE1.693527. Decoder values above are the output at the auxiliary mean, not an expectation of a nonlinear decoder or a likelihood estimate.

The cheap aligned Gaussian model retains substantially more linearly decodable current-state information in this fixture. This is a concrete warning against attributing the native neural deficit to missing composition or proposing a new architecture from its rollout error. Optimization, objective/inference restrictions and nonlinear latent gauge effects remain competing explanations. The affine probe alone is not a universal test of information loss or physical identifiability.

Future-state MSE includes irreducible process randomness. These probes are not new forecasting methods and are not an oracle conditional-reference bound. No sealed EVAL values were used here, and no fit, checkpoint selection or hyperparameter was changed. A claimed new mechanism would still require a new hypothesis and independent held-out evidence.
