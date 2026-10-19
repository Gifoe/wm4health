# C46 complete controlled ordinary-composition screen

UTC 2026-10-17T11:56:20.361657+00:00. Eight fresh neural fits, 27250 updates in total, sum of fit wall times 2559.9s (two concurrent CPU workers; this sum is not elapsed time or GPU time). Device selected by discarded TRAIN-only throughput. 1/8 selected at final update; 0/8 hit hard8000 cap. Original paper headline reproduction: no. New proposed methods:0.

| Ordinary neural arm | Seen ES | Held-out ES | Seen MSE | Held-out MSE | Held-out coverage90 |
|---|---:|---:|---:|---:|---:|
| O0D0 | 0.560776 | 0.559250 | 0.696858 | 0.686203 | 0.9182 |
| O0D1 | 0.549162 | 0.549072 | 0.667328 | 0.659076 | 0.9215 |
| O1D0 | 0.556948 | 0.556522 | 0.688653 | 0.680345 | 0.9083 |
| O1D1 | 0.553535 | 0.552041 | 0.674226 | 0.663501 | 0.9245 |

| Ordinary comparator | Seen ES | Held-out ES | Held-out MSE | Held-out coverage90 |
|---|---:|---:|---:|---:|
| ordinary_aligned_PCA_LGSSM | 0.450552 | 0.440893 | 0.431372 | 0.9016 |
| ordinary_unconditional_view_paths | 0.573703 | 0.576912 | 0.749344 | 0.8640 |
| zero_point_control | 0.788599 | 0.791791 | 0.734469 | not probabilistic |
| persistence_point_control | 0.969987 | 0.966596 | 1.159834 | not probabilistic |

All scores lower-is-better except coverage. ES is joint16×8 future energy normalized by sqrt128. Neural numbers average the two independent fits. Deterministic controls have degenerate intervals and no probabilistic coverage claim.

Interaction I=E10+E01−E00−E11 (positive=extra reduction beyond individual benefits):

```json
[
  {
    "seed": 2026461,
    "heldout_interaction": -0.015536316583165899,
    "seen_interaction": -0.020632757999313522
  },
  {
    "seed": 2026462,
    "heldout_interaction": 0.004143854836001992,
    "seen_interaction": 0.0042317855016638095
  }
]
```

Cheap ordinary alignment/SSM already competes with or beats the neural combinations on held-out path quality; a missing new world-model mechanism is not established.

One controlled carrier, four known system identities and four known views; diagonal temporal combinations held out, but all views/systems seen elsewhere and static anchors include those combinations. No independent real-domain validation, unknown-module discovery or post-training transplantation has been established.

O1 constrains both recognition and generative observation networks to ignore system bits. Thus O1/O0 is a joint modeling/inference restriction, not a pure observation-likelihood ablation. A context-aware recognition network with a shared decoder is a standard stronger variational alternative and has not been tested. O1 nominal parameter counts also include256 first-layer weights fed constant-zero system inputs. The cheap control uses a different fitting objective/capacity and exact Gaussian conditioning in its learned coordinates; no isolated architecture ranking follows.

Independent NumPy/SciPy verifier: 15360 origins in 192 retained joint-draw groups. Max absolute energy arithmetic discrepancy 6.38e-07. All selected+last checkpoint hashes, common-source hashes, eight schedules and original author core SHAs checked. EVAL opened only after all eight frozen selections and the ordinary control VAL selection.

Raw per-origin scores, samples, selections, source and data remain in the server C46 directory. Sampling Monte Carlo replicates are not additional training seeds. No test-set hyperparameter adjustment was made.
