# Cycle11 verified experimental results

## What actually ran

The exact-kernel diagnostic uses3independent context seeds x5flip probabilities x2horizons x2normalizations=60score rows, with32768iid stationary contexts per seed. Full configs, common random design arrays, logs and every grid result are retained. No parameter fitting is involved. Separately,12neural fits used4arms x3seeds,6000updates each on the privateRTX5090server. NativeSIGReg and nativeInfo-LDM logdet-entropy functions were reused unchanged with explicit controlled settings. These are new controlled experiments and source-component transfers, not original-paper reproductions.

## Oracle score discriminator

Negative score difference means a lower oracle latentEnergy risk for the fold. Information loss is in nats. Entries show mean+/-sampleSD across3MonteCarlo seeds, not a confidence interval. The p-grid shares designs within seed and is not a set of independent replications.

| flip p | folded-minus-identity ES,H1 | folded-minus-identity ES,H8 | one-step predictive MI lost | exact extra raw BayesMSE |
|---|---:|---:|---:|---:|
| 0 | 0.15460447 +/- 0.00139 | 0.13327036 +/- 0.000651 | 0.58446215 +/- 0.0023 | 1.62 |
| 0.1 | 0.0097330246 +/- 0.00165 | -0.024802674 +/- 0.00384 | 0.31310858 +/- 0.00239 | 1.0368 |
| 0.25 | -0.14687644 +/- 0.00438 | -0.084005608 +/- 0.00331 | 0.11057113 +/- 0.00201 | 0.405 |
| 0.4 | -0.23081159 +/- 0.00297 | -0.10203076 +/- 0.00255 | 0.016937896 +/- 0.00037 | 0.0648 |
| 0.5 | -0.24859851 +/- 0.00357 | -0.10282388 +/- 0.00127 | 0 +/- 0 | 0 |

Both frame marginals are exactlyN(0,I), dimension2. Unit-circle normalization also retains the counterexample atp=.25; all results are in theJSON. Identical frame marginals do not mean identical future-block joint distributions. The correct fixed-target conditional law is used on each side; no propriety theorem is violated. The analytic positive-noise existence argument and exact rawMSE formula are inSCORE_WITNESS.md.

## Actual learned-state test

Final fixed-budget checkpoints, independent state readouts, mean+/-sampleSD over3training seeds. StateMSE is summed over the2raw coordinates. Energy/information active parameter count differs from decoded arms; compute and active capacity are not claimed matched. No test/readout metric selected a checkpoint.

| arm | linear readout stateMSE | ExtraTrees stateMSE | sign-fiber separation ratio | active parameters |
|---|---:|---:|---:|---:|
| energy | 0.0019269882 +/- 0.00135 | 0.0076370873 +/- 0.00123 | 1.9981227 +/- 0.00163 | 11434 |
| energy_information | 0.00087371766 +/- 0.000217 | 0.0054770134 +/- 0.00209 | 1.9991824 +/- 0.000956 | 11434 |
| energy_observed | 0.0017726943 +/- 0.00106 | 0.0066242671 +/- 0.00196 | 1.998261 +/- 0.00145 | 15916 |
| observed_only | 0.0013503103 +/- 0.000922 | 0.0068240341 +/- 0.00121 | 1.9985319 +/- 0.00144 | 15916 |

Identity readout controlMSE is7.53902e-17; exact-fold control is2.0232561. A ratio near2preserves sign fibers in a near-affine embedding; exactfold gives0. The simple energy-only arm has stateMSE range[0.00055380197,0.0032446139], not the order2error of folding. Its three covariance spectra remain well conditioned. All12fits retain the observed state; no actual alias failure is demonstrated. Small absolute readout differences among healthy arms do not justify a new method.

Training-plus-probe elapsed780.018s. PyTorch tracked peak tensor allocation37673984bytes, excluding driver/context/reserved memory. The verifier replayed all60score rows from stored draws, checked analyticGaussian Energy risks, checked the weightedMC estimator against independent knownGaussian optima, safely loaded all12checkpoints and replayed all linear readouts. No scientific fit was rerun by verification.
