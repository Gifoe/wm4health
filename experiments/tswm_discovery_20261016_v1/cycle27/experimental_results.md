# Cycle27: competent controlled hidden-state baseline

General reusable time-series world modeling remains the objective. This cycle rejects one weak justification for a new framework. Final accepted directions: 0. Overall goal: ACTIVE, UNACHIEVED.

Native released slow-manifold generator/settings plus an explicitly new independent-control-schedule evaluation. This is not a complete paper dimension sweep, a new neural model, cross-dataset transfer, or proof that all controlled partial-observation problems are solved.

The fixed native mechanism is x1dot=-x1+u, x2dot=5*(x1^3-x2), observed y=x2+N(0,.01). dt=.01. Fifty training trajectories; only the first250 observations/249 transitions are used. Augmented latent dimension5 includes a constant and four stochastic coordinates. Three initialization seeds share that one training dataset. Each independent regime has64 fresh initial/noise/control trajectories at fixed physical parameters, not64plants. Prefix25/100/250; horizon251. All models receive the observed prefix and scheduled future controls, never true hidden state or future observations.

| Independent control dwell, prefix=250 | Source filtered MSE | Corrected filtered MSE | ARX MSE | Corrected 95% coverage |
|---|---:|---:|---:|---:|
| 50 (same distribution) | 2.007956 | 0.008857225 | 4.449711 | 97.535% |
| 5 (faster switching) | 0.04430864 | 0.0004196093 | 0.7488595 | 96.039% |
| 100 (slower switching) | 8.883271 | 0.01538269 | 17.47707 | 97.101% |

These are mean clean-output MSEs across all three initializations, not best-seed scores. Complete110 aggregates preserve all270 rows, two training checkpoints, two state-initialization conventions, all prefixes and the within-training-trajectory continuation. See outputs/analysis/aggregate.csv. The known-physics reference receives privileged current hidden state and is a numerical diagnostic, not a fair learned comparator.

The displayed final-source versus final-corrected contrast changes achieved training length as well as source math:source stops at220–280 while corrected continues to1000. It is not a matched-update single-factor estimate of the transpose's causal effect. Checkpoint100 supplies a matched-update diagnostic in the complete table. The defensible conclusion is competence of corrected standard EM with the predeclared train-only continuation,not that the entire final MSE reduction is caused by one code line alone.

The source smoother returns Cov(z_k,z_k+1|Y); the generator regression requires its transpose. Exact joint-Gaussian audit agrees with the native smoother at1.249e-16 and shows a .522644 generator discrepancy. Corrected standard EM changes only that generator-regression transpose, not the process-noise formula. This is a baseline implementation repair, not innovation. Source mode faithfully preserves the defect.

| Mode | Seed | Iterations | Train-only stopping | Latest average LL gain |
|---|---:|---:|---|---:|
| source_mode | 20262711 | 280 | train_likelihood_tolerance | -0.0341471 |
| source_mode | 20262712 | 250 | train_likelihood_tolerance | -0.0239599 |
| source_mode | 20262713 | 220 | train_likelihood_tolerance | -0.0364387 |
| corrected_em | 20262711 | 1000 | iteration_cap_underconverged | 0.202548 |
| corrected_em | 20262712 | 1000 | iteration_cap_underconverged | 0.130238 |
| corrected_em | 20262713 | 1000 | iteration_cap_underconverged | 0.136731 |

All corrected fits are still improving at the1000-update cap; do not claim convergence or residual model-class failure. Source stopping values are negative gains, not successful stationary EM convergence. The finite-budget corrected baseline is already competent, so extending it solely to seek failure is unnecessary. These budgets and stopping conditions were frozen before held-out scoring. Training log likelihood per observation and all seed traces remain available.

Resume bookkeeping limitation:first source seed already had negative average LL gain at native270;the resumed backend nonetheless ran one additional10-update block to280. That checkpoint did not improve the train-best selection. Actual traces and stopping counts are preserved;do not claim identical stop timing or silently rewrite them. This does not affect corrected fits or their declared1000-update budget.

State estimation also matters: native smoothed-time0 free-run and standard last-prefix filtering use the same weights and inputs but different conditioning approximations. Filtering improvements are ordinary inference repairs. Paired path bootstrap averages losses over initializations first;4000 resamples,seed20262781,95%percentile intervals. All contrasts, including negative or uncertain ones, are saved. They are exploratory, not multiplicity-corrected or independent training-seed inference. Different dwell cohorts are independent, not paired causal interventions.

Runtime of the amended study: 261.828s on one CPU thread; no GPU training. This excludes the preserved original slower native worker and one-time JIT/proof costs. Source-equivalent batching reproduced literal native iteration100 parameters within4.561e-11 and likelihood within2.758e-9. The source-mode first seed resumed native iteration270; other five fits used their predeclared initializations. Original source, logs and checkpoints remain unchanged.

Independent verification:4data hashes,387checkpoint hashes,270prediction hashes,242000physical state values,120posterior histories,270complete forecast arrays and1830scores. Canonical NumPy filtering/RTS, closed-form physical dynamics, independent augmented least-squares ARX and all-score recomputation passed. Maximum forecast mean difference2.561e-11,variance4.108e-9,score2.843e-14. Physical odeint/closed-form discrepancy1.596e-6 under predeclared tolerances. Result SHA256=ba8f88504cdb819988f131e783d776d57449b25812a1c4870e98cb7c9b242deb.

Gaussian NLL/coverage are pointwise marginal metrics. They do not establish correct joint path uncertainty, control policies, or general physical faithfulness. [Otto, Peitz and Rowley, actuated Koopman generators](https://arxiv.org/abs/2209.09977), [pinned official source](https://github.com/samotto1/KoopmanGeneratorEM/tree/ef830a6d4a6cff699492d54230fc2ad1729cab20).
