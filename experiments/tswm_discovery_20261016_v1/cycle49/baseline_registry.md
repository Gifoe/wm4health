# C49 completed baseline registry

UTC 2026-10-17T14:53:33.295967+00:00.8 fresh fits /18800 formal updates. Primary:16 raw groups /48 summary groups /12288 model-context-reveal scores, 512 independent synthetic physical paths and1536 paired density history contexts,128 iid future draws per history. Independent all-path/all-score replay PASS: max model difference0.00016, energy arithmetic4.44e-16, observation score1.82e-07.0 real domains.

This is an original controlled shared-core experiment, not an original-paper headline reproduction. Kernel/variational/marked-observation methods are prior mechanisms.
Both predictive cores9458 parameters; complete-clock ELBO additionally uses13944 training-only future-recognition parameters. Forecasts use only observed history. Same predictive initialization/data schedules per case/seed were independently checked.
Source physics: rotating OU and coupled bistability,2D signals. Learned latent8, neural autonomous discrete transitions, learned signal decoder; known calibrated sensor and density. TRAIN1024/VAL128/EVAL256 independent paths per process. Clean physical truth separated from training observation files; no EVAL model selection.
Formal device CPU selected by close discarded whole-step timings; CPU/CUDA ranking differed by less than1% and is not an efficiency conclusion.12 discarded qualification updates excluded.

| Fit | Updates | Selected step | Selected-step seconds | Whole-fit seconds |
|---|---:|---:|---:|---:|
| bistable__full_clock_elbo__seed2026491 | 2600 | 2600 | 154.248 | 154.254 |
| bistable__full_clock_elbo__seed2026492 | 2200 | 1600 | 94.647 | 130.186 |
| bistable__rb_score__seed2026491 | 1800 | 1600 | 132.555 | 148.982 |
| bistable__rb_score__seed2026492 | 1600 | 1600 | 131.406 | 131.410 |
| rotating__full_clock_elbo__seed2026491 | 3000 | 2000 | 120.653 | 181.302 |
| rotating__full_clock_elbo__seed2026492 | 2800 | 2800 | 167.337 | 167.343 |
| rotating__rb_score__seed2026491 | 2800 | 2000 | 165.220 | 231.132 |
| rotating__rb_score__seed2026492 | 2000 | 1400 | 115.274 | 164.655 |

Sum whole-fit time1309.264s, not GPU hours. Hard caps:['rotating__full_clock_elbo__seed2026491']; final-step selections:['bistable__full_clock_elbo__seed2026491', 'bistable__rb_score__seed2026492', 'rotating__full_clock_elbo__seed2026492'].
All200-step validation checkpoints, selected/last states, terminal optimizer/recognizer/RNG, schedules and raw forecasts retained. Selected validation checkpoints store predictive core weights; selected-time recognition/optimizer snapshots are not separately stored. A full matched-time replay of training would be needed to reconstruct them if required for posterior diagnostics.
