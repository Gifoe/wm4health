# C49 completed descriptive analysis

Fits8, formal updates18800. Independent all-future/all-score NumPy verification passed.

Physical energy uses full independent-pair scoring and RMS coordinate scaling. Neural rows average two training seeds; seed ranges are retained in aggregate.csv.

| Process | Sensor density | Arm | Reveal bins | Physical future energy | Mean forecast MSE | Observed score |
|---|---:|---|---:|---:|---:|---:|
| rotating | 0.25 | rb_score | 0 | 0.3126661 | 0.2150861 | -0.6934038 |
| rotating | 0.25 | rb_score | 12 | 0.2898969 | 0.1931681 | -0.8388083 |
| rotating | 0.25 | full_clock_elbo | 0 | 0.3120794 | 0.2123646 | -0.6931611 |
| rotating | 0.25 | full_clock_elbo | 12 | 0.2926532 | 0.1921886 | -0.8387463 |
| rotating | 1.0 | rb_score | 0 | 0.2900367 | 0.1826169 | -0.4329191 |
| rotating | 1.0 | rb_score | 12 | 0.2618782 | 0.1553355 | -0.6751537 |
| rotating | 1.0 | full_clock_elbo | 0 | 0.2857215 | 0.1783273 | -0.4335985 |
| rotating | 1.0 | full_clock_elbo | 12 | 0.2581330 | 0.1497279 | -0.6759874 |
| rotating | 4.0 | rb_score | 0 | 0.2926549 | 0.1839907 | -0.1729880 |
| rotating | 4.0 | rb_score | 12 | 0.2560852 | 0.1437511 | -0.4969119 |
| rotating | 4.0 | full_clock_elbo | 0 | 0.2832904 | 0.1754864 | -0.1752372 |
| rotating | 4.0 | full_clock_elbo | 12 | 0.2491861 | 0.1386111 | -0.4990623 |
| bistable | 0.25 | rb_score | 0 | 0.3390242 | 0.3048609 | -0.6893387 |
| bistable | 0.25 | rb_score | 12 | 0.2985495 | 0.2392920 | -0.8102756 |
| bistable | 0.25 | full_clock_elbo | 0 | 0.3357472 | 0.2861557 | -0.6898982 |
| bistable | 0.25 | full_clock_elbo | 12 | 0.2986814 | 0.2240175 | -0.8100094 |
| bistable | 1.0 | rb_score | 0 | 0.2935993 | 0.2188959 | -0.4398928 |
| bistable | 1.0 | rb_score | 12 | 0.2496341 | 0.1574062 | -0.6439133 |
| bistable | 1.0 | full_clock_elbo | 0 | 0.2882695 | 0.2061380 | -0.4408742 |
| bistable | 1.0 | full_clock_elbo | 12 | 0.2455378 | 0.1472883 | -0.6444662 |
| bistable | 4.0 | rb_score | 0 | 0.2888830 | 0.2077640 | -0.2029929 |
| bistable | 4.0 | rb_score | 12 | 0.2283272 | 0.1247738 | -0.4763625 |
| bistable | 4.0 | full_clock_elbo | 0 | 0.2833527 | 0.1978221 | -0.2033461 |
| bistable | 4.0 | full_clock_elbo | 12 | 0.2273811 | 0.1236211 | -0.4770840 |

## Evidence boundaries

A density change also changes history information. A larger error under sparse sensing is not itself evidence that the dynamics model failed. Compare methods on the same histories and pair physical episodes.
Observed scores omit a distribution-independent constant; negative values are valid. Compare methods within the same context. Relative percentages are not reported for these scores.
Across-reveal comparisons use only identical final12 targets. Full24 and remaining12 horizon scores are different tasks.
Episode-bootstrap intervals keep the three density views together. They condition on the two fitted training seeds and do not certify generalization across training seeds, process families or real domains.
Both arms share predictive capacity/initializations/data schedules; the ELBO control has additional future-recognition parameters. This screen is not a complete matched-time frontier or external baseline study.
Hard caps: ['rotating__full_clock_elbo__seed2026491']. Final-update selections: ['bistable__full_clock_elbo__seed2026491', 'bistable__rb_score__seed2026492', 'rotating__full_clock_elbo__seed2026492'].
A new framework contribution still requires a useful distinct interaction, current prior-art differentiation, competent controls and independent real-domain evidence.
