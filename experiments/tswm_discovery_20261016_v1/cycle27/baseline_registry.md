# Baseline registry

| Baseline | Executed status | Information and interpretation |
|---|---|---|
| Official controlled Koopman/EM module | Literal native first seed through270updates; all original checkpoints retained | Native dimension-sweep settings, not full dimension sweep or all paper results |
| Source-equivalent batched EM | Three seeds,220/250/280 updates | Source defect preserved; iteration100 replay proves numerical fidelity |
| Standard corrected EM | Three seeds,1000 updates each | Same dimension, data, initialization and inference access; generator covariance transpose only; capped while improving |
| Native smoothed-time0 forecast | All source/corrected weights and budgets | Prefix information only; free propagation through prefix is a modeling/inference convention |
| Standard filtered-prefix forecast | Same weights and budgets | Ordinary last-prefix posterior propagation; no future observation access |
| ARX10 | Train-prefix ridge selection, refit, evaluated | Same observed controls/output,10lags each; fixed candidate regularization set |
| Persistence | All conditions | Last observed scalar output repeated |
| Known physics / exact exponential lift | All conditions | Privileged true current state and coefficients, diagnostic only |
| WorldTS native numerical core | Previously executed cycle22 | Recent framework reference; no new run or transfer result here |
| CoDA / Meta-iLaD / OSSM / SimDist | Literature/source audit; no new fitting | Exact assumptions and public-asset availability in research notes; never claim reproduced |

[Otto, Peitz and Rowley, actuated Koopman generators](https://arxiv.org/abs/2209.09977), [pinned official source](https://github.com/samotto1/KoopmanGeneratorEM/tree/ef830a6d4a6cff699492d54230fc2ad1729cab20).

Data hashes,fit-input boundary,source commit and all270prediction paths are in outputs/native_em/fit_input_receipt.json and outputs/controlled_em_study/results.json. Dependencies stay task-local under cycle26/vendor/native_control. All raw NPZ data/weights/predictions stay on the server; compact verified outputs and reports are mirrored locally. No prior artifacts or environments deleted.
