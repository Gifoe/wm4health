# Cycle9 experimental evidence

All numeric experiments ran on the private RTX5090 server. Original cycles1–8 were preserved.

## Native inverse-score controlled transfer

| Seed | Native orientation mass | Generic quadratic orientation mass | Native excess nats | Quadratic excess nats |
|---|---:|---:|---:|---:|
| 2021 | 0.500000 | 0.994600 | 0.693147 | 0.005415 |
| 2022 | 0.500000 | 0.994603 | 0.693147 | 0.005411 |
| 2023 | 0.500000 | 0.994640 | 0.693147 | 0.005375 |


Six fits,600updates each; seeds2021/2022/2023, independently rotated train/validation/test angle grids. Native function `action_discrimination_loss` was extracted unchanged by AST from pinned AC-MTM `train.py`. Endpoint inputs contain only loop orientation. Each endpoint class contains20 indistinguishable held-out rotations; the correct index loss floor is log20, not zero. A correct orientation score can therefore have only5% index accuracy. Native score has16 free query parameters; generic degree-two feature score has72. This is an expressivity/known-repair diagnostic, not a capacity-matched method superiority claim or a native visual-control replication.

At zero query, native gradients are at most6.62e-8; tested random query perturbations increase loss. More importantly, row-constant cancellation gives a convex logsumexp objective; with balanced opposite rotations the class-conditional action mean is zero. The zero query is therefore a population/finite-balanced-bank minimizer regardless of an endpoint MLP's capacity. The test establishes a restricted-score boundary, not failure of all contrastive inverse learning. A numerical Hessian was not needed or computed; convexity is an analytic argument.

ACRO's released continuous-action pretraining normalizes action labels and predicted actions. Collinear positive actions1,2,0.25 normalize identically. This is an algebra/code contract, not a full ACRO training result. EP-IDM was source-audited only; its linear theorem and explicitly acknowledged endpoint-rank limitation are not contradicted by a nonlinear square-loop construction.

## Published DoTime checkpoint, official corrected suite

| Model | Full-suite RMSE | Effect-sign accuracy | Scored effects |
|---|---:|---:|---:|
| PFNRef | 0.666082 | 0.538108 | 4356 |
| Mean | 0.613954 | 0.534894 | 4356 |
| VAR-OLS | 0.674327 | 0.525941 | 4356 |
| NaiveOLS | 0.561588 | 0.773186 | 4356 |

Official `dot-Identifiability-v1`1.2.0:12,150episodes,9structures,1350perstructure. Native loader verified released manifest MD5s; shared-noise and target QA passed all12,150episodes. Level-sign denominator10,235 differs from effect-sign denominator4356. Near-zero effects are excluded by the native threshold0.1; raw level errors use every episode. `bow_graph` is included in the pooled descriptive result but is explicitly not nonparametrically identified. These accuracies are not guarantees of causal identification.

Checkpoint:`s9ho_all_causal/do_over_time_pfn_best.pt`, HF revision`bec6618c8b25f3a5d8cf608d4c597978d5a694ac`, SHA256`214db2fab16fb5fa17e68244abc0df3b7b0154b9fb2d2ce16b2be3f4d5d93618`,295,027,515bytes. It loaded24,922,633parameters strictly with native `load_dotpfn`; static checkpoint inspection found no unsafe pickle globals. It is the original published s9 causal model, **not the later s13 or redesigned recipe**. Later s10/s13 artifacts are visible in the retained metadata but were not executed here. No interventional-vs-observational training superiority is inferred.

Native `PFNRef` preprocessing/prediction is preserved. Batched inference agrees with8native scalar calls within8.94e-08; native final scorer is also retained. One checkpoint, no training, no hyperparameter selection. The fixed540episode subset was followed by one full-suite pass to remove sampling uncertainty. Full prediction/scoring took40.75s after preparing/loading the suite. Peak allocated CUDA memory719.6MiB. This excludes source/data downloads and parsing.

The nominal90%interventional outcome interval covers0.884033; mean width2.363502; no adjacent quantile crossings. These are outcome-level quantiles, not individual counterfactual-effect intervals. NaiveOLS is an unadjusted associative control with omitted-variable bias, not a causal oracle. Mean and VAR-OLS are action-blind controls; grouping is explicit.

This poor s9 result has sufficient existing explanations. The author's pinned erratum/re-test ledger already withdraws the interventional advantage and documents normalization, role/addressing and temporal aggregation repairs. Our execution confirms an available artifact; it does not discover those defects or establish a new gap.

Raw logs:`outputs/inverse_score/*log`,`outputs/dotime/*log`,`outputs/dotime_full/*log`. Predictions/configs/metrics are retained separately. A metadata-print wrapper initially used `suite.metadata` instead of native `suite.meta`; it failed after the download completed. The corrected wrapper reused the cached verified data. No scientific training job was restarted to address a timeout.
