# Actual results and independent confirmation

## Training sufficiency check

Native CSDI completed exactly20000 updates. CAL denoising MSE=0.024178682, selected step20000. Physical residual ratios are Duffing0.380050 and reservoir0.662679. Both beat the deterministic posterior mean on trajectory energy; only Duffing meets the<=.5 support threshold. The overall configuration is disqualified. Seeds2026902/2026903 were not launched; sealed comparison EVAL seed2026905 was not generated. No qualified native coupling failure claim exists.

## Independent baseline-only confirmation

This separately registered confirmation uses fresh seed20269080,16 independent four-plan groups per family, no selection or tuning on these queries. Both ordinary models were already frozen and CAL-qualified. It is not a favorable subset of a native-model comparison.

| Ordinary control | System | Energy | Exact reference | Excess vs reference | Physical residual ratio | Event probability MAE (two events) |
|---|---|---:|---:|---:|---:|---|
| unsupervised_pruned | Duffing | 0.207520593 | 0.205338775 | 1.0625% | 0.0457410 | [0.021060415200735597, 0.011254408088210054] |
| unsupervised_pruned | Reservoir | 0.027415440 | 0.027229472 | 0.6830% | 0.0378674 | [0.0022308349609375467, 0.0022621154785156835] |
| supervised_library | Duffing | 0.205354517 | 0.205338775 | 0.0077% | 0.0000283 | [0.0019738769531250314, 0.0012030029296875304] |
| supervised_library | Reservoir | 0.027363426 | 0.027229472 | 0.4919% | 0.0164165 | [0.0006576538085937952, 0.0005554199218750582] |


Energy is evaluated for the exact learned finite-mixture law against the exact nine-world reference, without forecast Monte Carlo error. Physical ratio is mean nearest-reference trajectory RMS divided by the reference mixture spread. The two actual events are max abs x>.75 and terminal x>.25. They were fixed before opening the fresh queries. Grouped bootstrap intervals for energy improvements over the deterministic mean are all below zero; see the machine-readable receipt. One fit per arm is not optimizer-seed robustness.

## Common persistent component versus independent branch sampling

For K4, independent nearest-reference-bin agreement is about2.6–2.7%. Shared-component sampling yields89.60%/92.22% for the unsupervised mixture and100%/100% for the supervised library. Every learned single-query marginal is unchanged by construction. Bin agreement is descriptive, not a certificate of exact common continuous physics.

The independent dense161-point k-grid evaluator gives K4 common-parameter RMS residuals of0.003873/0.000919 for the unsupervised control and0.00000965/0.000857 for the library (TRAIN-standardized state units). The common-versus-separate parameter-fit excess is small but not identically zero for all learned models. Exact shared reference worlds have gap<=1e-12. These are approximate discrete-simulator checks, not continuous-time or parameter-identification theorems.

## Replay and failures

Native CAL metrics replay512 values, maximum error5.41788836017e-14. The independent confirmation replays2432 values, maximum error2.77555756156e-16. Data/payload/checkpoint hashes and source immutability are checked.

The first classifier invocation stopped at step0 because int32 labels were invalid for CUDA CE; V2 only makes labels torch.long. The first confirmation invocation stopped during import before generating queries; V2 only adds the frozen simulator module path. The source attempts remain. Neither repair changes the scientific protocol, hyperparameters, sample selection or reported metrics.
