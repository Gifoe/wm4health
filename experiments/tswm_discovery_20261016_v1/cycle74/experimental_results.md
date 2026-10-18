# experimental results

Nine original TRAIN records,602924 fit points,582 DEV origins with128 free-running steps.

| Control | EMPS DEV NRMSE | F16 DEV NRMSE |
|---|---:|---:|
| ridge_arx_lag4 | 0.051000862 | 0.480657700 |
| ridge_arx_lag16 | 0.050615636 | 0.384285702 |
| ridge_arx_lag64 | 0.020693531 | 0.335650939 |
| persistence | 0.085628611 | 1.250291648 |
| train_mean | 0.956668734 | 0.873736932 |

All record-level metrics and hashes retained. Six ridge fits; ten per-system method rows; zero official-test or sensor-shift scores.

See CANONICAL_FINDING74_V1.md and the machine-readable source, data, result and verification receipts.
