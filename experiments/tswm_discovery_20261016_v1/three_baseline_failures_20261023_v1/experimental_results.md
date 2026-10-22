# Evidence at registration

Server inspection of the original ledger confirmed 24 rows and the presence of independent float64 verification for all rows. The two unresolved numerical-match cases are:

| Model/workload | Original seed | Paper MSE | Saved reproduced MSE | Relative excess | Saved selection |
| --- | ---: | ---: | ---: | ---: | --- |
| LatentTSF/DLinear, Electricity H336 | 42 | 0.185 | 0.19671606163567038 | 6.3330% | epoch9, stopped after14 epochs |
| LatentTSF/DLinear, Electricity H720 | 42 | 0.218 | 0.23907836392582896 | 9.6690% | epoch15, stopped after20 epochs |

The author scripts themselves specify Electricity step60 and seed42. The saved run used those values. This excludes the simple explanation that the prior reproduction independently chose a sparse evaluation stride. It does not establish that the paper's unpublished exact run used identical evaluation origins.

New six-run reproduction and dense-hourly workload results are pending. No new performance or improvement is claimed in this registration document. Actual terminal receipts and generated reports supersede this initial status without deleting it.

Separately, C71's three native fits each completed2000epochs/32000module updates and32000proposal updates. Its original evaluator failed at the CPU-class guard before data access. Recovery now calls the unchanged evaluator from a recorded launch environment; do not merge partial or final C71 metrics into this three-paper comparison.
