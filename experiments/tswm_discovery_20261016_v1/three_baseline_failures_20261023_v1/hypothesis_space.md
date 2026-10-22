# Competing explanations for the first observed failure

| Explanation | Discriminating evidence | Current status |
| --- | --- | --- |
| Metric/data/instrumentation error | Recompute saved predictions against raw chronological targets and verify source/data/AE identities; compare a fresh seed42 execution | Earlier float64 checks pass; current replay pending |
| Initialization or validation-selection variability | Frozen seeds42/43/44, same released recipe, all epochs/selections reported | Registered before new runs |
| Sparse origins hide or exaggerate operational performance | Evaluate each frozen selected model on all hourly origins and all60 phase partitions; compare its original sparse subset | Registered; cannot select a phase or retrain from Test |
| Forecast representation creates a transferable capability failure | Requires a failure surviving correct execution, training and simple comparable controls, followed by an independent workload | Not established; no new method proposed |

The paper gap is not assumed to have a unique explanation. Unavailable original author run artifacts may leave the reproduction discrepancy unresolved. Such uncertainty is reported, not converted into a mechanism claim.
