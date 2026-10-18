# Post-screen finite-population reference

The initial fixed SMC threshold is not calibrated to the finite-particle sampling floor. A nonzero Gaussian moment-projection discrepancy can occur even for independent exact target samples, especially at horizon 64. Interpreting every threshold failure as inadequate correction would be unsound.

After reading the initial sampler table, add exact-target iid references for every gain/horizon/particle/seed cell of terminal IS (144 rows). No temperature, sweep, particle, learned-model, or query setting is changed. This is a privileged statistical reference, not a deployable inference baseline. Each reference reuses its population size and replication labels; seeds generate independent draws, not identical innovations between algorithms. Report all rows, including references that themselves fail the original threshold. Also retain the original SMC result unchanged.

The primary target draws use an explicit innovation recursion; metric replay uses a separate SVD covariance expression. This supplement can contextualize error magnitude but does not establish equivalence, convergence, or a new-method gap with three replications.
