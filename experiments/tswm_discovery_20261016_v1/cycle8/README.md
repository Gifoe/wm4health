# Cycle8: real wind-tunnel qualification and controls

Evidence checkpoint; the full discovery goal remains **ACTIVE and unachieved**. No final direction or new method is accepted.

38 recordings /965,322rows /64intervention blocks audited. Completed ordinary dynamics/time controls on18recordings plus128sensor-stream quantization diagnostics. Ten primary PDFs and42pinned source/metadata files retained. Preserve sealed main/cycles2–7.

This cycle executes **new real-data control experiments**, not original WorldTS, TAWM, RP-GSSM or JEPA reproductions. All numerical fits and audits run on the remote research server; ordinary CPU ridge/gradient boosting is deliberately used before expensive neural training. Native source/configuration is pinned for the data-generating protocols and inspected firmware, not claimed to be the exact flashed board version.

Train: nine actuator-walk recordings2–10. Validation:11–12. Held-out test:13–16. Truncated recording1 has1016rows and is excluded by the inventory before any model metric. Ten independently recorded wave runs and two320k regime runs are separate OOD groups and select no hyperparameters. No window crosses a recording boundary. A16-step horizon is an observation-count horizon with variable physical duration, not a fixed-wall-time simulator certificate.

Future programmed actuator/sensor commands are known and allowed by the recorded open-loop protocol. Actual future timestamps and measured sensor values are not deployable inputs. The actual-future-dt arm is a privileged conditional comparator, **not an oracle lower error bound**: finite misspecified predictors can perform worse with more information. Predicted/past-time arms share the same fitted conditional dynamics weights and use only permitted histories/commands. Standard HGB can predict intervals but is not proposed as an innovation.

Ridge uses two RPM lags, polynomial command features and train-only scaling; six ridge penalties are chosen by validation-recording increment error. HGB direct-next-state controls are more flexible, not parameter/compute matched; their default capacity is frozen before the follow-up fit and is not tuned on test/OOD. Original fitted estimator score arrays are exported for five HGB fits; ridge validation-alpha grids are retained. Seven selected estimators are retained across two model bundles; counting twelve ridge candidates and five HGB fits gives17fits, not17independent replications.

All scores are on **measured RPM** in source units, normalized by3000². Neither RPM nor pressure is privileged physical latent ground truth. Equal recording-weighted means are descriptive, without row-wise confidence intervals or cross-environment significance claims. Ridge roots include rare regime-boundary windows; the HGB follow-up excludes crossing roots. Do not interpret that small protocol difference as a model effect.

Main decision: coarse-timer sensor blocks dominate the apparent OOD error, while a concrete quantization explanation is supported by data and inspected source. This prevents promotion of the score drop into a new latent-dynamics failure. Clock-conditioning improvements alone also do not establish a new method need; the relevant literature predates this experiment.

The seven required ledgers and diagnostic figure are saved here. Artifact manifest is produced only after completion and source/data/PDF verification. All results remain private.
