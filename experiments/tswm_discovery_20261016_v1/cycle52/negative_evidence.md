# C52 real-data input-contract audit

Decision: SOURCE_BOUNDARY_RISK_NOT_ACTIVE_UNDER_NATIVE_DEFAULT_WINDOWS.
Accepted final research directions: 0. Overall discovery remains active and incomplete.

The exact original third Autonomous Greenhouse Challenge data (DOI [10.4121/21960932.v1](https://doi.org/10.4121/21960932.v1), CC BY 4.0) were selectively acquired from the original archive byte ranges. Two full archives total 45,381,396,051 bytes; their directories required 4,983,156 transferred bytes. Time-series-only acquisition transferred 31,567,900 bytes and extracted 40 XLSX/DOCX files totaling 32,517,760 bytes. No image members were downloaded. Selected-member ZIP CRC32, HTTPS archive ETags, local SHA-256 and the copies on the large disk were checked; the full original archive MD5 values are declared source metadata, not independently verified full-download hashes.

The pinned unchanged TSWM loader and windowing source were run for both public experiment batches separately: context 256, horizon 16, stride 80, native NaN limits .1/6, split seed 0, held-out team ratio .2. Twelve team-batch groups produced 1,781 accepted windows.

| Public experiment | Split | Accepted windows | Histories using future targets |
|---|---|---:|---:|
| 1 | train | 815 | 0 |
| 1 | val | 163 | 0 |
| 2 | train | 685 | 0 |
| 2 | val | 118 | 0 |

Every accepted window has finite target and exogenous values at the final history index. Native full-window interpolation and history-only interpolation produce identical normalized histories; changing only future targets also changes no real accepted history. Independent retained-mask and row checks agree. A constructed valid positive control still changes normalized history by 0.666666627, confirming that the audit detects the source defect when its boundary condition exists; a finite-boundary control changes by zero.

The previous constructed-window result remains valid, but it does not explain behavior on these default real-data windows. Do not propose a missingness repair or claim reported Greenhouse performance was inflated from that fixture. No model training or predictive comparison was performed in C52.

The two public experiment archives must not be merged or treated as independent real domains. The paper repository does not uniquely identify which archive generated its reported Greenhouse table; this source-version ambiguity remains. Accepted windows overlap and are not independent statistical replicates. Exogenous future values may be declared model inputs, so exogenous-history dependence would require a separate information-availability interpretation.

Evidence: [machine-readable audit](outputs/native_audit_complete_v3/audit.json), [window rows](outputs/native_audit_complete_v3/window_audit.csv), [verification](outputs/verification_v1.json), [acquisition](outputs/acquisition_v2/receipt.json), [protocol](PROTOCOL.md).

Storage: user-authorized large-disk directory recorded in `outputs/storage_v1.json`; raw data and retained arrays stay on the server. No result synchronization to the local workstation after the user's instruction. Code, configuration, aggregate results and provenance are prepared for Gifoe/wm4health.

Rejected explanation: the constructed interpolation defect is active on the native default accepted Greenhouse windows. Observation count 0/1781. This says nothing about other strides, datasets, future missingness patterns or downstream models.
