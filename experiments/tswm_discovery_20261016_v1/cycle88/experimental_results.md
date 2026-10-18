# C88 developmental results

| Model | Parameters | Duffing MSE | Reservoir MSE |
|---|---:|---:|---:|
| stock | 126932 | 0.24129893 | 0.080687342 |
| lagged | 126932 | 0.2196222 | 0.082494544 |
| acrkn | 66272 | 0.39343813 | 0.25755426 |
| gru | 128864 | 0.45503266 | 0.029322788 |
| ordinary ARX4 | 1168 | 0.77514475 | 0.21605579 |

See DECISION_RECONCILIATION88_V1.md: the registered timing-remedy candidate is retired, while the stronger supplementary response check fails.

One optimizer seed,512 TRAIN trajectories and64 developmental VAL trajectories from new parameter systems. These are controlled cross-task experiments using pinned native source, not paper reproductions. Checkpoints selected by factual VAL MSE only; edited plans never participate in selection. Each neural arm receives1000 updates/Adam. Observation prefix65 and future32. Sensor coordinates standardized using TRAIN only. No independent EVAL pool is opened.

Maximum early-prefix change across two families, suffix starts80/88 and signs+/-: {"stock": 0.28130245208740234, "lagged": 0.0, "acrkn": 0.0, "gru": 0.0}. Maximum early RMS: {"stock": 0.04824123355297287, "lagged": 0.0, "acrkn": 0.0, "gru": 0.0}. Values are standardized sensor units. A zero prefix change alone does not establish correct future dynamics; full paired response errors and edited forecast MSE are retained in outputs/PAIRED_PLAN_RESULTS88_V1.json. ARX4 results are outputs/RIDGE_PLAN_RESULTS88_V1.json.

Original ordinary-fix factual/causality rejection gate: True. Additional selected-vs-mean/untrained competence check: True. Lagged response-error<=1.05 stock in every paired cell: False. These are developmental observations, not independent-test significance claims.

Decision: **UNRESOLVED_DEVELOPMENTAL_COMPARISON_NO_NOVELTY_ESTABLISHED**. Accepted novel research directions remain0; full user goal remains ACTIVE/UNACHIEVED.
