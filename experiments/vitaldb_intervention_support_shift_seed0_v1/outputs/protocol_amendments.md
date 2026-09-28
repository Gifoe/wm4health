# Construction-stage protocol amendments

These changes were made before examining any Round-4 TEST BIS outcome or prediction for the finally selected cells.

1. The first deterministic selection prototype greedily selected four eligible cells but repeated action cluster 5. TRAIN-only drug-regime descriptions showed that this choice overrepresented high-rate-tail schedules. The selection rule was changed to exhaustively choose four cells with distinct state and action clusters when feasible, then minimize TRAIN-only high-rate-tail prevalence and maximize TEST patient evaluability. The final cells are S11/A5, S3/A8, S5/A9 and S10/A4. The prototype's partial Low-support checkpoints were discarded, and all reported models were trained from scratch on the final support manifest.
2. The first random-removal prototype exactly matched the removed window count and state/action marginal counts but touched more independent five-minute blocks than Zero Support. A deterministic mixed-integer block-capacity step and a linear transportation allocation were added. The final Random control matches the number of windows, touched blocks, affected patients/cases and state/action marginal counts, while preserving every target-cell TRAIN window. Within-block removal fractions still differ and are reported in `random_control_balance.csv`.

The final selection criteria, random seeds, support manifest and code are saved in this experiment. No alternate cell set is included in the outcome analysis.
