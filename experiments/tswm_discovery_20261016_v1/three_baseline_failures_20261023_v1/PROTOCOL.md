# Three-paper failure campaign — 2026-10-23

User steering: select three closely related strong papers; execute their real implementations; inspect failed reproduction cells, adverse workloads and unstable assumptions; formulate research questions from those observations. The scientific objective remains a reusable time-series world-model contribution. Do not replace it with a particular dataset's score optimization.

## Primary papers

1. **WorldTS** — two-stage covariate-conditioned latent dynamics. Paper: <https://arxiv.org/abs/2609.31162>. Code: <https://github.com/decisionintelligence/WorldTS>. Existing execution pin `269426ea6217d7152a244da0e209d922acd46d4f`.
2. **LatentTSF** — frozen autoencoder and latent forecast/alignment objectives. ICML 2026. Paper: <https://arxiv.org/abs/2602.00297>. Code: <https://github.com/Muyiiiii/LatentTSF>. Existing execution pin `7c8ae947ee1220bf4e788ace6bc2f0f122cb26c2`.
3. **TSWM** — the benchmark/recipe in *On the Divergence of Accuracy and Mechanism Consistency in Time Series World Models*. Paper: <https://arxiv.org/abs/2610.01842>. Code: <https://github.com/HaochenZhang717/TSWM-Official-Repo>. Existing execution pin `2e5e48540ff0da48e8192163969a6c58df1d5a3e`.

They are related through latent state prediction and its claimed utility. TSWM is a benchmark and recipe, not a unique architecture. No independently established universal SOTA ranking is asserted. Additional inexpensive ordinary controls are allowed when needed to distinguish a mechanism, without expanding the primary-paper set.

## Reuse before rerunning

Inspect actual source/data/config/checkpoint identities and saved predictions. The previous baseline ledger contains four WorldTS runs and sixteen LatentTSF runs; C53 contains forty source-based TSWM subset fits. Their scopes differ. The original 24-cell ledger also includes DAG, which remains a historical control rather than a fourth primary paper.

Do not label source-component transfer, new tasks or all historical execution as original-paper headline reproduction. Keep input information and forecast semantics in separate comparison panels. In particular, future action/covariate access for WorldTS/TSWM cannot be compared unqualified to a history-only LatentTSF run. A forecast sensitivity to a future covariate is not automatically a causal intervention response.

## First unresolved failure: source-native LatentTSF discrepancy

The existing Electricity/DLinear cells have source-matched settings but exceed the previous 5% numerical-match band: H336 paper MSE 0.185 versus saved 0.19671606163567038; H720 0.218 versus 0.23907836392582896. These are observed replication discrepancies, not proof of a new world-model deficiency.

Before viewing any new result, freeze:

- Two horizons: 336 and 720. Three seeds: 42, 43, 44. Re-run all six in a new directory, retaining the old seed42 records read-only.
- Unchanged official AE checkpoint, source data bytes, 720 history, source `step=60`, batch32, learning rate0.001, max100 epochs, patience5, cosine schedule, latent prediction/alignment weights10/15, no perceptual/reconstruction objective, frozen encoder/decoder.
- Use the already qualified audit/resume driver, changing only seed and output/patch destinations. Retain and inspect the exact driver diff; verify upstream/data/AE hashes before and after. Do not restart from the old learned forecast weights.
- Keep native shuffled/unweighted-batch validation selection as supplied. Do not select a favorable seed, extend training, alter losses or tune on Test to close the paper gap.
- Report every run, its native validation-selected epoch, observed training curve, float64 recomputed MSE/MAE and all execution errors. A reproduced gap is a diagnostic fact; the author's unprovided run artifacts may prevent unique attribution.

This confirmation uses an already inspected benchmark and is not fresh research generalization. If seed42 does not replay, first diagnose environment/instrumentation, not the model. Seed variation may explain a gap but three initializations do not establish broad statistical robustness.

## More realistic inference workload, predeclared

The author scripts explicitly use `step=60` for Electricity. After all six native choices are fixed, evaluate their same weights at every eligible hourly origin (`step=1`) with unchanged raw chronological split and train-only normalization. This is a changed inference workload, not another paper-reproduction score or a new training protocol. Stream predictions and accumulate per-origin/per-channel errors on server storage; never load the full dense forecast tensor into local memory.

Report dense scores, all sixty origin phases relative to the split start, and the original sparse-origin subset. Use the frozen original models; no phase-specific model or hyperparameter selection. Phase reweighting can expose a fragile evaluation protocol, but alone does not establish a new latent-state learning mechanism. Do not retrofit a repair after observing these Test slices. A method hypothesis needs subsequent independent development and confirmation data.

## Failure attribution and selection of a research problem

For each retained case distinguish: (a) implementation/runtime failure; (b) source/protocol mismatch; (c) training/selection uncertainty; (d) scientific capability failure under an explicit information contract. Maintain competing explanations and one experiment that can eliminate each. A failure surviving ordinary controls under matched information and practical budgets can motivate method development. A privileged low-dimensional toy repair does not prove every broader framework unnecessary.

Previously closed WorldTS Loop/readout/state-selection/NESO configurations remain negative evidence; do not silently reopen them. C71 is closed out as an existing commitment using its unchanged frozen evaluation, not added as a fourth primary paper. Its physics subset cannot stand in for cross-domain evidence for the three-paper campaign.

## Storage, source, and stopping

Work executes on the single RTX5090. Data, weights, predictions and complete logs remain under the existing authorized large-disk storage roots. Only authored code/configs, bounded metrics and reports are published to `Gifoe/wm4health`; no result/data/model payload synchronization to the local workspace.

Do not infer a gap from nonfinite code, unfair future information, nonconverged training or a deliberately invalid benchmark. Do not require a completely unprecedented mathematical primitive: a new interaction can qualify if it has a distinct mechanism, strong comparable controls and independent reproducible utility. Stop a specific claim when its proposed cause is falsified; preserve the broader research objective.
