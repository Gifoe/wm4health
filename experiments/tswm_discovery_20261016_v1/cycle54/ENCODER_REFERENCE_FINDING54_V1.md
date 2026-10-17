# Encoder-reference attribution at frozen native parameter points

Training-only follow-up to TRAINING_OBJECTIVE_FINDING54_V1.md. No optimizer updates, latent-state labels, heldout observations or model selection were used.

| Initialization | Contract | A relative gradient difference | Recognition relative difference | Recognition cosine |
|---|---|---:|---:|---:|
|rp_linear_small_seed0|em=False|1.36993|0.90085|0.870396|
|rp_linear_small_seed0|em=True|3.27606e-14|0.414638|0.994119|
|rp_linear_small_seed1|em=False|1.13814|0.936518|0.776721|
|rp_linear_small_seed1|em=True|4.00084e-14|0.421537|0.995317|

Each relative difference is ||mean balanced B32 gradient - full N200 gradient|| / ||full N200 gradient||, at unchanged parameters. The25 balanced batches use each training sequence exactly four times. This is a finite-data reference comparison, not a population-gradient estimate or an independent learning replicate.

The existing EM contract removes the A-block discrepancy at these points. Recognition updates remain reference dependent, but their True-contract cosine is approximately .994-.995: the observed difference is predominantly magnitude, not an encoder-gradient reversal. Adam can respond differently to changes in gradient magnitude than raw SGD; these frozen values establish neither training damage nor a remedy.

The full native False loss and every gradient leaf exactly match the attributed per-target implementation. The True recognition gradient matches an explicit held-posterior M-step to about1e-16. These controls qualify the derivative comparison; they do not validate the approximate RPM likelihood or forecast performance. Complete per-leaf errors and unchanged checkpoint/source hashes are retained in the raw JSON.

Nondecomposable contrastive-gradient dependence is established prior art; see LITERATURE_REFERENCE_BOUNDARY54_V1.md. This audit supports a narrower parameter-block attribution, not a new method or a field-wide missing-capability claim. The matched four-fit native True learning control remains pending; no extra model/optimizer/configuration is proposed.

Raw summary SHA256: 180f37c3338e7296642fffa93ee4b5d0cd9681f8c58fbbd4bfea20b63fba5a28
