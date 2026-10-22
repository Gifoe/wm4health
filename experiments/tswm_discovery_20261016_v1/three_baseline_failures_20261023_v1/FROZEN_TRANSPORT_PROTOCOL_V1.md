# Frozen composition audit of the remaining discrepancy

The registered decoded-loss branch stopped architecture expansion on Validation. Test residuals remain7.21%/7.35% over raw, but are not grounds to reverse that branch or tune Test hyperparameters. The broad reconstruction/dynamics mismatch already has direct prior art. This follow-up diagnoses existing weights; it proposes and trains no method.

Freeze horizons336/720, seeds42/43/44, both original Val and Test origins atstep60, the same selected fixed100 raw/latent/decoded checkpoints, train-only scaler and released AE. No reselection, fine-tuning, calibration or new data. Every forecast is computed from history and available calendar marks only. Let `F` be the frozen raw DLinear, `E/D` the released encoder/decoder, and `R=D∘E`.

Evaluate these six fixed forecast paths, all before viewing their new scores:

1. `F(x)` — qualified raw replay.
2. `F(R(x))` — only reconstruct the observed history.
3. `R(F(x))` — only reconstruct the raw forecast.
4. `R(F(R(x)))` — reconstruct history and forecast; a fixed coordinate-transport control.
5. `D(F_latent(E(x)))` — selected native latent predictor.
6. `D(F_decoded(E(x)))` — selected official decoded-loss predictor.

The fourth path is a fixed composition, not a newly learned latent transition or a capacity-matched model. Reconstructing a raw forecast is not necessarily the optimal latent preimage. Consequently its error is **not a lower bound** on all AE-based predictors, and these comparisons do not uniquely isolate every causal factor.

Separately measure `R(y)` reconstruction using the true future **only as a labelled diagnostic**, never as a forecast or input to any of the six paths. Do not rank it among forecasts or call its error an irreducible forecasting floor. Record history reconstruction error separately.

For each nonraw path `P`, retain per-origin MSE/MAE and the exact decomposition `MSE(P)-MSE(F)=mean((P-F)^2)+2mean((F-y)(P-F))`. Signed cross terms distinguish amplification from cancellation; squared reconstruction error alone cannot explain a forecast deficit. Preserve every seed and both splits rather than selecting a favorable composition. Qualify all three learned paths against their existing Test predictions and prior deterministic Val scores, and verify all source/data/AE/checkpoint hashes before and after.

Store only bounded per-origin diagnostic arrays on the server data disk; no new checkpoint or full prediction payload is needed. Report all six paths regardless of ranking. No numerical outcome here constitutes independent method confirmation or a novel general world-model gap. Use the outcome to reject or narrow explanations; do not revive the untriggered iTransformer branch or develop a new architecture from this single benchmark.
