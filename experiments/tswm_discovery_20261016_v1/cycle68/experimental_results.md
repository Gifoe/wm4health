# Frozen controlled-system results

24 systems:2 families x6 VAL and6 independent EVAL systems. L3200/H100, aligned scalar output/control channels, deterministic exogenous controls. Unitless signals. Primary endpoint is MAE divided by observed-context sample SD, MC-averaged within system, then equal mean over systems/families. Raw RMSE is mean per-case RMSE, not pooled RMSE. No causal clinical interpretation, uncertainty calibration claim or significance claim.

## Validation

| Method | normalized MAE | linear | cubic | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
|bayes_indexed|0.60715437|0.62189595|0.5924128|0.39951954|0.47786612|
|bayes_same_view|0.62380471|0.66088591|0.58672351|0.40844307|0.4898758|
|clamped_simulate|0.6707191|0.6523407|0.6890975|0.4405399|0.5459223|
|context_mean|0.86728656|0.81739739|0.91717572|0.57011121|0.66840815|
|last_value|1.2015048|1.1903416|1.212668|0.81084396|0.94574937|
|latent_ridge_k16_a0.001|0.22771617|0.23023933|0.22519302|0.1528385|0.19187124|
|latent_ridge_k16_a0.1|0.22685445|0.22893148|0.22477741|0.1523205|0.19153604|
|latent_ridge_k16_a10|0.23316148|0.23726675|0.2290562|0.15603575|0.19710119|
|latent_ridge_k32_a0.001|0.23195503|0.24450587|0.21940418|0.15504146|0.19545116|
|latent_ridge_k32_a0.1|0.23149869|0.24419368|0.2188037|0.15470273|0.19490359|
|latent_ridge_k32_a10|0.23801518|0.24892998|0.22710038|0.15865236|0.2004866|
|latent_ridge_k4_a0.001|0.30409395|0.30582974|0.30235816|0.20254565|0.24385014|
|latent_ridge_k4_a0.1|0.30401038|0.30581005|0.3022107|0.20248562|0.24380728|
|latent_ridge_k4_a10|0.31044713|0.31118264|0.30971161|0.20652363|0.24807679|
|meta_id_author_ensemble|0.04020521|0.040165384|0.040245036|0.025625303|0.031893011|
|native_predict|0.78240613|0.80316551|0.76164674|0.51882509|0.64341595|
|raw_arx_d1_a0.0001|0.0077489605|7.7018409e-05|0.015420903|0.0055171266|0.006758889|
|raw_arx_d1_a0.1|0.05897628|0.066822554|0.051130006|0.038009214|0.044628161|
|raw_arx_d1_a1e-08|0.0077084345|7.7030375e-09|0.015416861|0.0054923201|0.0067302862|
|raw_arx_d3_a0.0001|6.4922652e-05|7.6937994e-05|5.2907311e-05|4.1632272e-05|4.88579e-05|
|raw_arx_d3_a0.1|0.057713895|0.0668972|0.048530591|0.037118927|0.04357668|
|raw_arx_d3_a1e-08|6.4930908e-09|7.6949769e-09|5.2912047e-09|4.1637413e-09|4.8863937e-09|

VAL-only selection: `{"latent": {"alpha": 0.1, "k": 16, "method": "latent_ridge_k16_a0.1"}, "raw": {"1": 1e-08, "3": 1e-08}}`. Receipt SHA256 `f397b95f3b0f618f529d071be708d18ceae9bf172b072479e53ee661594bc7d7`; written before EVAL scoring. No EVAL hyperparameter changes.

## Independent evaluation

| Method | normalized MAE | linear | cubic | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
|bayes_indexed|0.53755219|0.53172135|0.54338303|0.28806905|0.34618331|
|bayes_same_view|0.54499459|0.54026904|0.54972013|0.29306535|0.35482033|
|clamped_simulate|0.65068455|0.61791199|0.6834571|0.34349928|0.42824138|
|context_mean|0.79087858|0.74380007|0.83795709|0.41779381|0.50007053|
|last_value|1.0875703|1.1240042|1.0511364|0.57248325|0.66882924|
|latent_ridge_k16_a0.1|0.18545237|0.18353067|0.18737407|0.098288091|0.12794085|
|meta_id_author_ensemble|0.0317972|0.029730727|0.033863673|0.01703754|0.022008529|
|native_predict|0.82535763|0.79670712|0.85400814|0.44202684|0.53432346|
|raw_arx_d1_a1e-08|0.0062456988|6.1290329e-09|0.012491391|0.0032815275|0.0046122032|
|raw_arx_d3_a1e-08|6.1090332e-09|6.1320425e-09|6.086024e-09|3.005421e-09|3.6683085e-09|

## Paired descriptive contrasts

- latent_ridge_k16_a0.1 minus bayes_indexed: normalized-MAE delta -0.35209982, relative -65.5%; 12/12 systems lower.
- latent_ridge_k16_a0.1 minus bayes_same_view: normalized-MAE delta -0.35954222, relative -65.97%; 12/12 systems lower.
- meta_id_author_ensemble minus bayes_same_view: normalized-MAE delta -0.51319739, relative -94.17%; 12/12 systems lower.
- raw_arx_d3_a1e-08 minus meta_id_author_ensemble: normalized-MAE delta -0.031797194, relative -100%; 12/12 systems lower.

All per-system values and horizon halves are retained in result JSON. Native-context and selected-PCA reconstruction are descriptive diagnostics in FORECAST_LEDGER68_V1.json; low reconstruction error is not proof of predictive state sufficiency. Independent replay separately regenerates source equations, scores every retained row, refits ordinary controls, checks PCA with full SVD, redecodes operators and executes fixed fresh native calls.

