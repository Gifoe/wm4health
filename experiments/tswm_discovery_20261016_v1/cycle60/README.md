# C60: ordinary latent-state and GRU controls on two measured systems

| System | Control | Completed / expected | Finite rollouts | Mean test MSE / fit variance | Neural seed SD |
|---|---|---:|---:|---:|---:|
|N87|native_component_nssm|3/3|3|0.0101758|0.00598455|
|N87|ordinary_gru|3/3|3|0.0419812|0.0182712|
|N87|static|1/1|1|0.118216|N/A|
|N87|play|1/1|1|0.0516982|N/A|
|N87|ema|1/1|1|0.0633208|N/A|
|N87|hybrid|1/1|1|0.0478821|N/A|
|N87|arx|1/1|1|0.0584894|N/A|
|pick_place|native_component_nssm|3/3|3|0.00975243|0.00121676|
|pick_place|ordinary_gru|3/3|3|0.0346664|0.00908307|
|pick_place|static|1/1|1|0.396165|N/A|
|pick_place|play|1/1|1|0.646618|N/A|
|pick_place|ema|1/1|1|0.279222|N/A|
|pick_place|hybrid|1/1|1|0.512366|N/A|
|pick_place|arx|1/1|1|0.168394|N/A|

This completes the previously missing generic recurrent controls. Native-component NSSM and ordinary GRU use identical per-seed window-sampler code, 5000 Adam updates, observed history100, free future128 and full validation selection. They have 6312 and 6241 parameters respectively; state dimension, update law, device and runtime differ. This is not a pure stability-only ablation or equal-compute claim. The GRU addition was triggered by validation-only NSSM divergence before the first C60 neural test score; the amendment and superseded unopened waiter are preserved.

All selected model hashes were sealed before new scoring. Independent re-scoring of stored arrays gives maximum absolute discrepancy 2.77556e-17. See all seed rows in outputs/results60_v1.csv and outputs/EVALUATION60_V2.json.

Two separately fitted measured systems; the N87 split has historical C59 test exposure and remains exploratory. Pick-place contributes one chronological test record, not an independent-device population. All neural seeds are shown; record-level intervals are not seed intervals. No shared checkpoint, new causal identification, new architecture or field-wide unresolved gap is established.
