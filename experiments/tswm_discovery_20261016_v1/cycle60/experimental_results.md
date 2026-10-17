# C60 executed results

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

Neural updates completed in valid fits: 60000. The 45 synthetic qualification optimizer steps are separate. Model-seal -> first new test-scoring receipt -> evaluation ordering, original source hashes, selected/terminal checkpoints and every stored prediction hash were verified. Recomputed maximum primary discrepancy: 2.77556e-17 (relative checks also protect very large finite errors).

Native-component NSSM validation showed instances of short-sequence loss improving while full free rollout diverged. This motivated a parameter-comparable ordinary GRU control before new test scoring; the original NSSM fits were neither stopped nor retuned. Test outcomes do not select a new recipe. Full individual seeds, validation-selected steps and training runtime are in the CSV/JSON. The table reports a complete group mean only when every expected fit has finite outcomes; incomplete/nonfinite results are not silently averaged away.

Two separately fitted measured systems; the N87 split has historical C59 test exposure and remains exploratory. Pick-place contributes one chronological test record, not an independent-device population. All neural seeds are shown; record-level intervals are not seed intervals. No shared checkpoint, new causal identification, new architecture or field-wide unresolved gap is established.
