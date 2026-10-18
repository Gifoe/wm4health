# Baseline registry

Source: PRO-DYN 733d986d349e9e71ded5d9cdeeeed8d3323d83b4. Native unchanged models and optimizer, scheduler, validation, early stopping and block forecasting executed. The adapter is loaded through namespace packages to avoid unrelated eager imports; NumPy Inf alias restores a removed constant. The full TFB CLI/report layer was not executed. Official ETTh1 H96 hyperparameters were used.

Eight of nine planned fits completed, 21440 successful-fit optimizer updates, plus three aborted native attempts with at least3200 recorded updates (exact final count unavailable), 357.6 seconds of successful-fit training. Four fits use the unchanged native loader; four use a cached-tensor data-copy compatibility wrapper qualified with128 bitwise window comparisons, identical forward/gradients and180000 stress lookups. One per-window-copy diagnostic failed and is retained. Three training seeds, three fixed inference RNG streams. Native block-loop replay maximum raw-unit difference 2.064e-05; all 108 metric rows replayed, MSE difference at most 2.22e-15. All selected/terminal weight and prediction hashes verified.

Three ordinary controls: persistence, seasonal-24 and validation-selected direct ridge, implemented for this audit. No new WorldTS or LatentTSF fit in this cycle. No PHNN/C-PHAST execution.

ETTh1 is one previously examined measured system, with overlapping evaluation windows. Training seeds are initialization repeats, not independent systems. This is an exploratory screen. Inference RNG variation comes from native ProbAttention sampling; it is not learned predictive uncertainty. Extended point error alone does not establish incorrect physical dynamics. The source task makes no causal intervention guarantee.
