# C54 result-figure contract

Destination: research-audit figures, suitable for a full-width scientific report. No manuscript or venue claim is implied.

Question: do retrospective state recovery, learned transition structure and causal conditional prediction agree, and does the existing native EM option change the latter?

Sources: completed outputs/evaluation54.json, outputs/em_option_learning_v1/evaluation/summary.json and outputs/causal_risk_interpretation_v1/summary.json. Rendering waits for the completed all16-model scoring and read-only risk analysis. Model and map selection remain unchanged. All16 runs, two tasks and six original horizons must be retained.

Panel map:

1. recovery_and_dynamics: two task rows, three metric columns; all individual initializations for retrospective smoothing R2, transition eigenvalue error and h10 conditional-mean error. No seed selected or hidden; no significance inferred from seed count.
2. causal_risk: two task rows, raw and predictable-energy-normalized conditional-mean discrepancy columns. Each of the eight fits per task has a line at every defined horizon. Exact zeros trigger a linear axis rather than a fabricated numerical floor. Undefined normalization is explicitly recorded and marked. Raw errors remain visible beside normalized errors.
3. paired_source_option: one panel per task, both exact initializations; native True-minus-False conditional-mean error with the already registered pointwise95 sequence-bootstrap intervals at all six horizons. Negative means lower error for True. These are fixed-model sequence intervals, not initialization/domain uncertainty or multiplicity-corrected significance.

State maps use training latent labels only as diagnostics. Smoothing is retrospective; rollout uses the first50 observations. No full-law probabilistic calibration claim is made. There is one data draw per task. Method colors: native False blue, existing native True orange, corrected EM black. Markers/line styles also identify initializations. Figure3 colors identify the two initialization pairs, with explicit labels. Editable text remains in SVG; PNG is a preview. No external image model or architecture rendering is used.

Traceability: exact input hashes, source fields and plotted values go into plotted_values.json; output hashes and plot source hash go into manifest.json. Input hashes are verified unchanged after rendering. Formal and dummy output directories are distinct, and dummy figures are visibly labeled. Only a dummy-layout qualification may run before scoring. Visual inspection remains required for the final actual-data renders; successful execution or geometry checks alone cannot certify it.

The plotting source does not fit, select, rank or modify any model. No accepted direction is generated from a figure. Canonical QA ledger: visual-composer/qa-ledger.md.
