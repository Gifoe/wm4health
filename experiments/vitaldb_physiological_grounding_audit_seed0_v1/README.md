# VitalDB physiological grounding audit, seed 0

This experiment freezes the Round-2 prospective RSSM and audits current-state and transition representations. It does not train a new world model or use TCI CE as a forecasting input. The original patient split, preprocessing and complete-future-action windows are reused.

**Qualified Outcome A:** current BIS/MAP/HR are highly linearly decodable from the current latent (R² 0.986/0.924/0.983), but predicted 300-second latent displacement weakly recovers their changes (R² 0.374/0.083/0.356), with limited improvement from a small MLP. Predicted transition directions and patient-balanced response ordering are much weaker than factual-anchor representations. The gap does **not** grow with horizon or worsen specifically under intervention changes. The factual-anchor encoder has already observed the endpoint physiology, so this comparison is diagnostic and does not prove causal rollout drift. Read the [full report](outputs/FINAL_REPORT.md) and [decision criteria](outputs/decision_criteria.json).

Run on the machine holding the original case arrays:

```bash
python -u src/extract.py
python -u src/probes.py
python -u src/geometry.py
python -u src/finalize.py
```

Large per-window latent arrays are stored under `outputs/arrays/` on the compute server and excluded from Git. The committed checkpoint hash, cohort IDs, analysis configuration, trained probe parameters, metrics, plots and integrity checks make the analysis traceable.

The factual-anchor latent at `t+h` contains physiology observed by that time, including the target endpoint. It is a diagnostic representation, not a prospective forecast. Strong factual-anchor probing cannot alone show that the model has learned a causal physiological mechanism.
