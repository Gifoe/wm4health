# Controlled intervention-support shift on VitalDB

Fourth diagnostic of the `wm4health` sequence. Historical/current physiology and future dose schedules are clustered using TRAIN data. Four state–action combinations are withheld from prospective RSSM training while both marginal regimes remain observed. All outcomes are real factual held-out VitalDB trajectories. This is not a causal counterfactual experiment.

Requires the Round-1 aligned numeric case arrays, the Round-2 prospective-action code, and the Round-3 five-seed checkpoints and saved prediction arrays on the data host. The experiment makes no new VitalDB download. Python packages used by the recorded run are listed in `outputs/pip_freeze.txt`; the essential packages are PyTorch, NumPy, pandas, SciPy, scikit-learn, FAISS, Matplotlib, PyYAML and joblib.

Run from this experiment directory on the data host, with the three prior experiments adjacent:

```bash
python src/discover.py
python scripts/optimize_random_blocks.py
python scripts/apply_random_blocks.py
python src/describe_regimes.py
python src/control_balance.py
python src/train_conditions.py
python scripts/stamp_checkpoints.py
python src/predict_conditions.py
python src/support_diagnostics.py
python src/evaluate_conditions.py
python src/finalize.py
```

The first four commands use no outcome labels or predictions. They write the fixed support-cell manifest and exact-window/block/patient/marginal-matched random-removal control before any condition is trained. The later commands save checkpoints, predictions, patient-weighted statistics, plots and `outputs/FINAL_REPORT.md`. Full per-window arrays remain on the data host; compact tables, figures, checks and checkpoints are published in Git.

`bash scripts/run_pipeline.sh` runs the sequence in a clean copy. Reruns should use a separate checkout or clear this experiment's generated outputs, because the scripts intentionally preserve existing checkpoints when their support-manifest hashes match. `outputs/protocol_amendments.md` records construction-stage decisions made before viewing target TEST outcomes.
