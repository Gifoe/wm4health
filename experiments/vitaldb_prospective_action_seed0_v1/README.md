# VitalDB prospective action diagnostic, seed 0

This experiment compares the same compact deterministic RSSM trained with a held last action during future rollout against the same RSSM trained with observed future administration at each rollout step. Future actions are the conditioning schedule; future physiology and TCI CE never enter forecasting inputs or loss. A third generic supervised control, if run, must use matched capacity. The source cohort, preprocessed case arrays, splits and normalization come from `../vitaldb_intervention_grounding_seed0_v1`.

Run on a CUDA machine from this directory after regenerating or restoring Round-1 `data/cases` arrays. Use a Python environment with PyTorch, NumPy, pandas, scikit-learn, PyYAML and Matplotlib; the exact first-run environment is recorded at `../vitaldb_intervention_grounding_seed0_v1/outputs/pip_freeze.txt`:

```bash
PYTHON=/path/to/python bash scripts/run_pipeline.sh
# Inspect all saved results and choose A, B, C or D from the stated decision rule.
python src/finalize.py --outcome C --interpretation-file outputs/interpretation.md
```

`data/cases` arrays are intentionally not duplicated in Git. If absent, regenerate Round 1 numeric data with its `src/prepare.py`, then run this experiment. The Round 1 large-change subset is retained as originally defined; it indicates a recently confirmed change. Future-action divergence is analyzed separately to identify prospective treatment variation. The high-rate sensitivity was added after observing the initial figures and is exploratory. `scripts/run_after_training.sh` resumes evaluation from saved checkpoints. Finalization checks the saved predictions against the metric ledgers and packages code, metrics, plots and checkpoints; large per-window prediction arrays remain on the compute server.

The example finalization command reproduces the published Outcome C. Change that argument and the interpretation text if a fresh run supports a different result. The published report discloses that most high-divergence benefit occurs near high pump-rate episodes.
