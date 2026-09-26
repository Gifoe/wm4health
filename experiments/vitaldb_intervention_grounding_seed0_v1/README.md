# VitalDB intervention grounding diagnostic — seed 0

Numeric-only, patient-disjoint first diagnostic experiment. The question is whether competent short-horizon BIS forecasting can coexist with weak learned drug-exposure/timing information. This is an audit of simple forecasting baselines, not a proposed final method.

Server root: `/root/experiments/vitaldb_intervention_grounding_seed0_v1/`.

## Reproduce

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install pandas scikit-learn vitaldb scipy requests pyyaml pyarrow matplotlib
.venv/bin/python src/prepare.py
.venv/bin/python src/rate_validation.py
.venv/bin/python src/provenance.py
.venv/bin/python scripts/run_pipeline.py
# Inspect all metrics, then explicitly write the evidence-based classification:
.venv/bin/python src/report.py --no-plots --outcome A_OR_B_OR_C --interpretation-file outputs/interpretation.md
```

The server has a working CUDA PyTorch installation; the venv inherits it. On a fresh machine, install a CUDA build of PyTorch suitable for its driver first. The exact recorded environment is in `outputs/pip_freeze.txt`. Configuration, exact cases and patient IDs, per-case preprocessing decisions, source metadata hashes, training logs, checkpoints and test predictions are preserved. Downloads are resumable. Only allowlisted numeric CSV tracks and metadata are acquired through the official VitalDB API; waveform/full-case container downloads are forbidden.

`src/prepare.py`: cohort, direct RATE preference, causal resampling, masks and per-case arrays. `src/core.py`: train-only normalization, dynamically sliced windows, event definitions and three models. `src/train.py`: fixed-budget forecasting training. `src/evaluate.py`: forecast metrics, frozen linear probes, timing shifts, dose removal, matched shuffle and scaling. `src/report.py`: exact tables and PNG/PDF figures. `src/checks.py`: CE/future-input isolation, patient separation, causal fill, shifts and physical-zero checks.

CE is device-computed TCI reference data, never a forecasting feature or loss. All models use history only. The compact deterministic RSSM holds the last observed action during future rollout. No claim of a causal treatment effect follows from this experiment.

Required results are under `outputs/`, figures under `plots/`; start with `outputs/FINAL_REPORT.md` after the run and evidence review finish. A missing final report or nonzero run status means the experiment has not been completed.
