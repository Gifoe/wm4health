# C83: local-field composition capability audit

Original controlled development experiment. Read COMPOSITION_AUDIT83_V1.md for actual verified outcomes and their limits. No accepted final research direction is implied. PROTOCOL83_V1.md was frozen before data generation; the primary metric, strong source-informed control, noisy/clean pairing, solver guards and development gate were specified in advance.

Reproduce on the existing isolated FIM runtime from the project root, using a fresh versioned output location when outputs already exist:

```powershell
& .\.venv\Scripts\python.exe -I -B -u .\cycle83\scripts\data83_v1.py
& .\.venv\Scripts\python.exe -I -B -u .\cycle83\scripts\capture83_v1.py
& .\.venv\Scripts\python.exe -I -B -u .\cycle83\scripts\verify83_v2.py
& .\.venv\Scripts\python.exe -I -B -u .\cycle83\scripts\export83_v1.py
```

Dependencies/source/assets are the pinned C80 runtime, SOURCE_MANIFEST80_V1.json, ASSET_MANIFEST80_V1.json and DEPENDENCY_INSTALL80_V2.json. No global package changes are required. All raw generated data, local ordinary-model coefficients and prediction arrays reside in the configured15TB storage paths; GitHub contains code, provenance, compact results and figure. No local payload sync. The queued verification/export script waits on the actual numerical process and never opens a reserved dataset or changes model selection.

SOURCE_AUDIT83_V1.json records public primary source inspection only. CHOP/ICON/GICON code availability is verified; those models were not run on this carrier. Full-state known-parent synthetic experiments are not hidden-state identification, real causal discovery or a reproduced original-paper benchmark. Earlier experiments remain intact.

## Final attribution supplement

Reproduce with verify83_v2.py, attribute83_v2.py and finalize83_v4.py; see NUMERICAL_VERIFICATION83_V1.md for retained failed-check provenance. Read ATTRIBUTION_AUDIT83_V1.md for independently checked path-level diagnostics and the research decision. Additional source boundaries: PRETRAINING_PRIOR_BOUNDARY83_V1.md, MODULAR_INFERENCE_BOUNDARY83_V1.md, METAILAD_BOUNDARY83_V1.md. The original publication_v1 snapshot is retained on the server. Numerical models and results are unchanged.
