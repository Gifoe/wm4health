# Clean-directory reproduction entry point

`scripts/fresh_reproduce53_v2.py` stages an isolated copy and can execute the fixed native codec, 40-fit grid, scoring, independent arithmetic verification and descriptive analysis. It requires the pinned original TSWM source and both original Greenhouse crop batches. `required_data_manifest_v1.json` specifies the 26 public XLSX inputs used by the native loader; C52 records their selective acquisition and archive provenance. Optional energy files are not read by this input contract.

Use Python 3.10 with CUDA-enabled PyTorch and the original repository requirements. The executed environment uses Torch 2.8.0+cu128, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, einops 0.8.1, pyarrow 25.0.1, openpyxl 3.1.5, PyWavelets 1.8.0 and matplotlib. The loader and model probe check dependencies before fitting. No packages or drivers are installed by this runner.

First stage without executing training:

```text
python scripts/fresh_reproduce53_v2.py --run-root <new-run-directory> --storage-root <new-empty-storage-directory> --upstream <pinned-TSWM-source> --data-root <parent-containing-greenhouse3_original_v1>
```

Use the same arguments plus `--resume --execute` to run. `--resume` reuses validated terminal steps; partial fit folders stop for inspection instead of being overwritten. An existing live runner PID prevents a concurrent execution. The data directory is shared read-only except for native cache creation; every existing Parquet cache must equal a fresh native reconstruction of the pinned raw XLSX input before it is reused. Use a writable data copy when the original loader needs to create caches.

The fresh runner records generated source hashes and real qualification receipts. It removes only the two original run's historical initialization-comparison blocks from the instrumented V7 trainer. Exact AST equality of the remaining trainer is checked; model/data/optimizer/loss/selection code is preserved. The original source/data qualification and concat checkpoint probe receive path changes only. Historical crash or memory-repair receipts are never manufactured. The supervisor/selection-seal filenames are compatibility schemas and explicitly identify the fresh runner as their writer.

Validation limit: clean staging, generated-program syntax, the exact trainer AST transformation, all 26 input identities and all 12 existing raw-to-Parquet frame comparisons have been verified. The complete fresh 40-fit pipeline has not been rerun. This staging qualification is separate from the original C53 grid and does not supply additional training seeds or independent experimental evidence. Read `outputs/FRESH_ENTRY_PREFLIGHT_V1.json` for the checked scope.
