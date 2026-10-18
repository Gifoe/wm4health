# C64: frozen observational trajectory reuse under intervention

Status: CLOSED GAUSSIAN ATTRIBUTION. Overall research goal ACTIVE and UNACHIEVED. Accepted final research directions: 0.

This cycle executed train-only Gaussian MLEs and oracle sampler attribution on a fully observed feedback system. It is not a neural diffusion training run, real-data experiment, or full reproduction of TDS. The classical causal AR baseline is substantially stronger than direct block conditioning. A retained full joint Gaussian also supports normalized temporal causal factorization without retraining. No new framework is validated.

Read `experimental_results.md` for all fixed comparisons and `negative_evidence.md` for rejected interpretations. `final_research_direction.md` specifies the remaining provisional test and abandonment criteria. The seven requested reports, machine-readable outputs, immutable scripts and protocol are included. Raw arrays/checkpoints stay on the authorized server storage, with paths and SHA256 receipts.

Reproduce on a separately provisioned environment with Python 3.10, NumPy, SciPy and CPU PyTorch: set the project storage registry; fetch the three pinned upstream utility files listed in `outputs/SOURCES64_V1.json` and verify hashes; run `qualify64_v1.py`, then `fit64_v2.py`, `samplers64_v1.py`, `causalize_joint64_v1.py`, `replay64_v3.py`, `iid_reference64_v1.py`, `close64_v1.py`. Outputs are deliberately exclusive-create. Use a fresh output directory/storage registry for an independent run. The original upstream files are not redistributed because a root license was not located. Their receipt includes pinned links and checksums.

Operational version history: `fit64_v1.py` was not executed; its normal-equation helper is unsuitable at H1. Actual fitting uses SVD in v2. Replay v1 failed parsing before execution; replay v2 stopped at the exactly rank-deficient H1 QR check and produced no scientific output. v3 fixes the QR rank cutoff at 1e-12, compares predictions rather than nonidentified coefficients, and passes. No data, learned model, query, sampler budget, precision floor or scientific metric changed in these repairs.
