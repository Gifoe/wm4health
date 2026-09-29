# VitalDB multi-timescale dynamics feasibility, seed 0

Three direct multi-horizon conditions share the same preprocessed VitalDB cohort, patient split, prospective future action sequence, four supervised horizons, forecast loss, optimizer, epoch permutations, validation protocol, and seed. The unchanged Direct-MH checkpoint is reused. Direct-MH-Capacity enlarges the single history state; MT-Dynamics splits a shared history encoding into fast and slow states. The fast state updates every 10 seconds; the slow state updates after each completed 60-second action block. No CE track enters training.

Run `python src/train.py`, `python src/evaluate.py`, `python src/probes.py`, then `python src/finalize.py` on the server from this directory. The source data arrays are reused in place. Large per-window caches stay on the server; compact results, plots, and model checkpoints are versioned.
