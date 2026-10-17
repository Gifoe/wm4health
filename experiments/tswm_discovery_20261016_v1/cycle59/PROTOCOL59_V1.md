# C59: Real measurement provenance and mature-control qualification

Date: 2026-10-18. Overall goal ACTIVE / UNACHIEVED. Accepted research directions: 0.

## Question and scope

Can the official MagNetX tutorial sample support a defensible conditional-response audit, with an identified clock, complete measurement rows, preserved history, and competent existing dynamical controls? This is a data/model qualification and exploratory baseline study, not a new architecture or a reproduction of the 14-material PI-HNO experiment. One tutorial material cannot establish cross-system world-model generalization.

The official Challenge 2 contract already supplies past B,H, future B and temperature for future H. Universal time steps, initialization and waveform handling are existing goals. A proposed framework must eventually improve a concrete capability or efficiency over these and other strong world-model controls, not rename them.

## Frozen source and storage

Repository and exact commit/blob identities are in source_selection59_v1.json. Download only official tutorial-2 Python/text sources and three small N87_5 CSV/gzip arrays. Verify byte count and Git blob SHA1 on the server, then save SHA256 provenance. Large arrays stay in the configured G: research data subtree; source and compact evidence stay in cycle59. No local result-file synchronization, no package/driver changes, no changes to older experiments.

## Audit before model fitting

Record shape, temperature distribution, finite values, raw row identity and duplicate hashes. Sampling interval must come from pinned run_1_dataset.py (62.5 ns), not magnitude or filename inference. Inspect the source preprocessing for periodic wrapping, whole-record response centering, target-based phase alignment and row preservation. A tutorial parser may legitimately demonstrate periodic steady response; it is not automatically a valid causal boundary-conditioned deployment protocol.

Preserve unmodified raw arrays. Any later causal experiment must not use unseen future H to center, shift, normalize, initialize or select histories. Raw complete rows must remain disjoint across training/validation/test; do not randomly split neighboring windows. Filename N87_5 does not by itself prove exact frequency, device identity, reset history or release equivalence to PI-HNO.

## Existing-control qualification

Use the actual official equation functions and their published constants/bounds where available. Check installed runtime dependencies read-only before selecting an execution path. Do not silently install packages or replace the original training pipeline and call it a reproduction. A core-equation extraction or solver port, if necessary, must be named as such and checked against the unchanged source and an analytic linear case.

The ordinary control family to consider includes linear/nonlinear state-space, context/history-conditioned neural recurrence, and standard Preisach/play plus relaxation. Their availability or implementation difficulty is not evidence of a gap. Do not use raw Gaussian HMM or instantaneous regression as the sole strong comparator on hysteresis.

## Stop and continuation rules

If row identity or physical time cannot be established, do not invent it. If the small sample only supports one material/operating regime, use it for execution qualification and retain the need for independent systems/complete-run data. If a mature control solves the tested capability, stop that necessity claim. A residual error alone must first be checked against initialization, preprocessing, input information, capacity and optimization.

No new-method claim, fixed expected gain, test-tuned hyperparameter or accepted final direction is authorized by this qualification. The next experiment will depend on the observed audit, not on a predetermined positive story.

## Primary sources

- https://github.com/minjiechen/magnetchallenge-2
- https://github.com/minjiechen/magnetchallenge-2/tree/ab4913ca63ebda1f8eab429c64d3e93f96204713/tutorials/tutorial-2
- https://www.princeton.edu/~minjie/magnet.html
- https://arxiv.org/html/2608.02965v1
- https://collaborate.princeton.edu/en/publications/unified-time-domain-foundation-models-for-hysteretic-passive-comp/
