# Frozen-gradient replay without retraining

Two tiny typed JSON parameter cards are included under `outputs/training_audit_replay_v1/`. They contain the exact arrays and dtypes of the two synthetic audit checkpoints. Array-by-array JSON roundtrip equality was checked. Original runtime checkpoint files, all signal files and the large score caches remain on server storage.

Set up the pinned native source, CPU dependencies, project layout and storage configuration described in `REPRODUCE.md`, and generate the deterministic data with `prepare54.py`. The cards record the original dataset hash. To replay the frozen gradient values, run:

```
python -B -u scripts/replay_audit_point54_v1.py --seed 0
```

This compares the native all200 and25 balanced batch32 A-scaling derivatives with the published values, for both False and True posterior-gradient options. It performs no optimizer step, latent-label read or heldout read. The original source and configuration hashes remain required. It is a replay of native calculations from frozen parameters, not an independent inference implementation or a fresh model-training reproduction.

The one actually executed replay and its numerical errors are recorded in `outputs/training_audit_replay_v1/replay_seed0_verification.json`. Both parameter cards have exact serialization checks; do not infer that both CLI replay seeds were rerun unless their separate verification receipts exist.

The four additional learning controls use `EM_OPTION_LEARNING54_V1.md`. The active automatic scoring pipeline is now `postprocess54_v3.py`: wait for the primary12 and supplemental4 model seals; prepare the4 extra training-only maps; fit the12 original maps and score the primary models; score the4 extra models; generate the joint reports. This preserves all16 models and maps before the first heldout comparison. Completed joint reports will point to `outputs/REPORT_MANIFEST54_V2.json`; their existence is required before claiming the full comparison complete.
