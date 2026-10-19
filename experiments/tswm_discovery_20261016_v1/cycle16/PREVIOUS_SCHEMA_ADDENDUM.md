# Cycle14 descriptive data-schema correction

The sealed cycle14 `baseline_registry.md` described six historical/raw covariates. That count was wrong: it came from the config field `covariate_dim=6`, not the actual input tensor width. The immutable `outputs/native_chart/results.json` records exact channel order:

- NP: OT, Grid load forecast, Wind power forecast — two raw covariates.
- PJM: OT, System load forecast, Zonal COMED load foecast — two raw covariates (publisher spelling preserved).

Cycle14 code and saved tensors used the actual pivoted raw columns, and checkpoint/source/data hashes and metrics are unchanged. The original descriptive registry is preserved as sealed evidence; this addendum corrects its prose without modifying those artifacts. No meaning of the config field is guessed here.

Cycle16 uses six newly specified deterministic calendar conditions and one observed target. Those six features are distinct from the two raw forecast columns. Future raw forecast values are not provided to this new pilot because their issuance-time availability has not been verified.
