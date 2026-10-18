# Implementation notes before training

The native gradient smoke passed with all12,968,068 parameters enabled; five batch16 steps used peak667,510,272 bytes. No model code was replaced. The first full-size native loss is finite, and optimizer updates change weights. This is a functioning-training check, not a convergence claim.

Original generated `drift_obs` arrays are the field at clean states. Native `_training_forward` appends input observation locations before its internal corruption stage. Our external common noise means its appended locations are noisy coordinates. The training loader therefore analytically evaluates the synthetic field at these actual coordinates (`drift_obs_native`), retaining original clean arrays. This prevents misaligned location/target supervision and changes neither evaluation information nor native architecture/loss.

Selection uses native normalized field l1 (`losses/l1`), rather than the uncertainty-weighted scalar loss, on the same unselected VAL cases for every arm. Native internal corruption is disabled after rebuilding native DataPreparation; external iid .01 noise is fixed/common. Original native normalized l1 and uncertainty weighting remain active during training.

The screening threshold was fixed before data generation. Observed acceptance is high in these bounded families. We will report this weak-selection regime, without changing thresholds to manufacture an effect. The experiment tests transfer of the mechanism claim beyond the earlier linear construction; a null result is informative.

Every500-step checkpoint retains weights/RNG/batch-RNG state and hashes, not optimizer state. These are evaluation/replay checkpoints; exact mid-fit optimizer resume is not claimed. Fresh reproduction follows fixed seeds and source. The original long-budget C71 campaign remains separate and live.
