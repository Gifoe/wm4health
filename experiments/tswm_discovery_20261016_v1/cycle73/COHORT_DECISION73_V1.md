# C73: prospective cohort eligibility

This source/metadata audit does not run a forecasting model or establish a new research gap. Nine pinned files were verified; five generator ASTs were inspected without importing or executing the author programs. No signal or image files were downloaded. The source revision and file hashes are in outputs/COHORT_ELIGIBILITY73_V1.json.

| Design | What the actual protocol changes | Decision for joint unknown-interface/mechanism shifts |
|---|---|---|
| Wind pressure control | One fixed hatch/control-target experiment; feedback determines fan loads | Does not supply a joint-shift sequence; future loads are endogenous |
| Wind changepoints | Intake-fan command changes while other settings stay fixed | Input-transient task, not evidence of changed physical laws |
| Smooth polarizers | Optical actuator sweep and three input-distribution blocks; fixed sensor parameters | Useful declared optical/interface condition, insufficient joint dynamics change |
| CRL Buchholz / CITRIS | Independent SCM samples / temporal latent actuator process with per-step interventions; fixed camera | Keep the static and temporal experiments separate; neither alone supplies the required joint-shift sequence |

These judgments concern the proposed contract, not the datasets' general value. In particular, CITRIS genuinely contains temporal dependencies; this audit does not call the entire CRL dataset static. If its nonintervened actuators are treated as future latent state, their future truth is not a valid known-driver input. A predeclared intervention plan has a different role. Metadata does not prove an arbitrary forecasting model will fail.

Historical C22 wind recordings and C69 scored physical-system records cannot be relabeled fresh. Hashes of the directly checked old protocol/registry are retained; this limited check does not certify every other possible record as unused. The four screened designs alone do not justify starting a joint-shift model comparison. A future constructed sensor change must be explicitly identified as a synthetic observation intervention on real trajectories, while a plant input intervention stays distinct from a changed transition law.

The next work remains independent data/contract selection and complete ordinary observer, nonlinear-alignment, state-space and replay controls. No distinct new learning mechanism is validated. C71's formal budget and assessment stay unchanged. No final direction accepted; overall research goal ACTIVE/UNACHIEVED.
