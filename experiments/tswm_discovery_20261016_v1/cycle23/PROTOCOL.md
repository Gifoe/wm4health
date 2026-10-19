# Cycle 23: qualify the system-identity contract before belief-update research

Status: prospective source-derived audit; no measured new outcomes at writing.
This is an audit of a released generator, not an accusation that its observational forecasting results are invalid. Whole-sequence normalization was already disclosed in cycle17. Here we isolate the distinct normalization *inside* the nonlinear system and test whether turning off the final normalization repairs it.

Research motivation: a reusable world model may need to preserve the identity of one physical system across different future inputs and successive information updates. A generator whose system changes with the complete future input is unsuitable as ground truth for that added capability. This is a protocol prerequisite, not itself a new model or publishable research gap.

## Fixed experiment

- Unchanged `PuraFilippo/sysid_benchmarks` source revision `4c20d94ccab51abf4ebf68aa8810f4596381dd41`, cached in cycle16. Source manifest must pass before and after. Do not retrain or modify its released checkpoint.
- 128 fresh random Wiener-Hammerstein systems, nx<=5, native pole ranges [0.5,0.97] and phase [0,pi/2], native burn-in 200 and retained length930. System seed20262301; original input20262302; alternative input20262303; independent calibration20262304; additive noise20262305. System generator draws and order match native source.
- For each system keep its parameters, complete burn-in, input prefix, and additive noise realization fixed. Independently replace future white-noise suffixes after retained index400 or744. Alternatives have exactly the same iid N(0,1) law as the originals; neither amplitude extrapolation nor adaptive selection.
- Four deterministic simulation pipelines before common additive noise: (1) raw causal WH (no normalizers), (2) native internal pre-nonlinearity full-trajectory normalization only, equivalent to native `normalize=False`, (3) complete native pipeline with separately recomputed final input/output normalization, (4) fixed-per-system normalization calibrated on a separate 930+200 white-noise run and reused for all input branches. Calibration must not use audit branch outputs. Raw causal arm has different scale/operating point and is a **contract control**, not a forecasting quality competitor.
- For each cut and arm record past output RMS change divided by original output standard deviation, past input RMS change, exact max differences, fraction of systems exceeding relative1e-8. Use all128, no filtering of difficult systems. Prefix should be identical in a causal fixed-system simulator. Strict proper G1 implies output at the first changed input index is also unchanged for causal arms; main metric uses indices strictly before the cut.
- Eight fresh independent seeds are separately replayed through unchanged native `WHDataset` (both normalize flags) versus the decomposed implementation; require float32 arrays exactly equal before accepting audit aggregates. Native normalization moments, nonlinearities, and source default noise preserved.
- Save system parameters, every original/branched input/output, normalizers, common noise, and per-system statistics on server. Independent verifier uses its own NumPy state recursion/nonlinearity and checks representative systems across both cuts, exact raw prefixes, saved hashes, and all aggregate metrics.
- No models are selected using these data. No marginal calibration/covariance experiment from cycle17 is repeated. No scientific worker is restarted based only on a transport timeout.

## Competing explanations and action

H1 final coordinate normalization only: internal-only arm should retain zero past effect. H2 system itself depends on full input via internal normalization: internal-only arm changes past outputs, fixed-per-system control restores prefixes. H3 implementation mismatch: unchanged native replay fails; then audit has no scientific interpretation until repaired in a new attempt. Report actual results, including near-zero exceptions.

If H2 holds, do not run a causal/active-information experiment on the released generator unchanged. A newly trained causal generator is a different data protocol, must have new train/validation/test systems, and cannot be called reproduction of the published benchmark. A normalization repair is ordinary methodology, not the proposed research contribution.

## Innovation boundary / next gate

Belief-update tower/martingale coherence is already studied by Martingale Posterior Neural Processes (ICLR2023) and From Drift to Coherence (ICML2026). Generic context-conditioned system identification is also old, including this native model and Meta-iLaD. A subsequent candidate must identify a specifically controlled sequential-dynamics requirement and decision consequence beyond those works, with a competent coherent positive control; absence of model code for a recent paper is not evidence of an unresolved scientific gap. There are currently zero accepted final directions.
