# Reproduce the C81 development screen

This directory is a new experiment on a released dynamics checkpoint. It is neither the FIM-ODE paper protocol nor a reproduction of JFR, WENDy, or the neural-ODE function encoder.

Prerequisites are the immutable C80 source/asset manifests, its isolated dependency prefix, the qualified base checkpoint and public ODEBench JSON, plus the project storage configuration. See the C80 source/asset/dependency audit for exact upstream revisions, file hashes and runtime versions. No benchmark trajectories or neural weights are stored in GitHub. Preserve the prior cycles and use a fresh output directory for a rerun: scripts refuse to overwrite closed artifacts. This is a server experiment package, not a portable one-command installer.

Execution order, using the qualified Python environment on the server:

```powershell
python -I -B scripts/cohort81_v1.py
python -I -B -u scripts/capture81_v1.py
python -I -B scripts/metadata81_v2.py
python -I -B -u scripts/verify81_v2.py
```

The capture process records the numerical worker's exit code and full stdout. `cohort81_v1.py` must close before any inference. `run81_v1.py` rebuilds encoding/statistics within each context split and never fits on the independent clean trajectory. `features81_v1.py` uses the original frozen model with a diagnostic hook at its final linear layer; an external correction is applied without modifying checkpoint parameters. `metadata81_v2.py` is a post-opening audit of equation identifiers and original successful-path indices; its whitespace parsing fix changes no fit or cohort.

`verify81_v2.py` recomputes stored metrics/identities and all selected readout fits using direct window quadrature and augmented least squares. Version 2 also handles a whole candidate family failing, consistently with the frozen protocol; the numerical experiment itself is version 1 throughout. Administrative export scripts do not select or refit models. `export81_v4.py` creates the report and immutable upload snapshot only after verification. Export-only preflight versions are not needed to reproduce the numerical result.

`PROTOCOL81_V1.md` and `COHORT81_V1.json` define the decision gate and which systems remain reserved. No reserved-evaluation scores are produced in this cycle. Context reconstruction, a new initial condition under the same equation, and general world-model transfer are different claims; only the first two are measured here.
