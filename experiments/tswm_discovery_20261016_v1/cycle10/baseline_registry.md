# Cycle10 baseline and provenance registry

| component | role | protocol/status |
|---|---|---|
| thummd/dotime at8f227406a5b17c2040cfdcc680b493729ef47d00 | unmodified simulator classes |15 controlled linear paired runs; not released-suite replication |
| source-free linear recurrence/spectral covariance | independent contract |15 paths plus5 stationary/spectral checks; exact rational identities retained |
| SciPy least_squares | existing covariance matching |120 local fits, diagonal-lag model only; no global certificate |
| ordinary OLS + Student-t intervals | existing randomized experiment estimator |1200 trials; extra intervention data;40 unique seed blocks per world/N |

All code, fixed grids, seeds and raw numerical outputs are retained. `sources/dotime/SOURCE_PIN.json` hashes every relevant retained native source and verifies the actual imported dependency against it. The whole upstream native package was already retained and verified in sealed cycle9. No modification to prior experiments, native dependency or shared environment. Main runtime uses Python3.10.20 and Torch2.8+cu128 with the recorded runtime profile. No unexecuted paper method is represented as reproduced.
