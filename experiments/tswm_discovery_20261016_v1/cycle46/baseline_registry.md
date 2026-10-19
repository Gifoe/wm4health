# C46 baseline execution registry

Pinned NCDSSM core7212bba9616d81c948f507eef9eb1f7d5cf1954e, MIT. Original filter/auxiliary ELBO unchanged; context parameter adapters and proper joint generative sampling disclosed. Neural latent/aux4, observations8;64-hidden two-layer recognition/emission networks, continuous stable linear generators. O0D0/O1D0 nominal11948 parameters, O0D1/O1D1 nominal11620; O1 inactive-input difference disclosed.

| Fit | Updates | Selected | VAL ES | Seconds |
|---|---:|---:|---:|---:|
| o0d0_seed2026461 | 2000 | 1250 | 0.558316 | 187.5 |
| o0d0_seed2026462 | 3000 | 3000 | 0.560507 | 280.0 |
| o0d1_seed2026461 | 5250 | 4500 | 0.546781 | 488.9 |
| o0d1_seed2026462 | 4000 | 3250 | 0.538380 | 374.4 |
| o1d0_seed2026461 | 3250 | 2500 | 0.544425 | 307.8 |
| o1d0_seed2026462 | 2000 | 1250 | 0.566737 | 186.8 |
| o1d1_seed2026461 | 3250 | 3000 | 0.555403 | 307.5 |
| o1d1_seed2026462 | 4500 | 3750 | 0.539947 | 426.9 |

Ordinary aligned PCA4/LGSSM: TRAIN PCA,64 static anchor alignments, per-system VAR1 and per-view Gaussian observation maps. Ridge selected from five fixed values using96 seen-cell VAL histories; no EVAL adjustment. Not maximum-likelihood EM or parameter-matched neural comparison. Other controls: per-view TRAIN whole-path sampling, zero, persistence. RP-GSSM/RAMP/DiLA are prior-art boundaries here, not executed comparative reproductions.
