# Baseline and protocol registry

| Object | Pin / data / execution | Classification |
|---|---|---|
| CHiLD source | MinghaoFu/CHiLD,28cc8a8a702018c9c54bd02e26cbd8ccf756ab99;110 small source/config files,474898 bytes | Private pinned source, not the real-world dataset zip |
| Native synthetic A | Native stationary_z_link,seed772,z=[1,2,4],100000 seven-step trajectories; X=(100000,7,4),Y=(100000,7,7) | Original released generator, source issues disclosed |
| Full CHiLD model | A.yaml architecture/loss/AdamW;train98976,val1024,seed770,B64,10 full epochs,15470 updates; no batch cap | CLOSE source/config execution; manual Torch orchestration/container adapter, different data-loader RNG/environment; not exact paper-table replication |
| Real-world CHiLD components | Three seeds81/82/83; unchanged model/prior classes, native finite forward/backward/Adam step and eval ablation | Component audit only, no full real-data reproduction |
| Conditional-density experiment | Native 33537-param inverse MLP vs ordinary33538-param normalized Gaussian, same hidden128/depth3;3 seeds;2000 updates each | New controlled component experiment, not CHiLD latent-recovery reproduction |
| Independent-noise A | Same seed772,shape/graph/mixing; only hierarchy-global innovation indexing repaired | Explicitly modified benchmark control; not original data and not a certificate of all theorem assumptions |
| Linear recoverability | Train-only supervised OLS to true factors, context widths1/3/5;native fixed train/val split | Signal-recoverability certificate, not an input/supervision-matched unsupervised baseline |
| Piecewise conditional integration | Full-real-line affine-region enumeration of scalar inverse MLP at fixed parents; analytic identity/abs/shifted-abs checks | Numerical scalar diagnostics, not a global joint-density certificate |

The real-world transition priors sit in an ordinary outer Python list. In this
path none of40 prior tensors belongs to the model optimizer parameter set. The
synthetic implementation uses registered modules and is a separate code path;
do not transfer that finding to its optimizer. External launchers/logging were
bypassed; no third-party logging credential was used.
