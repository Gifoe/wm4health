# Negative evidence and claims not established

1. A generic "marginal calibration is insufficient for joint trajectories" paper is unoriginal. Function-valued models, AR-CNP, TNPs, efficient autoregressive buffers, and generative meta-dynamics already address dependence.
2. A new covariance neural head is not justified by this gate. An ordinary covariance fit on separate systems improved joint NLL and total coverage substantially, at identical calibrated means/marginals.
3. Scalar variance temperature is insufficient here. It leaves total-functional coverage51.953% versus50.781% native; its joint NLL difference is+.000873 with a pilot interval spanning zero.
4. A claim that this frozen baseline ignores system context is contradicted by RMSE.131670 native versus1.406205 wrong-context. Do not replace a competent released baseline with an undertrained toy.
5. Full covariance is not perfectly calibrated across all summaries (DCT1=87.891%). Neither this residual nor an ordinary repair establishes a new, important, unsolved mechanism. Input-dependent bias, non-Gaussianity and oracle normalization remain competing explanations.
6. Whole-sequence normalization depends on future outputs, including intermediate nonlinear inputs. No prospective raw-unit deployment or intervention claim follows. No inference that the source paper's ensemble predictive coverage failed: its native ensemble primarily combines means.

Cycle16 covariance-repair line remains closed after full end-to-end validation did not show the proposed benefit. No data/weights/source were overwritten. Accepted novel directions0.
