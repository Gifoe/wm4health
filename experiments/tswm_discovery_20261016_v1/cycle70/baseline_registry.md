# Execution and information registry

| Item | What actually ran | Boundary |
|---|---|---|
| RP-GSSM source a75225d5edceb972bb872dc9a335608733567dd4 | Native ACLGParam, continuous Flax ActionNetwork, discrete mapper, ConstrainedIVFreeEnergy.get_posterior, Gaussian chain moments |15 exact source blobs verified; not a proxy neural model. |
| Ordinary interface/flag adapters | Discrete initialization signature; retain initial-state flags; constant/no-bias output masks | Authored outside upstream source. All affected original outputs/errors retained. |
| Independent dense reference |64-dimensional joint Gaussian conditioning | Known true dynamics/sensor likelihood, privileged parameters. |
| Independent replay | Joseph-form Kalman filter plus RTS smoother | Separate algorithm; full and16-observation prefix per sequence. |
| Meta-iLaD | Primary-paper scope review only | No released implementation at pinned revision; zero native fits. |
| Compositional continual benchmark | Primary-paper/project review only | Robotics neighbor; no claimed suite reproduction. |

Historical correction: RP-GSSM had actual native training in C2 and action-prior source inspection in C46. C70 is a narrow previously unexecuted branch qualification, not a first reproduction. The same C2 JAX environment is reused unchanged. The transition mapper networks are randomly initialized and frozen; known dynamics, Gaussian likelihood and fixed covariance are privileged. No trained representation or system identification claim.

