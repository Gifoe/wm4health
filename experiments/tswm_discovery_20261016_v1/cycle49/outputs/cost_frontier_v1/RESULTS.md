# Fixed-time quality comparison

Budgets60/120/180 seconds were sealed before primary EVAL. Best observed-VAL checkpoint available within each budget is used. All points are retained.

| Process | Budget seconds | Density | Score energy | ELBO energy | Score minus ELBO % |
|---|---:|---:|---:|---:|---:|
| rotating | 60 | 0.25 | 0.3175717 | 0.3218942 | -1.34 |
| rotating | 60 | 1.0 | 0.2938129 | 0.2922710 | +0.53 |
| rotating | 60 | 4.0 | 0.3105495 | 0.2877281 | +7.93 |
| rotating | 120 | 0.25 | 0.3132713 | 0.3178374 | -1.44 |
| rotating | 120 | 1.0 | 0.2900511 | 0.2873712 | +0.93 |
| rotating | 120 | 4.0 | 0.2950197 | 0.2822308 | +4.53 |
| rotating | 180 | 0.25 | 0.3126661 | 0.3120794 | +0.19 |
| rotating | 180 | 1.0 | 0.2900367 | 0.2857215 | +1.51 |
| rotating | 180 | 4.0 | 0.2926549 | 0.2832904 | +3.31 |
| bistable | 60 | 0.25 | 0.3386766 | 0.3379103 | +0.23 |
| bistable | 60 | 1.0 | 0.3054223 | 0.2967053 | +2.94 |
| bistable | 60 | 4.0 | 0.3091828 | 0.2914323 | +6.09 |
| bistable | 120 | 0.25 | 0.3306399 | 0.3402954 | -2.84 |
| bistable | 120 | 1.0 | 0.2925257 | 0.2921359 | +0.13 |
| bistable | 120 | 4.0 | 0.2898858 | 0.2857091 | +1.46 |
| bistable | 180 | 0.25 | 0.3390242 | 0.3357472 | +0.98 |
| bistable | 180 | 1.0 | 0.2935993 | 0.2882695 | +1.85 |
| bistable | 180 | 4.0 | 0.2888830 | 0.2833527 | +1.95 |

Energy is lower-is-better; negative differences favor the observation-averaged score. Rows average two training seeds. All density views of a physical episode remain paired.
This is a measured fit-time frontier under the actual server workload; launch/setup/preflight are excluded. It is not a cross-hardware efficiency guarantee.
The bootstrap describes finite test-episode uncertainty conditional on the two fitted model seeds. It does not certify real-domain utility or isolate the effect of analytic observation averaging from changing the training objective.
