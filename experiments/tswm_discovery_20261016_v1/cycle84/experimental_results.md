# C84 results: the broad attribution requirement is insufficient

The source audit identifies substantial prior art for observation adaptation, selective updates and residual compensation. The numerical qualification additionally rejects a universal requirement to name the changed internal component before producing correct observable action predictions. No new learned-model research gap is established.

| Check | Actual result |
|---|---:|
| Common-control trajectory pairs | 192 |
| Maximum observable discrepancy | 2.77555756156e-15 |
| Independent continuous-ODE pairs | 12 |
| Maximum independent ODE discrepancy | 6.66133814775e-16 |
| Matched action-response pairs checked | 168 |
| Maximum paired response discrepancy | 3.08086889333e-15 |
| Ordinary affine fits / independent test rollouts | 8 / 64 |
| Maximum ordinary-fit rollout discrepancy | 6.11732886568e-14 |
| Saved trajectory scalar values independently checked | 187824 |
| Pinned public text sources rehashed | 328 |

These are exact-construction numerical witnesses, not 192 independent scientific tasks or neural training seeds. The eight ordinary fits have the correct linear family, noiseless full observations and nested 64/256-transition budgets. Their success cannot establish nonlinear or partially observed adaptation performance. Independent exact linear propagation differs from the main RK4 reference by at most 1.92458504689e-09; independent QR and SVD coefficient fits differ by at most 5.99000016255e-15.

## Why component labels do not define the objective

Let x'=f(x,u), y=g(x), with invertible g. An alternative model uses state z=g(x), identity observation, and z'=J_g(g^(-1)(z)) f(g^(-1)(z),u). For the same initial observable state and every allowed action plan, both models produce the same y. This is established state-space conjugacy. One tested nonlinear transformation also preserves the known input vector (0,1) and the zero reset; rank-deficient known actuation alone does not name every latent coordinate.

The 96 separate named-coordinate intervention pairs differ in the physical operator applied. They have positive recipient discrepancies, but those cannot be reported as action-response errors: do(x_0=0.8) and do(z_0=0.8) are different interventions after changing coordinates. Physical anchors or known kinematic structure can exclude the ambiguity; we make no claim that it survives those assumptions.

## Decision and limits

The generic attribution-first formulation is not an accepted direction. The narrower question of physically anchored response calibration remains only a development question. None of ILA, CSR, MoVie, Cross-Embodiment Transfer or AdaJEPA was executed here. This audit does not refute CSR's restricted theorem, reproduce an adaptation paper, demonstrate a failure after valid physical anchoring, or validate a new method.

Original research objective: ACTIVE / UNACHIEVED. Qualified final directions: 0. All raw arrays stay on the server; the GitHub snapshot contains code, compact results, source hashes and reports.
