# C55 population-objective control

No final research direction is accepted. The overall discovery objective remains active.

Source: [RP-GSSM](https://github.com/gatsby-sahani/rp-gssm/tree/a75225d5edceb972bb872dc9a335608733567dd4), checked unchanged against all21 pinned files. See PROTOCOL_V2.md and DERIVATION.md.

For the ideal scalar in-model case, the auxiliary population objective has a proper-likelihood lower bound attained at the true point. This rules out an unavoidable population-objective bias as the explanation in this specific setting. The literal reporting-jitter convention is retained separately. Multi-start searches are numerical checks, not the global-optimum proof. Some starts can stop away from the best point; all144 results remain visible. Neither fact alone identifies the cause of C54's finite-bank multidimensional training failure.

| a true | Noise r | T | Jitter | Best numerical a | Absolute a error | Objective excess over true likelihood bound | Starts within1e-6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
|0.5|0.03|6|0e+00|0.499994747|5.25e-06|3.95e-12|6/6|
|0.5|0.03|6|1e-05|0.500016355|1.64e-05|3.03e-08|6/6|
|0.5|0.03|24|0e+00|0.50000114|1.14e-06|1.63e-12|6/6|
|0.5|0.03|24|1e-05|0.500014738|1.47e-05|3.05e-08|6/6|
|0.5|0.3|6|0e+00|0.499999903|9.68e-08|1.75e-15|6/6|
|0.5|0.3|6|1e-05|0.500001635|1.63e-06|5.41e-10|6/6|
|0.5|0.3|24|0e+00|0.499999718|2.82e-07|8.83e-15|5/6|
|0.5|0.3|24|1e-05|0.500001815|1.81e-06|5.55e-10|5/6|
|0.5|3.0|6|0e+00|0.50000054|5.4e-07|3.09e-15|4/6|
|0.5|3.0|6|1e-05|0.500000154|1.54e-07|4.22e-11|4/6|
|0.5|3.0|24|0e+00|0.499999422|5.78e-07|6.4e-15|4/6|
|0.5|3.0|24|1e-05|0.500000147|1.47e-07|4.41e-11|4/6|
|0.95|0.03|6|0e+00|0.949999988|1.17e-08|3.57e-14|6/6|
|0.95|0.03|6|1e-05|0.950001588|1.59e-06|5.44e-08|6/6|
|0.95|0.03|24|0e+00|0.95000001|9.52e-09|1.04e-13|6/6|
|0.95|0.03|24|1e-05|0.95000253|2.53e-06|5.87e-08|6/6|
|0.95|0.3|6|0e+00|0.949999983|1.73e-08|2.94e-15|6/6|
|0.95|0.3|6|1e-05|0.950000463|4.63e-07|2.52e-09|6/6|
|0.95|0.3|24|0e+00|0.950000002|1.66e-09|-3.33e-16|6/6|
|0.95|0.3|24|1e-05|0.950000587|5.87e-07|3.27e-09|6/6|
|0.95|3.0|6|0e+00|0.950000374|3.74e-07|3.58e-14|4/6|
|0.95|3.0|6|1e-05|0.950000264|2.64e-07|1.5e-10|4/6|
|0.95|3.0|24|0e+00|0.950000053|5.34e-08|5.81e-15|4/6|
|0.95|3.0|24|1e-05|0.949999976|2.4e-08|2.89e-10|4/6|

The checks compare finite native code with independent scalar equations, reference Gaussian quadrature and exact target sigma-point expectation. Version1 stopped at the qualification stage because it omitted the source's jitter contribution to adjacent cross covariance; version2 corrected that copied convention. No native package, C54 checkpoint, C54 readout or C54 scored result changed. No new neural training was run.

Exact qualifications are in outputs/QUALIFICATION55_V2.json; all numerical starts are in outputs/population55_v2.csv and outputs/population55_v2.json. Full iteration traces stay on configured server artifact storage with hashes in the JSON.
