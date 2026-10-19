# C48 independently verified descriptive results

Formal fits: 16; updates: 40750; numerical flags: 0.

Scores use TRAIN-standardized targets. Seed ranges describe two optimization seeds, not uncertainty across independent worlds.

| Process | Arm | Reveal hours | MSE seed mean | Min seed | Max seed | Tail24 MSE mean |
|---|---|---:|---:|---:|---:|---:|
| ett | node | 0 | 0.0646841 | 0.0644535 | 0.0649147 | 0.0832458 |
| ett | node | 24 | 0.0417644 | 0.0415748 | 0.0419541 | 0.0417644 |
| ett | fixed_dde | 0 | 0.0679714 | 0.0679113 | 0.0680314 | 0.0868521 |
| ett | fixed_dde | 24 | 0.0435282 | 0.0431022 | 0.0439542 | 0.0435282 |
| ett | learned_dde | 0 | 0.0675558 | 0.0669184 | 0.0681932 | 0.0863522 |
| ett | learned_dde | 24 | 0.0434121 | 0.0426518 | 0.0441724 | 0.0434121 |
| ett | gru | 0 | 0.0658051 | 0.0635921 | 0.0680182 | 0.0832203 |
| ett | gru | 24 | 0.0431851 | 0.0401105 | 0.0462598 | 0.0431851 |
| ett | ridge | 0 | 0.0732976 | 0.0732976 | 0.0732976 | 0.0946744 |
| ett | ridge | 24 | 0.0452926 | 0.0452926 | 0.0452926 | 0.0452926 |
| ett | persistence | 0 | 0.0779690 | 0.0779690 | 0.0779690 | 0.1020660 |
| ett | persistence | 24 | 0.0496342 | 0.0496342 | 0.0496342 | 0.0496342 |
| ett | seasonal24 | 0 | 0.1123274 | 0.1123274 | 0.1123274 | 0.1374534 |
| ett | seasonal24 | 24 | 0.0644178 | 0.0644178 | 0.0644178 | 0.0644178 |
| jena | node | 0 | 0.1551004 | 0.1524770 | 0.1577237 | 0.2120558 |
| jena | node | 24 | 0.1556371 | 0.1548462 | 0.1564280 | 0.1556371 |
| jena | fixed_dde | 0 | 0.1550265 | 0.1540707 | 0.1559823 | 0.2095815 |
| jena | fixed_dde | 24 | 0.1358621 | 0.1305220 | 0.1412022 | 0.1358621 |
| jena | learned_dde | 0 | 0.1575976 | 0.1563051 | 0.1588901 | 0.2149124 |
| jena | learned_dde | 24 | 0.1297083 | 0.1268393 | 0.1325772 | 0.1297083 |
| jena | gru | 0 | 0.1382987 | 0.1344392 | 0.1421583 | 0.1992946 |
| jena | gru | 24 | 0.1104120 | 0.1078987 | 0.1129253 | 0.1104120 |
| jena | ridge | 0 | 0.1598312 | 0.1598312 | 0.1598312 | 0.2392916 |
| jena | ridge | 24 | 0.1188755 | 0.1188755 | 0.1188755 | 0.1188755 |
| jena | persistence | 0 | 0.2075381 | 0.2075381 | 0.2075381 | 0.2915467 |
| jena | persistence | 24 | 0.1584764 | 0.1584764 | 0.1584764 | 0.1584764 |
| jena | seasonal24 | 0 | 0.2477439 | 0.2477439 | 0.2477439 | 0.3335200 |
| jena | seasonal24 | 24 | 0.1722848 | 0.1722848 | 0.1722848 | 0.1722848 |

## Real-observation continuation on identical target hours

Compare only original future hours25-48. Negative differences favor incorporating the true first24-hour prefix.

| Process | Arm | Seed | Unrevealed tail MSE | Revealed tail MSE | Change % |
|---|---|---:|---:|---:|---:|
| ett | node | 2026481 | 0.0826457 | 0.0415748 | -49.70 |
| ett | node | 2026482 | 0.0838458 | 0.0419541 | -49.96 |
| ett | fixed_dde | 2026481 | 0.0874533 | 0.0439542 | -49.74 |
| ett | fixed_dde | 2026482 | 0.0862509 | 0.0431022 | -50.03 |
| ett | learned_dde | 2026481 | 0.0872581 | 0.0441724 | -49.38 |
| ett | learned_dde | 2026482 | 0.0854464 | 0.0426518 | -50.08 |
| ett | gru | 2026481 | 0.0806593 | 0.0401105 | -50.27 |
| ett | gru | 2026482 | 0.0857812 | 0.0462598 | -46.07 |
| ett | ridge | None | 0.0946744 | 0.0452926 | -52.16 |
| ett | persistence | None | 0.1020660 | 0.0496342 | -51.37 |
| ett | seasonal24 | None | 0.1374534 | 0.0644178 | -53.13 |
| jena | node | 2026481 | 0.2111596 | 0.1548462 | -26.67 |
| jena | node | 2026482 | 0.2129519 | 0.1564280 | -26.54 |
| jena | fixed_dde | 2026481 | 0.2124682 | 0.1305220 | -38.57 |
| jena | fixed_dde | 2026482 | 0.2066948 | 0.1412022 | -31.69 |
| jena | learned_dde | 2026481 | 0.2180566 | 0.1268393 | -41.83 |
| jena | learned_dde | 2026482 | 0.2117681 | 0.1325772 | -37.40 |
| jena | gru | 2026481 | 0.1928131 | 0.1078987 | -44.04 |
| jena | gru | 2026482 | 0.2057760 | 0.1129253 | -45.12 |
| jena | ridge | None | 0.2392916 | 0.1188755 | -50.32 |
| jena | persistence | None | 0.2915467 | 0.1584764 | -45.64 |
| jena | seasonal24 | None | 0.3335200 | 0.1722848 | -48.34 |

## Attribution limits

Hard-cap fits: []. Final-step selections: [].
VAL numerical-refinement flags: []. Inspect individual diagnostics before attributing differences to memory.
Paired per-origin differences and win fractions are descriptive; overlapping histories and temporal dependence preclude treating all origins as iid replicates.
The scalar NODE is a weak ablation. Learned delay versus fixed delay shares initialization and schedules; GRU/ridge are stronger history-aware controls.
An existing delay method winning supports its utility under this protocol. It does not establish a novel framework. A weak arm losing does not establish a research gap.
A concrete new learning interaction and independent confirmatory evidence remain required for the full discovery objective.
