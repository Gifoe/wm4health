# Research disposition

**Research Question:** Does the official PRO-DYN H96 forecasting comparison preserve its relative behavior under native longer block rollout?

**Evidence:** Eight of nine planned fits completed, 21440 successful-fit optimizer updates, plus three aborted native attempts with at least3200 recorded updates (exact final count unavailable), 357.6 seconds of successful-fit training. Four fits use the unchanged native loader; four use a cached-tensor data-copy compatibility wrapper qualified with128 bitwise window comparisons, identical forward/gradients and180000 stress lookups. One per-window-copy diagnostic failed and is retained. Three training seeds, three fixed inference RNG streams. Native block-loop replay maximum raw-unit difference 2.064e-05; all 108 metric rows replayed, MSE difference at most 2.22e-15. All selected/terminal weight and prediction hashes verified.

| Arm | View / horizon | Origins | MSE | Seed SD | MAE |
|---|---|---:|---:|---:|---:|
| Informer | common384 / 96 | 33 | 0.9331022 | 0.010694581798324812 | 0.6779045 |
| Informer | common384 / 192 | 33 | 0.9669725 | 0.008272868631990338 | 0.6979174 |
| Informer | common384 / 384 | 33 | 0.9980760 | 0.004767339959112647 | 0.7174020 |
| Informer | official96 / 96 | 3389 | 0.9249635 | 0.00632975047686514 | 0.6810566 |
| Informer_DYN | common384 / 96 | 33 | 0.5679840 | 0.007851532810893387 | 0.5287258 |
| Informer_DYN | common384 / 192 | 33 | 0.6203967 | 0.005527005731711553 | 0.5623397 |
| Informer_DYN | common384 / 384 | 33 | 0.6873893 | 0.010263024908111467 | 0.6051204 |
| Informer_DYN | official96 / 96 | 3389 | 0.5568774 | 0.023039633940105813 | 0.5311778 |
| Informer_PRO | common384 / 96 | 33 | 0.5638934 | 0.009103074482894463 | 0.5268823 |
| Informer_PRO | common384 / 192 | 33 | 0.6182903 | 0.005334984876761213 | 0.5605443 |
| Informer_PRO | common384 / 384 | 33 | 0.6818904 | 0.011967672020920967 | 0.6020416 |
| Informer_PRO | official96 / 96 | 3389 | 0.5488728 | 0.021384906985164674 | 0.5259923 |
| persistence | common384 / 96 | 33 | 2.0042792 | n/a | 0.9635653 |
| persistence | common384 / 192 | 33 | 2.0255401 | n/a | 0.9818464 |
| persistence | common384 / 384 | 33 | 2.0643298 | n/a | 1.0024529 |
| persistence | official96 / 96 | 3389 | 1.6558519 | n/a | 0.8453581 |
| ridge | common384 / 96 | 33 | 0.5234649 | n/a | 0.5036765 |
| ridge | common384 / 192 | 33 | 0.6508461 | n/a | 0.5845343 |
| ridge | common384 / 384 | 33 | 0.8036656 | n/a | 0.6666722 |
| ridge | official96 / 96 | 3389 | 0.5145175 | n/a | 0.5057308 |
| seasonal24 | common384 / 96 | 33 | 0.6704037 | n/a | 0.5108028 |
| seasonal24 | common384 / 192 | 33 | 0.7042884 | n/a | 0.5426519 |
| seasonal24 | common384 / 384 | 33 | 0.7470838 | n/a | 0.5756023 |
| seasonal24 | official96 / 96 | 3389 | 0.6211385 | n/a | 0.4849246 |


**Existing Methods:** Direct ridge, seasonal prediction, multistep system identification and dynamical-system reconstruction must be distinguished from a new framework.

**Proposed Idea:** None accepted from C62.

**Falsification:** A deficit removed by ordinary controls or correct training/interface cannot justify a new architecture. No test-set rescue tuning is performed.

**Feasibility:** Eight completed fits executed on one RTX5090, pinned original source and original ETTh1 blob.

**Next Experiment:** Select an independent, source-grounded residual capability after comparison with competent state-space/DSR controls. Do not rerun this test set to choose a loss.

ETTh1 is one previously examined measured system, with overlapping evaluation windows. Training seeds are initialization repeats, not independent systems. This is an exploratory screen. Inference RNG variation comes from native ProbAttention sampling; it is not learned predictive uncertainty. Extended point error alone does not establish incorrect physical dynamics. The source task makes no causal intervention guarantee.

Overall discovery remains ACTIVE_UNACHIEVED; accepted final directions: 0.
