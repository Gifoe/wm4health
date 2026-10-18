# Time-series world-model discovery: C62 status

Overall research goal: ACTIVE_UNACHIEVED. Accepted final research directions: 0.

## Latest result and scope correction

C62 is a partial PRO-DYN source-protocol reproduction on measured ETTh1:8 of9 planned fits completed, with21440 successful-fit optimizer updates. Three formal attempts ended in native Windows access violations; at least3200 aborted-attempt updates were recorded, with exact crash-time totals unavailable. Four completed fits use native window copies and four use a cached tensor copy wrapper qualified with128 exact window comparisons, identical forward/gradients and180000 stress lookups. A per-window-copy diagnostic also failed. The wrapper did not eliminate all native faults; root cause remains unknown.

Crucially, the selected official scripts use L=H=96. PRO and DYN then have the same computation after renaming linear_PRO to linear_DYN. Same-weight, same-RNG CPU forward and gradients matched exactly in training and evaluation modes. The original choice of this equal-length cell was insufficient to isolate PRO/DYN mechanisms. There are two distinct neural architectures in the completed comparison, not three. PRO has a complete three-seed set; DYN's two fits are auxiliary duplicate executions. See cycle62/EQUIVALENCE_NOTICE62_V1.md and cycle62/outputs/PRO_DYN_EQUIVALENCE62_V1.json. This corrects the earlier pre-test note describing different modified computational paths.

All eight sealed fits were evaluated using the unchanged native CPU batch forecasting path. H96 full rolling forecasting and common-origin96/192/384 free block rollouts are separate. Persistence, seasonal-24 and validation-selected ridge receive the same past measurements. No checkpoint or method is chosen by test outcomes. Numerical loop qualification and all retained metric rows are independently replayed. This is not a full paper-table reproduction, GPU inference benchmark, causal intervention experiment or new world-model method.

| Arm | View | Horizon | MSE |
|---|---|---:|---:|
| Informer | common384 | 96 | 0.9331022 |
| Informer | common384 | 192 | 0.9669725 |
| Informer | common384 | 384 | 0.9980760 |
| Informer | official96 | 96 | 0.9249635 |
| Informer_DYN | common384 | 96 | 0.5679840 |
| Informer_DYN | common384 | 192 | 0.6203967 |
| Informer_DYN | common384 | 384 | 0.6873893 |
| Informer_DYN | official96 | 96 | 0.5568774 |
| Informer_PRO | common384 | 96 | 0.5638934 |
| Informer_PRO | common384 | 192 | 0.6182903 |
| Informer_PRO | common384 | 384 | 0.6818904 |
| Informer_PRO | official96 | 96 | 0.5488728 |
| persistence | common384 | 96 | 2.0042792 |
| persistence | common384 | 192 | 2.0255401 |
| persistence | common384 | 384 | 2.0643298 |
| persistence | official96 | 96 | 1.6558519 |
| ridge | common384 | 96 | 0.5234649 |
| ridge | common384 | 192 | 0.6508461 |
| ridge | common384 | 384 | 0.8036656 |
| ridge | official96 | 96 | 0.5145175 |
| seasonal24 | common384 | 96 | 0.6704037 |
| seasonal24 | common384 | 192 | 0.7042884 |
| seasonal24 | common384 | 384 | 0.7470838 |
| seasonal24 | official96 | 96 | 0.6211385 |

ETTh1 is one previously examined measured system; test windows overlap. The model has only past seven-channel observations and known future calendar marks, with no physical action plan or future measured forcing. Growing point error cannot by itself identify a physical-dynamics failure. Approximate-attention random seeds are algorithmic randomness, not samples of the physical future. No novel framework claim is justified.

## Storage and publication

Raw measurements, selected/terminal weights and forecasts remain on the authorized server G disk. Only authored code and compact evidence are uploaded to Gifoe/wm4health; no local result-file synchronization. The unmodified upstream source is recorded by pinned Git objects and SHA256, without source mirroring because the pinned repository has no license file. All original failures, amendments and design corrections are retained.

## Preserved earlier evidence

C60: a learned ordinary NSSM repaired tested memory-bank deficits on measured N87 and pick-place systems. C61: a learned ordinary explicit-duration HSMM met conditional-density/survival thresholds in two controlled duration families. These are exclusions of tested recipes, not proofs that every real dynamics problem is solved.

The objective remains a reusable time-series world-model framework. Generic subsystem composition already has PHNN, input-output identification and C-PHAST precedents; these were scope-checked, not executed here. The next candidate needs a concrete residual beyond competent state-space/DSR learning and an independent mechanism/domain. The latest experiment establishes baseline qualification and a design boundary, not a publishable new direction.

Exact current results: cycle62/experimental_results.md, cycle62/outputs/SUMMARY62_V1.json and cycle62/PARTIAL_CLOSURE_AMENDMENT62_V1.md. Historical snapshots remain unchanged.
