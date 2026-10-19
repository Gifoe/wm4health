# General time-series world-model direction: not yet established

**Research Question:** Can one reusable learned state adapt efficiently across physical systems whose inputs change both dynamics and observability,while retaining predictive and simulation fidelity?

**Evidence:** This particular question is **not established by current experiments**. Cycle26 supplies counterevidence to a tempting alternative explanation:exact autonomous maps can have wrong forced responses solely due to endpoint insertion at coarse time.

| Case at h=.64 | Endpoint | Symmetric | Split4 | Exact discrete input |
|---|---:|---:|---:|---:|
| Linear control | 76.4589% | 12.4296% | 15.8646% | 0.0000% |
| Nonlinear source | 75.7138% | 12.7942% | 15.7219% | not applicable |

**Existing Methods:** Bilinear controlled-generator EM,GRU/NARX,stochastic state-space models,CoDA and physics-map+MSM are direct controls. Falasca2025/2026 already cover broad response and memory motivation. Their sufficiency in a properly matched transfer experiment remains to be tested.

**Proposed Idea:** No new architecture selected yet. A potential transferable state/filter/generator interface must first show a residual finite-data interaction that these existing controls do not solve. Generic action-conditioning,posterior correction,memory,semigroup consistency and input integration are not innovations by themselves.

**Falsification:** Abandon the novelty claim if ordinary controlled identification,regularized latent/readout adaptation or sufficient excitation removes the failure;if physical timing/interface corrections explain the gain;or if apparent benefits depend on privileged observations,test tuning or one plant.

**Feasibility:** Native small controlled generators and motor/slow-manifold equations fit one5090. This cycle measured 67.20s and 206.94MiB peak allocated for its different oracle audit;these timings do not predict neural-training cost. No global runtime/dependency change.

**Next Experiment:** Execute and independently verify the exact-source bilinear-EM positive control before designing a modern transfer model. Freeze cross-system calibration/held-out action protocol only after source fidelity checks. Do not infer novelty from failure of an untrained/frozen forecasting head alone.

Final accepted directions: **0**. Goal **ACTIVE,UNACHIEVED**. This is an evidence-driven candidate ledger,not a completed top-conference research claim. Dataset optimization is not the final target.
