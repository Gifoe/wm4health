# C84 baseline and execution registry

All pinned public source is cached on the server. None of these five learned baselines was trained or evaluated in this cycle.

| Repository | Exact commit | Cached text files | Inspection / execution status |
|---|---|---:|---|
| agentic-learning-ai-lab/adajepa | 51d8665b7978824bd218decab9e05ddb6eb1f47b | 132 | Actual adaptation trainer, parameter selection, reset, online planner and configuration inspected; no checkpoint run. |
| ExistentialRobotics/cross_embodiment_transfer | 3988511598a39c8a8a6530055b72609bf1b8b373 | 17 | Source-freezing call and aligner objectives/optimizers inspected; no robot reproduction. |
| ripl/invariance-through-latent-alignment | 62fed94f6726c6f2346564733d5719677b38d14e | 83 | Adaptation entry, agent interface and default adapt_dyn_model=False inspected; optional dynamics adaptation exists. |
| yangsizhe/MoVie | f721518846ffd1de4fdf09a4fc380b720ad04183 | 5 | README, two adaptation entry points and configs only; incomplete core implementation inspection. Git tree also lists released checkpoint files; no weights downloaded. |
| yupei-yang/CSR | 3724511b4461759112f89ee141bfbae302c2d513 | 91 | Simulation environment, transition/change-factor model and adaptation/expansion entry inspected; source also includes other benchmark directories. No reproduction claim. |

The executed ordinary control is a full affine output-space dynamics fit by unregularized least squares: eight fits, four fixed seeds, 64/256 noiseless target transitions, eight independent test plans per fit. It has the correct linear family and full observed state information, so it is a qualification control rather than a general learned-world-model baseline.

Numerical witnesses use two explicitly defined controlled systems and three invertible sensor maps. They are not trained networks. C71's original-budget neural experiment remains separate and is not counted as completed C84 evidence.
