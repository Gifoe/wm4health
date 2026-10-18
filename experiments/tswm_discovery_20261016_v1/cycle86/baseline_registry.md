# Models, data and reproduction status

* Native sampler: OpenFIM commit cee2bb53be1e8e92b73f98dfdbe3605efc9119cc; PathGenerator.solve_equations AST body unchanged; public-source bytes checked against retained cache. Exact integrator is our substitution, not its original solver.
* Source-derived populations: three independent4096-mechanism pools, two-dimensional symmetric linear flows,16 initial states per mechanism; all exact trajectories finite. The nine threshold/bag conditions share each pool.
* Snapshot probes: ordinary affine regression of maximum growth rate on initial energy. No trajectory is supplied. This is a diagnostic of distributional association, not a world model.
* Factual meta-predictors:18 trained fits, two small MLP-based existing architectures, three data arms, three training seeds. All use1024 windows,8 observed states,16 future targets, identical paired initialization/batch schedules and3000 updates. One architecture assumes a symmetric linear generator; the direct predictor has no such transition constraint. They are not pretrained universal models.
* Ordinary controls: persistence and per-context ridge linear identification. Ridge choices are source-VAL-only. All three selected ridge values coincide, so their identical evaluation arrays are one effective baseline, not three independent replications.
* Evaluation: new seed2026864,1024 common unselected mechanisms, retained/rejected subgroups; new seed2026865,1024 retained mechanisms under original and permuted initial-condition pairings. Each evaluation pool is shared across all training seeds/models.
* Prior C71 original-budget campaign remains separate and live at the last process check. C86 does not shorten, restart or score its reserved rows.
* No official Pythia/T2-z/CGS/Bilinear-World-Model/FIM checkpoint is executed in C86. No original-paper headline result is claimed.
