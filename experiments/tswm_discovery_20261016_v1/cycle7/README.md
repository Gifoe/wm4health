# Cycle 7: action-equivalence objective audit

Scientific jobs complete; overarching discovery goal remains **ACTIVE and unachieved**. No direction is accepted for publication development.

27 matched fits and 6 finite objective witnesses executed on the remote RTX5090. Source [ActSWM](https://github.com/KeithBlackmore/ActSWM-) pinned to `3d2eab6dccd3034506f2cec0ee227415b291f75b`. Fourteen primary PDFs archived with hashes.

This is a **new controlled mechanism experiment**, not an original-paper Minecraft reproduction, not a real-data forecasting result and not a demonstrated new method. Unchanged native class/function bodies are loaded by AST; the model is deliberately reduced, and the physical-state encoder is fixed to identity. The learned encoder, pixel information bottleneck and native SIGReg gradients are outside scope. No existing source or earlier sealed cycle was modified.

Three 4D deterministic plants use x_next=.8*x+[0,.04,-.03,.02]+force(a)*e0. Force is zero, .35*a, or .35*sign(a)*relu(abs(a)-.5). Actions are randomized independently of state, with 30% exact zero and 70% uniform[-1,1], including during training. Train/validation/test counts are 8192/1024/2048. Cohort seeds are 15001/15002/15003 plus 100*plant_index. Three initialization seeds share these cohorts: they measure initialization variability, not independent dataset replication.

The native ARPredictor is reduced to width32, depth1, heads2, context1 and horizon3; 15,312 trainable parameters. All arms instantiate the same 267,777-parameter frozen random native inverse readout; only full uses its loss. Native-default AdamW1e-4, wd1e-3, clipping1 and 1,500 updates of batch128 are held fixed. Arms are prediction only; prediction+hinge(weight.5,margin.3); and prediction+hinge+frozen-IDM(weight1). Fixed-step final is primary; minimum factual-validation MSE checkpoints are secondary, with no test selection. Native prediction loss is independently matched to the evaluator before every fit. Native alternate-action rollout retains the first context action and zeroes later actions; its first-step hinge is therefore a constant, disclosed rather than silently corrected.

Cosine separation is evaluated in the fixed physical coordinates; no claim of representation-coordinate invariance is made. Null-plan sanity checks assert exact equality of physical trajectories for action*.4 in null/deadzone plants. Candidate regret compares the chosen plan with the true best among a fixed bank of 64 plans at 256 held-out initial states. This is offline candidate ranking, not closed-loop MPC; the physical oracle is never used for training or checkpoint selection.

The default auxiliary objectives amplify false effects in the tested plants. A perfect physical predictor can have worse native total objective than a deliberately wrong predictor. However, prediction-only already solves these fully observed tasks, and existing learned inverse models/action abstractions are relevant controls. This result supports a scoped objective mismatch, not a novel general research problem.

See experimental_results.md, hypothesis_space.md and final_research_direction.md. Runtime/data/source/checkpoint hashes are included in the artifact manifest after sealing. All content remains private.
