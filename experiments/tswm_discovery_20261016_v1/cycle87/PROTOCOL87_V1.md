# Native nonlinear sampler comparison, C87 V1

Status: prospective experiment, not a paper reproduction or accepted research direction.
Parent evidence: C86. Goal remains ACTIVE/UNACHIEVED. Prior turn was a status response (no new experimental progress).

## Question and limits

Does complete-trajectory rejection affect inference and free simulation by the unchanged native FIM-ODE architecture, after equal-data/optimization pairing and competent ordinary identification controls? We test a data operation, not a new architecture. The paper (arXiv:2602.08733v1, Appendix B.2) documents 9 initial conditions and whole-ODE rejection at magnitude 100/nonfinite values. Its actual checkpoint's exact generated dataset is unavailable. This experiment uses our own bounded nonlinear families and threshold 2.5, not the paper's exact pretraining population, and cannot attribute a released checkpoint defect causally.

## Fixed synthetic population

Two 2D autonomous families, mixed in a single shared learner:

- Duffing: dx=v, dv=mu*x-beta*x^3-gamma*v+b, multiplied by speed. Independent uniforms mu[-.5,1.5], beta[.5,1.5], gamma[.1,.8], b[-.3,.3], speed[.6,1.6].
- Stuart-Landau: dx=(mu-beta*r^2)*x-omega*y, dy=omega*x+(mu-beta*r^2)*y, multiplied by speed. mu[-.4,1.4], beta[.3,1.2], omega[.5,2], speed[.6,1.6].

These families have no positive finite-time explosion. Screening can reject valid finite solutions. Screening: 9 iid ICs N(0,.8^2 I), 201 times over [0,10], RK4 with 4 substeps per .05 interval. Whole-filter accept iff all components of all 9 paths remain strictly below 2.5 and finite. TRAIN candidate pool 8192/family, VAL 1024/family, seeds 2026870 and 2026871. Abort V1 if either family provides fewer than 1024 accepted TRAIN mechanisms; do not silently change threshold or pool. Keep rejected mechanisms and all counts.

Training contexts: first 2 paths, first 33 times [0,1.6], iid additive noise .01. Source ground-truth field labels are allowed as in FIM synthetic pretraining. 32 query locations per mechanism, independent U[-2.5,2.5]^2, common between arms. Uniform query grid avoids changing field supervision when the context bounding box changes. This differs from the paper's trajectory-derived box and is disclosed.

Arms (1024 mechanisms/family, 2048 total):
1. whole_filter: first accepted mechanism IDs, original first 2 paths.
2. resampled_ic: exactly the same IDs/field targets, independently resampled 2 ICs from the original Gaussian.
3. unselected: first 1024 mechanism IDs from the unscreened population, original first 2 paths.

Thus arms 1/2 isolate the conditional IC selection while retaining the mechanism marginal. Arm 3 additionally changes the mechanism population. All arms retain equal field-location and context budgets. At least 128/family VAL mechanisms are held separately; no TRAIN IDs overlap VAL. Context noise is identical by arm position, model seed and split. Comparisons are not action interventions.

## Native model and budgets

Pinned OpenFIM cee2bb53be1e8e92b73f98dfdbe3605efc9119cc, unchanged FIMODE model code. Existing pinned public weights initialize all arms identically, 12,968,068 parameters. This is controlled adaptation, not full from-scratch pretraining. All native trainable parameters are enabled. Source native forward/data preparation/loss are used. External common noise replaces internal stochastic corruption, recorded explicitly. Original normalized l1 loss and native uncertainty weighting retained. Batch16, AdamW 1e-4, decay1e-4, gradient clip1, 3000 updates; seeds2026872/3/4. Pair initialization, minibatch indices and torch RNG by seed. Checkpoints at0/500/1000/1500/2000/2500/3000. Select using a common unselected VAL pool and normalized field l1; no held-out evaluation access before all selections are sealed. A five-step native gradient smoke/throughput/memory test precedes launch and does not access EVAL. GPU jobs must fit alongside the confirmed-live C71 job, without stopping it.

## Evaluation, held unopened until selection

Fresh seed2026875, 1024 candidate mechanisms/family. Use first128 accepted/family for a paired query cohort. Compare their original 2 context paths against a fixed permutation of the same multiset of 2-path IC pairs; recompute physical trajectories for each mechanism/IC pairing. Exact mechanism and IC marginals preserved, but visited-state/information content is not matched. Also report first128 unselected/family, with unconditioned contexts. Fresh query ICs are iid N(0,.8^2 I), common to every model. Integrate free simulation to10, using the same qualified RK4 solver, after context only. No future observations or drift labels enter inference. Report field errors separately on context-visited locations and query rollout truth locations, and simulation at horizons1/5/10. MSE, mean/median trajectory NRMSE (RMS truth denominator floored .1), finite completion and divergence counts all reported, no case deletion. Every numerical failure receives explicit failure accounting, never a finite-only favorable aggregate.

## Ordinary controls and qualification

Persistence and weak-form polynomial vector-field identification degrees1/2/3, ridge0/1e-6/1e-4/1e-2/1. Fit from the same noisy context only, with constant included. Degree/ridge selected on common VAL, no mechanism/family label provided. Also frozen pretrained native model. Exact family equations and clean field labels are simulation/reference diagnostics only, not ordinary input. Native competence requires clear VAL improvement over frozen checkpoint and finite trajectories; a remaining loss descent at update cap is reported, not declared converged. If ordinary controls suffice, method-necessity is not supported. The nine-fit comparison remains exploratory evidence even if favorable; acceptance also needs prior-art differentiation and independent held-out family evidence.

## Falsification and artifacts

Retire the current strong shortcut claim if the native learner is robust to re-pairing, the effect is explained by context coverage alone, or the apparent repair trades off normalized/source errors without a consistent benefit. Keep negative evidence. Do not call ordinary resampling novel. Independently replay saved arrays/weights and solver tolerances before publishing. Raw arrays/checkpoints/logs stay on the15TB remote disk; source, hashes, summaries and figures go only to Gifoe/wm4health. No result/model/data synchronization to local storage.
