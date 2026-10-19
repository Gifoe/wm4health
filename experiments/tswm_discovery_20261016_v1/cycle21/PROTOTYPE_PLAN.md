# Independent-cohort development test: action-responsive targets and state sufficiency

Status before execution: candidate development, not accepted novelty. Cycle20 gives a real controlled failure:4D joint/inverse representations recover actions almost perfectly while losing a visible fresh gain that changes physical transitions;12D/full-observation controls largely repair it. Thus the precise question is compact causal sufficiency, not an impossibility for all world models. Older causal state abstraction, predictive states, observation prediction and recent stochastic JEPAs remain necessary comparisons.

Candidate mechanism: learn a full-observation teacher from the SAME offline training transitions, using no physical/reward labels. Query its one-step predictions at all four in-support binary actions. Across training states, differences between those predictions estimate an action-responsive output span. Project teacher absolute future predictions onto that span, then train a compact world's CURRENT-state encoder to predict those projected targets. Unlike retaining only controllable input coordinates, this requires keeping any uncontrollable current driver needed for the controlled outcomes. Unlike only contrasting actions, projected absolute futures can retain additive incoming drivers as well. This is a proposed interaction of established primitives, not a claimed new causal-abstraction theory or proof of universal identification.

Projection P=U U^T: compute the12x12 covariance of teacher action-mean-centered predicted observations on TRAIN only; retain the smallest eigen-subspace explaining95% of action-response energy. No physical channel identities, reward, test signal or hand-set rank2. Record spectrum/rank. Global linear output subspaces are a limitation: nonlinear observation geometries may require richer targets; these tests use invertible linear observation mixes and do not establish that extension.

Independent data: fresh train/validation/test seeds20262111/12/13 (+1000*family index);192/64/128 trajectories,64 recorded transitions after64 burn-in. Reuse the three explicitly declared physics families from cycle20, not its trajectories/checkpoints. Add a queueing family with current random arrivals g in{.4,1.6}: q'=max(0,.8q+.7g-(.8+.4u));p'=max(0,.75p+.2q+.45g-(.5+.3v)). This is a new simple mechanism family, not real datacenter evidence. Nine AR(.98) nuisance coordinates;12D orthogonal observation mix, train-only normalization. Same four binary actions in all splits. No future gain or oracle physical labels in model fitting.

The three original families are the clean incoming-driver comparison. Queueing additionally has physically equivalent actions when clipping empties a queue, a known inverse-objective limitation from cycle7. Treat it as an exploratory stress test, not an independent isolation of incoming-innovation loss. Compare against the raw observation teacher there and do not attribute all queueing errors to the new candidate mechanism.

Three seeds2021/22/23. Teacher: raw-observation two-layer64 MLP,1500 Adam updates,lr1e-3,batch256. Student: same4D two-layer64 encoder/predictor/inverse and variance/covariance regularizer as cycle20. Auxiliary48-output head gets targets from teacher bank, with per-output training mean removed and a SINGLE global training RMS scale (avoids amplifying near-zero channels independently). Auxiliary weight1. No test tuning. All student inference backbones retain the same compact interface; teacher and auxiliary head costs are reported separately.

Ten specifications per family/seed (120 student fits):
1. joint+inverse,3000 updates (budget control);
2. observation-future+inverse,3000;
3. joint+inverse+current-observation reconstruction,3000; same48-output head with four repeated observations;
4. joint+inverse+full teacher future bank,1500 student+1500 teacher;
5. joint+inverse+teacher action-contrast bank,1500+1500;
6. joint+inverse+projected teacher absolute future bank,1500+1500 (candidate);
7. same with a fixed random output subspace of the SAME learned rank,1500+1500;
8.12D joint+inverse,3000 (capacity control);
9. source-informed MoP/EMA control,3000: unit-normalized context/heads/EMA targets,2 predictors, context-only router CE weight.1, hard-assignment load-balance value.01, EMA.99, plus the common inverse/anti-collapse controls;
10. that same MoP/EMA backbone+projected teacher bank,1500+1500 (stochastic candidate).

MoP control is a fully specified task adaptation, not a native OGBench execution. The hard-assignment load-balance statistic is piecewise constant; do not claim it independently supplies a gradient. Source paper's predictor/encoder distinction is respected. Standalone SMWM, full RSSM, CDL and source-native temporal tasks remain follow-ups if this prototype survives.

Exactly one fixed run, no post-test weight/architecture selection. All world fits, teachers and privileged DIAGNOSTIC probe choices are completed before test metrics. Reuse linear/MLP frozen-current-state and successor probes from cycle20; labels never update world encoders. Decode physical states only for evaluation. Evaluate current/gain information, model and frozen-probe successor errors, oracle successor decoding, within-support current-gain flips and their physical effects. Queueing's gain-blind oracle is the average of the two gain-conditioned successors, not successor-at-mean-gain (clipping is nonlinear).

Readiness: every raw teacher must have physical validation normalized MSE<.05; otherwise no method claim for that family. Preserve all failures and null results. Need repeatable gains over equal-capacity/budget reconstruction, future-bank and properly stochastic controls, not only the weak cycle20 arm. Failure or recovery by an existing control falsifies architecture necessity. Positive results would justify an independent realistic/native time-series test and deeper overlap review; these synthetic prototypes alone cannot finish the research-discovery goal.
