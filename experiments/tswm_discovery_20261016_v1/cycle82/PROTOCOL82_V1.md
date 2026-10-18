# C82: Can a pretrained joint dynamics prior be reused as a local mechanism?

This exploratory cross-contract experiment tests an unresolved prerequisite for composing low-dimensional learned world models into larger systems. It is not an original-paper reproduction, a new architecture, or an assertion of novelty. FIM-ODE's published task assumes a closed autonomous observed state; a neighborhood inside a larger system may not satisfy that assumption even when the target component's direct parents are all visible.

## Question, competing explanations and scope

Can a frozen joint vector-field estimator recover a target mechanism whose inputs are observed, when a neighboring input variable is driven by an omitted boundary state? Distinguish (H1) general prior/estimator error, (H2) inadequate local state coverage, and (H3) a practical mismatch between a closed-system pretraining prior and target-only mechanism inference. A failure outside the original task is not a bug or a rebuttal to the paper.

Candidate extension, only if justified later: pretrain or adapt an estimator for a target component with arbitrary observed boundary drives, then compose target estimators on a sparse graph. The component estimators need not model the boundary variables' own dynamics. Classical input-output identification and compositional operators are direct competitors; generic modularity or control conditioning cannot be claimed as new.

## Frozen controlled carrier

Four target mechanisms, parameters (a,b,c,d):

1. (.6,.15,1.1,.1)
2. (.9,.08,-.8,.2)
3. (.4,.25,.7,-.15)
4. (1.1,.12,-1.0,-.1)

For each use the autonomous three-state system

```
dx0/dt = -a*x0 - b*x0^3 + c*x1 + d*x0*x1
dx1/dt = -.4*x1 - .1*x1^3 + k*x2
dx2/dt = -.3*x2 + .05*x0
```

Compare k=0 (closed x0,x1 subsystem) with k=1.2 (open subsystem). The target f0 is identical and independent of x2 under both. All component fields are degree<=3, within the broad algebraic class used by the pretrained FIM; this is not a guarantee that the coefficient/trajectory distribution matches pretraining.

Two independent initial-condition seeds2026821,2026822; reset per mechanism and pair k so both regimes start from identical eight initial states drawn N(0,diag(1,1,1.2^2)). Eight clean context trajectories,256 uniformly spaced observations on[0,8]. Solve the generating system with DOP853 rtol1e-9,atol1e-11. No measurement noise in this first screen, to separate closure from measurement-noise denoising. Do not tune the four equations or seeds after results. No state/trajectory clipping or case selection on model errors.

For each mechanism/seed pair construct256 identical two-dimensional query points for both k values. Draw candidates uniformly from the common coordinate bounding box of their context convex hulls. Require membership in both 2D hulls and distance<=.1 to both context clouds after pooled training-context coordinate standardization. Use a fixed RNG derived from the pair seed. Take the first256 accepted points from at most100,000 candidates; if insufficient, report qualification failure without changing the threshold. These restrictions are coverage diagnostics, not a proof of density or identifiability. Query true drift is used only for scoring. A three-dimensional query uses x2=0; this is a fixed hypothetical state, not a future truth input.

## Methods and information groups

- Native frozen FIM-ODE conditioned on x0,x1 only; report only its first drift component. Use the C80-pinned base weights/source/runtime and native cached helper.
- Native frozen FIM-ODE conditioned on x0,x1,x2. This is a richer-information closed-system diagnostic, not a fair operational local baseline. Score its first component at query x2=0. Its superiority alone cannot establish the cause of a local-model error.
- Target-only weak degree-three polynomial ridge, using x0,x1 only. The library contains the true target family and is explicitly source-informed. It is an ordinary parametric identifiability control, not blind equation discovery.
- Target-only weak random-Fourier ridge,128 features plus bias, fixed N(0,1) frequencies/Uniform phases seed202682. This is an ordinary flexible local baseline, not a new method.

For weak fits, standardize both coordinates from fit contexts, form trapezoidal integrals on all windows of widths8,16,32 in every path, regress normalized x0 increments, and RMS-scale weak-design columns. Select ridge from{.001,.1,10} using central-state rollout MSE on the final65 visible context points after fitting first192 points. The given x1(t) validation path is an available inner-context input; no clean field labels or independent rollout targets enter fitting/selection. Refit on all256 points. No cross-case tuning. Preserve all candidate failures. Record polynomial design singular values and numerical condition diagnostics.

## Qualifications and actual simulation

Before comparative scoring, compare native cached versus direct model_forward on17 context queries, separately for2D and3D, atol=rtol=2e-5. Also compare the2D path with an explicitly zero-padded third coordinate; investigate a mismatch before interpreting dimension effects. Check source/weight hashes before/after. CPU two-thread execution and process-only efficient-core affinity keep the original C71 GPU campaign running.

Besides field errors, simulate the target component under two new declared input plans:

```
u_j(t)=.75*sin(omega_j*t)+.25*cos(.5*omega_j*t)
omega=[.8,1.5], x0(0)=[-.6,.6], t in[0,6],257 points
```

This is do(x1(t)=u_j(t)), with the full-information diagnostic queried at x2=0. The supplied future x1 values are planned interventions, not future observations. A local model predicts dx0 using its estimated f0(x0,u). Use RK45 rtol1e-4,atol1e-6,20,000 RHS-call and absolute state1e6 failure guards, no clipping. Generate exact target controlled trajectories from the fixed f0 with DOP853. These are conditional-input primitive simulations, not a free-running high-dimensional network experiment.

Report all16 context tasks,64 field-method records and128 rollout records, all errors/failures, parameters, selected candidates, timings and raw arrays. NRMSE uses the corresponding clean target standard deviation only for scoring. Preserve regime pairs and initial-condition seeds.

## Development decision

Do not infer a new gap from low scores alone. A closure-focused followup is justified only if the local FIM's closed-regime median field NRMSE<=.3; at least6/8 paired tasks have open field NRMSE>=2*closed; full-information open median<=.3; and an ordinary local control has open median<=.15 with median open/closed ratio<=1.5. Report the ordinary control families separately; do not use an outcome-selected method as a deployable selector. Other outcomes are unqualified or negative for this specific candidate.

Even a pass is only a development signature, not causal attribution or a final direction. Matched context geometry, simpler conditional priors, open-drive training, a genuinely larger composed system, controlled impulses and independent systems/noise must still be tested. Strong local identification, ICON/CompNO-style composition, neural-ODE function encoders and ICL-SysID are explicit novelty boundaries. C71 reserved evaluation and C81 reserved systems remain untouched.
