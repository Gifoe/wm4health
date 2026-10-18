# C85: do existing repairs solve an explicitly anchored shift task?

Frozen before source training, target fitting or test generation. This is a cross-task transfer of existing components and an original controlled experiment. It is not a reproduction of robot-policy or JEPA-planning paper results, and introduces no new method.

## Task and scientific endpoint

Learn a source simulator, then calibrate it once from limited target trajectories and eight paired static physical calibration states. The target observation map is introduced only after source training. Evaluate independently generated observable trajectories and differences between two physically specified action plans. Do not score an unanchored internal component label. The paired calibration data are explicit additional information, available to all methods.

Four cases: neither change; observation only; dynamics only; both. Source observations are the full two-dimensional physical state standardized using source TRAIN mean/std. The changed sensor map is g(s)=R(0.5) diag(0.8,1.2) [s+0.15 tanh(s)] + (0.2,-0.15). It is invertible; this formula is used only by data generation and not supplied to any fitted neural or dictionary model. Static anchor states are eight fixed source-standardized points sampled uniformly from [-1.5,1.5]^2; their target measurements include the same observation noise as target trajectories.

Two mechanisms, dt=0.05, RK4, dimension two:

* Controlled Duffing: q'=v; v'=-kq-cv-0.2q^3+0.7u. Source k=1,c=0.4; changed k=1.5,c=0.2. Independent initial q,v uniform [-0.8,0.8].
* Controlled Brusselator: x'=1-(b+1)x+x^2 y+0.2u_1; y'=bx-x^2 y+0.2u_2. Source b=1.5; changed b=2.4. Independent initial x in [0.5,1.5], y in [1,2].

Commands are uniform [-1,1], held for eight steps. Actuation units/interfaces remain fixed. The physical coordinate calibration and named source/target observation channels are specified. Target noise is additive Gaussian std 0.01 in standardized observation units. Source trajectories are noiseless. EVAL starts from a noisy initial measurement, shared within each action pair, and scores against the deterministic noise-free observable future mean. No model receives target physical states except the eight static anchors.

Seeds 2026851 and 2026852 are independent data/model seeds for an initial falsification screen, not evidence of stable improvement. Source TRAIN:128x64 transitions; source VAL:16x128 transitions. Target TRAIN:4x64; target VAL:2x64; target EVAL:8 initial states, two action plans each,128 steps. Source standardization uses source TRAIN only. Separate deterministic seed namespaces define every partition. Target EVAL is generated only after all candidate fits and a validation-only selector have been saved for that condition.

An EVAL action contrast shares the first32 actions, then increases the first control coordinate by0.5, clipped to the allowed action bounds, during steps32:64. The state evolves freely after action changes. Both members have the same initial physical state; no future observed state is fed back to any simulator.

## Native source and adaptations

Use Cross-Embodiment Transfer commit3988511598a39c8a8a6530055b72609bf1b8b373 from the C84 pinned cache. BCObsAgent supplies the actual encoder, decoder, forward and inverse networks, two hidden layers of64 units, latent dimension2. Its unused policy head is neither fitted nor evaluated. This size and task differ from the paper. The pure neural utilities are extracted by AST from the pinned utils.py to avoid importing uninvoked robotics dependencies; no function body is reimplemented or environment stub provided.

Source warmup:2000 calls to the unchanged native update_dyn_cons, lr3e-4, batch128. Then a standard observable-output-error polish uses8-step free prediction plus0.1 reconstruction and0.001 inverse-action loss, Adam3e-4, batch32. Evaluate source VAL after2000,4000,8000,16000 polish updates. Stop at the first gate passing both median H32 NRMSE<=0.15 and H128<=0.35. Retain the best validation checkpoint if none passes, mark source capability unqualified, and do not interpret target transfer failure for it. Source training is intentionally simulator-oriented; it is not the paper's behavior-cloning protocol.

For each qualified source, compare:

1. Frozen source simulator.
2. Native ObsAligner with frozen source dynamics, default cycle/dynamics/WGAN-GP weights, source-copy target initialization, plus a common paired-anchor step. Run1000 native discriminator/generator iterations and1000 anchor updates:2000 target encoder/decoder updates. This paired extension is reported explicitly, not called the unchanged paper method.
3. Ordinary joint output-error adaptation of the same source encoder, decoder and forward model:2000 updates,8-step observable prediction +0.1 reconstruction +10 anchor loss +1e-4 mean squared displacement from initial parameters. The inverse/action-policy heads stay fixed. No adversarial loss or new architecture.
4. Ordinary output-space delta-map regression: degree<=3 polynomial features and128 cosine features plus linear terms/bias, each with ridge lambda in{1e-6,1e-3,0.1}, source-coefficient prior strength in{0,1e-3,0.1,10}. Source dictionaries are fitted from the same source TRAIN. Target coefficients use target TRAIN. Static anchors are available but unused by this baseline.

Neural adaptation lrs{3e-4,1e-3}; checkpoints at0 and every200 encoder/decoder updates. Candidate/model/checkpoint selection uses mean target VAL H32 observable NRMSE only, never EVAL. Save all candidate summaries and selected checkpoints. Source/target information, update counts, trained parameter counts and wall time must be recorded. These methods do not have identical trainable parameter counts or computational cost; no matched-compute superiority claim is allowed. Every method family is selected on the same target VAL, and a combined best-existing selector is also saved before EVAL.

## Metrics and decision

Report all EVAL H32/H128 trajectories. NRMSE is coordinate-mean RMSE divided by that reference trajectory coordinate's std, floored at0.05 in normalized observation units. Action-response error is RMS error of the predicted difference divided by RMS true difference floored at0.05. This is a common observable action interface, not a named-latent intervention. Nonfinite or abs(state)>1e4 rollouts are failures, counted explicitly without clipping or dropping them from the primary aggregate. Numerical reference is independently checked by DOP853.

The development question receives no support in a case if the validation-selected existing repair already has median H128 NRMSE<=0.2 and median response error<=0.3. A failure only motivates further analysis: source competence, coverage, finite-data observability, optimization, model capacity and existing active/joint identification must still be excluded. Frozen alignment failing under true dynamics change is expected outside its invariant-dynamics premise and cannot itself establish a research gap. Two source seeds and two controlled systems cannot establish broad generality or a final paper direction.

All raw data, weights, prediction arrays and logs stay on the server data disk. Only authored code, configuration, compact verified evidence and reports go to the authorized GitHub repository. No local result-payload synchronization.
