# C83 — Composing local pretrained fields into larger controlled systems

Frozen before generating C83 trajectories or scores. This is an original cross-contract development experiment, not reproduction of FIM, UNICON, GICON, or CHOP. No final research gap is assumed.

## Question and competing explanations

C82 supported target-only use of FIM under an open neighborhood on simple low-dimensional systems. Test whether independently inferred local fields can form a larger autonomous simulator and propagate a declared intervention. H1: the pretrained prior gives useful composition with limited noisy context. H2: ordinary local identification explains any benefit. H3: local errors compound or extrapolate in global simulation. H4: implementation or integration errors explain apparent failure. Native-interface equivalence, true-field integration, common data and strong local controls address H4/H2. A successful toy screen alone establishes neither novelty nor general latent-state transfer.

## Frozen carrier and information

Two system families, two parameter settings per family, two initial-context seeds (2026831, 2026832), and noise levels 0 and .03: 16 context tasks. All coordinates and each target's exact local parent set are given to every method. Graph discovery, hidden states, unknown sensors and observational causal identification are outside this screen. Parameters/equations are withheld from fitted methods; polynomial degree is explicitly source-informed for the polynomial family.

FitzHugh–Nagumo directed ring: six units, 12 states. For unit i with cyclic predecessor j,
`dv_i = v_i - v_i^3/3 - w_i + g*(v_j-v_i) + I_i`,
`dw_i = eps_i*(v_i + .7 - .8*w_i)`.
Settings `(g,I_base)=(.15,.3),(.4,.6)`, `I_i=I_base+.1*sin(i)`, `eps_i=.15+.02*i`. Parent orders are `(v_i,w_i,v_j)` and `(w_i,v_i)` respectively.

Repression ring: seven positive scalar states, `dx_i=alpha_i/(1+x_j^3)-beta*x_i`, predecessor j. Settings `(alpha_base,beta)=(2,1),(4,.8)`, `alpha_i=alpha_base*(1+.05*sin(i))`. Parent order `(x_i,x_j)`. This rational family prevents treating the exact polynomial library as universal.

Each context: four independent initial states, 256 points on [0,8], DOP853 rtol1e-9 atol1e-11. FHN v initial Normal(0,.7), w Uniform(-.5,.5); repression initial Uniform(.5,2). Add independent Gaussian observation noise with std `.03 * clean context coordinate std`; clean/noisy tasks share trajectories and initial states. No denoising using hidden clean targets. Test initial states are an independent fixed generator seed `context_seed+10000`; two initial states per task, each with a free and clamped run. Clamp `do(x_0(t)=x_0(0)+.5*sin(.8*t))`; known plan replaces only coordinate 0, all other coordinates evolve from the fitted coupled field. Never feed future factual states to predictors.

## Methods and selection

1. Unchanged pretrained FIM-ODE (OpenFIM cee2bb53be1e8e92b73f98dfdbe3605efc9119cc; HF 00875c253d025dd33fd70ae751089b1be3a7c174), local contexts padded to dimension3, use only each target's first output. Batch local queries for efficiency. Before scores, compare every local target at 17 context locations with original `make_fim_vf_fn`, atol2e-5 rtol2e-5. Qualify true-field solver separately. No neural weight updates.
2. Local weak degree3 polynomial regression, all monomials through degree3, normalized coordinates, 20 features for 3 parents and 10 for 2. Strong source-informed family control for FHN.
3. Local weak random Fourier features, 128 cosines plus bias, seed202683, dimension-specific fixed Gaussian frequencies. Same context, normalization, local graph and integration as polynomial.

Both ordinary methods fit integrated state-increment equations at widths8,16,32 with trapezoidal quadrature and column RMS scaling. Ridge candidates .001,.1,10. Fit first192 context observations; choose one global ridge for the composed model by four full coupled validation simulations on last65 observations, initialized at observation191. Validation compares to observed noisy values. Failed validation is recorded and cannot win. Refit all256 observations at selected ridge. If all candidates fail, preserve failure and do not substitute a true-field model. Native FIM uses all256 context observations and has no target-task tuning. Pretraining cost is additional information and compute, not a matched training budget.

## Evaluation and guards

192 free/clamped model trajectories (16 tasks ×3 methods ×2 independent initial states ×2 regimes). 257 points on [0,8]. RK45 rtol1e-4 atol1e-6 for all methods; declared 20,000 RHS call or absolute-state1e6 guard, no clipping. Preserve partial/failed predictions and messages. True equations generate reference trajectories with DOP853; independently run true field with evaluation solver to rule out numerical reference mismatch. A fixed diagnostic set of128 points sampled without replacement from the clean context union reports target-field errors; used only for reporting after model selection. It is not a general off-support evaluation.

Report global trajectory NRMSE, per-coordinate normalized error, unclamped-coordinate error, paired intervention-response error, local field error, finite completion, timings and context-support distance along both reference and predicted trajectories. Primary trajectory score and gate use the mean of per-coordinate `RMSE/std(reference coordinate)`, denominator floored at1e-12. Exclude clamped coordinate from clamp score. The intervention-response score compares predicted clamp-minus-free trajectories to the same difference of reference trajectories, excluding coordinate0. Use failed-record counts alongside finite medians; never drop failures silently. Nearest-neighbor distances are descriptive coverage, not identifiability certificates. No test-based hyperparameter selection or noise/seed resampling.

## Development decision, not paper acceptance

Continue the generic local-foundation composition route only if native FIM completes all trajectories, median unclamped NRMSE <=.3 in each family/noise group, and beats both ordinary controls' task-level median NRMSE by >=25% in both noisy families without >25% clean-group degradation. Otherwise report the capability/failure and do not claim generic stitching as a new method. This heuristic development gate cannot establish novelty. A failure against a source-informed baseline alone is not proof that pretrained inference is useless. Any new repair requires a separate frozen protocol and closest-work check.

All raw context, models and prediction arrays remain on server15TB project subdirectories. Publish authored code, source/asset hashes, compact metrics, plots and reports to Gifoe/wm4health. Preserve earlier work and unrelated jobs. Independent checks recompute identities, scores, coefficients, native field queries and references before publication.
