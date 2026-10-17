# C49 completed literature boundary

# C49 development note: observation-marginalized conditional world learning

Status: exploratory, no accepted final direction, no formal trained new-method result. C48's original16 fits remain live and unchanged. This note is a different learning-objective rescue of C47, not a test-set retune of its fixed-feature/grid recipe.

Research unit remains a reusable time-series world-model framework. Known components can contribute through a distinct useful learning interaction; an unprecedented mathematical primitive is not required. A new name for a standard scoring rule is insufficient.

## Source-backed boundaries checked on 2026-10-17

- [NeurIPS2023 signature-kernel Neural SDE training](https://papers.neurips.cc/paper_files/paper/2023/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html), particularly paper section3.4, already learns conditional future path laws from past trajectories with a proper kernel score. [Author implementation](https://github.com/issaz/sigker-nsdes) explicitly supplies a conditional notebook. Nonadversarial conditional path-distribution training and the U-statistic correction are existing methods, not this project's novelty. Source reviewed; not executed as a reproduction in this round.
- [AmbientGAN, ICLR2018](https://www.cs.utexas.edu/~ecprice/papers/ambientgan.pdf) and [AmbientFlow](https://arxiv.org/html/2309.04856v2) already learn latent object distributions through known noisy/lossy measurement operators. AmbientFlow section3 introduces a variational posterior network and states measurement-identifiability conditions. Measurement marginalization does not automatically identify a physical latent state.
- [Exact Ambient Diffusion](https://arxiv.org/html/2404.10177v2), section3, already uses a consistency objective and independent-sample unbiased estimators to recover clean distributions under its specified additive-noise assumptions. Generic consistency or two-sample debiasing cannot establish our novelty.
- [JMLR2018 sigmoidal Gaussian Cox inference](https://www.jmlr.org/papers/volume19/17-759/17-759.pdf), AppendixA, covers the marked-Poisson characteristic-functional identity. The analytic identity is classical.
- [High Rank Path Development](https://arxiv.org/abs/2405.14913) studies filtration-sensitive process distances and conditional generation. Information growth and conditional process learning are occupied research areas; abstract inspected, no complete-source reproduction claimed.

## Concrete rescue mechanism to investigate

A learned history encoder defines a conditional initial-state distribution. Shared neural stochastic dynamics generate latent future paths; a decoder gives a signal path. The observation model may create noisy readings at state-dependent random times. Score the conditional observed-future distribution while analytically averaging over measurement randomness given each generated signal path. Simulate latent process randomness, but not discrete future observation events during gradient computation.

For a piecewise-constant marked Cox observation model, let each bin have rate r_t, count N_t, Gaussian marks with mean mu_t and covariance Sigma. Let S_t be the sum of its marks. For count probe alpha_t and mark-sum probe omega_t:

    E[exp(i sum_t(alpha_t N_t + omega_t^T S_t)) | signal path]
      = exp(sum_t r_t (exp(i alpha_t + i omega_t^T mu_t
                              - 0.5 omega_t^T Sigma omega_t) - 1)).

For M independent latent futures with conditional feature vectors phi_m, use

    (|sum_m phi_m|^2 - sum_m |phi_m|^2) / (M(M-1))
      - 2 Re(mean_m phi_m * conjugate(observed_feature)).

Average over probes. Fresh probes from a suitable full-support distribution define a kernel in expectation; a finite frozen probe bank alone does not imply strict identification. A known observation model is an assumption requiring calibration or sensitivity analysis. State-dependent acquisition differs from exogenous missingness and must be modeled jointly.

Potential interaction: conditional state inference and stochastic dynamics might be learned without a future-conditioned variational smoother while retaining information conveyed by observation timing. Whether this improves quality or cost over competent likelihood/variational models is unknown. Rao-Blackwellization and conditional kernels individually are old; evidence must demonstrate a useful interaction under actual partial observations.

## Competing explanations and stop conditions

H0: complete likelihood/strong variational inference or an ordinary conditional kernel world model already provides equal or better quality per total cost. This would remove the current improvement claim.

H1: analytic integration reduces observation-induced gradient variance enough to improve learned future laws under sparse informative observations. Qualify the estimator first; then compare fully trained shared-core models, with a strong future-conditioned variational baseline and matched-cost frontier.

H2: any apparent gain is only a finite-ensemble bias, inadequate likelihood optimization, a chosen frequency bank, known-emission privilege, or weak event-sampling baseline. Use an unbiased estimator, frozen TRAIN/VAL choices, complete clock likelihood, and identical information/measurement calibration. A REINFORCE variance comparison is a qualification, not a sufficient scientific baseline.

H3: observed-law matching fails to recover relevant future queries because the observation operator is nonidentifying. Report observable predictive utility; do not claim physical-state identification from matching alone.

## Immediate authorized action and evidence boundary

Implement and execute the analytic marked-observation characteristic function and independent-sample score qualification on the server. Verify against independently enumerated Poisson/Gaussian expectations, finite differences and finite-ensemble expectation identities. No EVAL tuning, new architecture claim, new trained-method claim, or formal research-gap acceptance follows from this qualification.

If the computation qualifies, the next formal protocol must use fresh trajectories/data, learned history-conditioned states and neural dynamics, and independently held-out future-law scoring. C47's EVAL cannot select its probes, weights or hyperparameters. C48's mean-forecast comparison remains a separate source-baseline screen.


# C49 current observation-process boundary check

Retrieved2026-10-17 while formal fits continued, before EVAL opening. Exploratory primary-source review, not original-paper reproduction or evidence of absence of other work.

| Primary source | Verified scope | Implication for this project |
|---|---|---|
| [EndoClock](https://arxiv.org/html/2608.09266v1), Aug2026 | Audits where state/acquisition evidence survives synchronization: values, cell summaries, native timing, or external-only channels. It explicitly distinguishes a structural no-witness condition from failure to detect a statistical signal. | Keeping observation counts/timestamps is established motivation. Lost information cannot be recreated by a new downstream loss. Our binned count/sum model retains the sufficient statistics specified by its own synthetic sensor; it does not recover arbitrary sub-bin information. |
| [Design-Ignoring versus Design-Respecting World Models for Epidemiology](https://arxiv.org/html/2609.30679v1), Sep2026 | Holds a simple latent structure fixed while changing an observation likelihood to respect a cluster-randomized test-negative design. It evaluates sensitivity to sampling changes; it also states that its conditional enrollment model does not identify absolute incidence. | An observation-aware model is not automatically novel. Its causal contrast and design assumptions differ from our passive conditional-future problem; do not equate those tasks or claim our simulator establishes causal effects. |
| [World Models Under Asynchronous Sensor Observations](https://arxiv.org/html/2609.07299v1), Sep2026 | Studies staleness/time-to-refresh features under several degrees of coupling between refresh schedules and system/control behavior. Reported schedule utility depends on its causal role. | Generic schedule features are occupied prior art. Future sensor-density metadata is permissible in our specified calibrated sensor, but unknown future measurements are not. Predictive history information changes with density, so sparse-sensor errors alone do not establish a new gap. |

The current candidate's differentiating hypothesis remains narrower: whether analytic observation averaging inside conditional stochastic-world training improves a shared predictive core beyond complete-clock variational training at useful total cost. This must survive actual held-out future-quality comparisons. The sources above motivate careful information contracts; they do not prove this particular objective novel or necessary.

If the screen is positive, the next controls should test approximate-inference limitations and a fair quality/cost frontier before any framework claim. Fresh real-domain evidence and comparison with conditional kernel-generative and observation-aware latent models remain required.


# C49 closest-overlap challenge: noise-integrated kernel generative learning

Primary sources checked2026-10-17 while the fresh formal fits were running. No empirical result from these sources has been reproduced in this supplement.

- [Convolutional Maximum Mean Discrepancy for Inference in Noisy Data](https://arxiv.org/html/2604.12022v1), Apr2026, compares model distributions after known noise convolution, relates translation-invariant kernels to noise-smoothed kernels, and estimates model parameters under measurement error. Its stated framework centers on additive error independent of latent truth.
- [Nonparametric Deconvolution and Denoising using Simulation Based Inference](https://arxiv.org/html/2606.21907v1), Jun2026, learns latent generative distributions using convMMD and uses them for denoising. A noise-aware likelihood-free latent generative objective is already an existing research route.
- [OASIS](https://arxiv.org/html/2606.22572v1), Jun2026, puts measurement/selection transformations into simulations and uses observed-distribution MMD to construct a pseudo-posterior over simulator parameters. A known observation pipeline inside distributional matching is also occupied.

## Exact overlap in the implemented objective

Let K(do|x) be a specified observation kernel and k(o,o') the Fourier-feature kernel. Define

    k_bar(x,x') = E_{o~K(.|x), o'~K(.|x')} k(o,o')
    k_cross(x,y) = E_{o~K(.|x)} k(o,y).

The implemented score is the ordinary observed-law kernel score

    E_{x,x'~P_theta(.|history)} k_bar(x,x')
        - 2 E_{x~P_theta(.|history)} k_cross(x,y),

estimated with independent latent futures. The analytic marked-Poisson characteristic function computes these expectations. This is an observation-kernel pullback/integration of an existing scoring rule; the algebra and U-statistic are not new mathematical contributions.

Possible differentiation is a useful *conditional state/dynamics learning interaction* under informative sparse observations, without a training-only future smoother. Merely adding a time axis, using a Cox sensor instead of additive Gaussian noise, or renaming convMMD would be insufficient by itself. Finite-data future-quality/cost improvements, a discriminating mechanism ablation, competent inference controls and fresh real-domain evidence would be necessary to establish the value of that interaction.

The current two-arm experiment changes both training objective and use of a future-conditioned variational recognizer. A gain cannot by itself be attributed solely to analytic averaging rather than objective geometry or approximate-inference constraints. If promising, distinguish these explanations before claiming a new mechanism. If ordinary complete-clock inference already covers the behavior, deprioritize the recipe.


# C49 scientific decision

Decision: DEPRIORITIZE_CURRENT_RECIPE

Accepted final directions: 0

The current observation-marginalized conditional-score recipe does not provide a stable general quality/cost advantage over complete-clock variational learning of the same predictive core. This completed screen supports lowering its priority. It does not rule out conditional stochastic-world modeling or observation-aware learning generally.

## Verified primary evidence

Eight fresh fits /18800 updates.16 raw evaluation groups /48 summary groups /12288 model-context-reveal scores;512 independent synthetic physical paths,1536 paired sensor-density histories,128 joint future samples per prediction. No independent real domain was evaluated.

Equal-weight means across the three pre-specified sensor densities and two fitted seeds, open-loop24-bin physical energy (lower is better):

| Process | Observation-averaged score | Complete-clock ELBO | Score minus ELBO |
|---|---:|---:|---:|
| Rotating | 0.2984525914 | 0.2936971191 | +0.0047554724 (+1.62%) |
| Bistable | 0.3071688372 | 0.3024564767 | +0.0047123605 (+1.56%) |

Rotating differences are adverse in both fitted seeds: +0.00546158 and +0.00404937. Bistable changes sign: +0.01330112 and -0.00387640. Updated12-bin means also do not favor the score recipe, but before/after update benefits must use identical final12 targets; full24 versus12 is not a valid update comparison.

All original future predictions, noise draws and energy/MSE/observed scores were independently replayed. Max model discrepancy1.59899e-4 under pre-frozen atol/rtol3e-4; energy arithmetic4.44e-16; observed-score discrepancy1.82e-7. Native predictive initialization and training-index schedules match within case/seed. Verification optimizations changed I/O, equivalent numerical evaluation and independent-group scheduling only; primary fits/scores were not rerun.

## Pre-registered fixed-time evidence

The60/120/180-second rule was sealed before primary EVAL opening. All72 summary groups /18432 model-context-budget entries were reported;22 unique weight evaluations, reusing identical verified primary results when appropriate. New frontier predictions underwent independent model/energy/MSE replay.

| Process | 60s mean energy delta | 120s mean energy delta | 180s mean energy delta |
|---|---:|---:|---:|
| Rotating | +0.00668030 | +0.00363422 | +0.00475547 |
| Bistable | +0.00907795 | -0.00169634 | +0.00471236 |

Negative deltas favor the score recipe. Bistable120s is a small favorable mean; its episode-bootstrap interval[-0.00795610,+0.00463150] crosses zero. These intervals keep density views together and condition on the two fitted model seeds; they are not uncertainty across training seeds or real domains. No single budget is selected from EVAL.

## Preserved exploratory rescue, not a selected method

Three of18 two-seed-mean process/budget/density cells favor the score recipe, all at density0.25: rotating60/120s and bistable120s. Some sparse-budget differences agree in both seeds, but the benefit does not persist uniformly with training budget or process. Preserve this as a hypothesis about sparse-observation calibration/selection, not as a demonstrated new mechanism or main contribution.

If an independent real-domain failure later makes this hint important, first distinguish ordinary observation-coverage augmentation, validation-policy choice and approximate-inference constraints from analytic averaging. A calibrated thinning-augmentation or competent observation-aware filter is a necessary ordinary control. Do not tune loss weights or select the120s/density0.25 subgroup on this EVAL.

## Novelty and attribution

The score is a pullback/integration of an existing kernel score through a specified observation kernel. Conditional Neural SDE scoring, Ambient methods,2026 convMMD/deconvolution generative learning and observation-aware SBI already cover major ingredients. A useful distinct learning interaction could still contribute, but this screen does not demonstrate one. The two-arm comparison also changes objective geometry and training-only future recognition, so it cannot isolate analytic averaging as the causal explanation of a difference.

One maximum-budget fit and three final-step selections limit convergence claims. Two controlled2D process families with a known calibrated sensor do not establish broad real-world utility or physical latent identification.

## Next action

Return to source-grounded real-data failure discovery before adding another loss. The pinned native TSWM Greenhouse preprocessing has now failed an input-boundary contract on a constructed window accepted by its own missingness filter: changing only future target values changes the imputed observed history. Measure real-data prevalence before interpreting a baseline comparison; preserve original-protocol reproduction separately from any information-safe correction. This is an implementation issue, not a new research direction.

No final research direction is accepted. Overall discovery goal remains ACTIVE_UNACHIEVED.

