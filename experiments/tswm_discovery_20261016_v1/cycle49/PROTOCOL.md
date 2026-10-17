# C49 frozen conditional learning feasibility screen

Frozen before any formal fit or EVAL opening. No accepted research direction is implied. This is an original controlled experiment, not a full original-paper reproduction. The marked-observation transform qualification has passed; it had0 optimizer updates and no data splits.

Question: can observation-marginalized proper conditional scoring learn useful stochastic future evolution from informative, noisy sparse observations at a competitive quality/cost, compared with complete-clock variational learning of the same predictive world law?

## Data and information

Two independent2D stochastic processes: rotating OU and coupled bistable dynamics. Physical source uses Euler substeps0.05, three substeps per observation bin.48 bins per episode, first24 history/last24 future. Sources burn64 substeps from standard-normal initial states. TRAIN1024/VAL128/EVAL256 independent physical trajectories per process with disjoint seeded streams.

Known calibrated sensor: expected bin count a*(0.3+1.2*sigmoid(0.9*x0+0.4*x1)); N is Poisson. Two-dimensional Gaussian marks have mean physical x and std0.35; S is the sum of marks. Conditional on x,N: S~Normal(N*x,N*0.35^2*I). Bin acquisition is piecewise constant; no continuous-time point-process accuracy claim beyond this measurement model.

TRAIN a in{0.5,1,2} per episode; VAL a=1. EVAL a in{0.25,1,4}, paired on the same256 physical paths with independent observation randomness. Neither future counts/marks nor physical truth is an input at prediction time. a and noise calibration are known. Complete N,S and zeros enter the history encoder. TRAIN-only count/mark-sum scales normalize encoder/probe inputs. Clean future physical signals are saved separately from observation files; never used for optimization, early stopping, or parameter selection.

EVAL includes independent physical paths, not new real domains. Density contexts are paired views, not independent physical worlds. Physical decoder coordinates represent signals; latent coordinates are not claimed physically identified.

## Shared predictive model and compared objectives

GRU history encoder32 -> Gaussian initial latent8. A shared autonomous latent Markov generator uses bounded neural drift64x2, state-independent learned diagonal innovation, and a learned linear2D signal decoder. Euler step0.15 is the predictive model's discrete transition; no hidden continuous-SDE/solver-convergence claim. Same predictive core initialization and training-index schedule for both arms within a process/seed. Head initialization is ordinary neural initialization, not an oracle.

1. `rb_score`:8 iid latent future paths per training history; analytically integrate the complete calibrated marked-observation kernel; unbiased independent-pair conditional Fourier kernel score.128 fresh probes per update. Probe spans1/4/8/24 with random offsets, Gaussian weights and scales0.3/1/3. The full-span component has full frequency support. Finite computation does not guarantee finite-data identification.
2. `full_clock_elbo`: same predictive core, with an additional future-conditioned reverse GRU32, posterior initial distribution correction and posterior transition-drift correction. Same transition innovation in prior/posterior gives an exact conditional Gaussian step KL in the discrete model. Include all Poisson count terms, Gaussian mark-sum terms and initial/step KL; do not condition the likelihood on acquisition times while omitting zeros. Use4 iid posterior paths. Future recognition is training-only; forecasts use only the shared history encoder/prior. Extra inference parameters and cost are disclosed, not hidden by matching predictive parameter counts.

Conditional proper scoring, latent stochastic models, variational inference, Rao-Blackwellization and the marked-Poisson identity are existing mechanisms. The experiment tests an interaction, not an unprecedented operator. A gain against a weak event-sampling/REINFORCE control would not suffice; the primary control is a complete-clock future-conditioned variational model.

## Fit/selection and budget

Two fresh seeds2026491/2026492 x two processes x two arms =8 fits. Adam1e-3, gradient norm clip1, batch32. Maximum3000 updates, VAL every200. Stop after1500 only if six consecutive checks fail to improve the best VAL score by0.2%. Same selection criterion for both arms: fixed256-probe observed-future kernel score,32 iid predictive paths, fixed common validation randomness. No clean-state selection and no EVAL tuning. Save every validation checkpoint, selected/last models, optimizer/RNG, indices and actual timing.

This is a feasibility budget, not a full matched-compute frontier or evidence of universal likelihood failure. Final-step selections/caps/poor training fit remain explicit caveats. A future claimed method needs competent external baselines and fresh real-domain validation.

Before formal fits, qualify the changed model/likelihood path with discarded updates and check actual CPU/CUDA throughput. No pretrained/warm-start formal weights. Run one formal fit at a time on the selected device to avoid unmeasured GPU contention; C48 CPU jobs remain untouched.

## Locked evaluation contract

Seal all selected checkpoint identities before opening EVAL.128 iid predictive futures per history; report physical future joint energy score and mean MSE, and separately the observed-future proper score using a fresh fixed512-probe bank. Compare matched target horizon/units and paired physical paths. A real-prefix continuation may be reported using first12 newly observed bins, appending them to history and retaining the latest24; compare the original final12 targets. Physical truth is used only for scoring.

Interpret signs across both seeds and both processes. No iid confidence claim from3 sensor-density views. A numerical or information leak invalidates attribution. Qualified independent observation-feature/score reconstruction and raw source reproducibility are required before a final predictive claim. A positive synthetic result is only a candidate mechanism; novel framework utility still requires stronger independent-domain evidence and prior-art differentiation.

Stop/rescue: if complete-clock ELBO matches or beats rb_score at comparable total cost, deprioritize this recipe. If gains are only count fit, a finite probe artifact or expensive added simulation, the intended stochastic-future improvement is unsupported. Do not tune EVAL to rescue it.
