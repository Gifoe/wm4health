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
