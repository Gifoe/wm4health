# Score selection across learned targets: exact construction and limits

All claims here concern choosing representations, not propriety on a fixed target. This is not a counterexample to Branch-JEPA's fixed-target population theorem, its full EMA/decoded/reconstruction objective, or Info-LDM's stated information-saturation assumptions.

Let R be a planar rotation and X stationary standard Gaussian in2D. The process is

```
Y = Q*rho*R*X + sqrt(1-rho^2)*epsilon,
P(Q=-1)=p, epsilon~N(0,I), independent.
```

For0<rho<1 it is a stable, positive-density, ergodic Markov chain: each transition contracts two copies under commonQ/epsilon byrho. Its marginal is exactlyN(0,I), but its conditional law is a mixture, not a Gaussian unless the mixture degenerates. All state coordinates are observed.

The map phi(r*cos(theta),r*sin(theta))=(r*cos(2theta),r*sin(2theta)), withphi(0)=0, has two inverse branches and Jacobian determinant2 away from the origin. It preserves the standardGaussian marginal exactly: radius is unchanged and a uniform independent angle remains uniform after doubling. It is globally Lipschitz (a finite bound suffices), is two-to-one, and obeysphi(X)=phi(-X). Both learned-target candidates have equal dimension, scale and exact frame marginal. For this symmetric noise model, the projected process is also Markov, and its kernel is independent ofp. This removes an insufficient-history explanation for the oracle comparison.

For a fixed target lawP, the Bayes Energy risk is0.5*E||Y-Y'|| withY,Y' conditionally independent given the context. This is the score of the correct conditional law, not an approximation caused by a small branch count.

## Exact strict preference exists

At the stationaryrho=1 boundary, conditional onX=x, the future is supported on+Rx and-Rx. Its Bayes Energy risk is2p(1-p)||x||. The folded future is deterministicphi(Rx), giving zero Bayes Energy risk. Thus for every0<p<1, the fold strictly lowers the latent criterion despite its information loss. The same result holds with unit-normalized targets, where the identity risk is2p(1-p) and folded risk remains0.

The boundary is used only for an analytic existence argument; it is not the executed positive-noise chain or an ergodic example itself. Asrho tends to1 from below, conditional samples converge almost surely, all radial distances have integrable Gaussian envelopes, and unit distances are bounded. Dominated convergence therefore preserves a strict risk advantage for somepositive-noise, ergodicrho<1. Numerical values at the separately frozenrho=.9 grid are empirical evidence, not a claimed exact threshold from this continuity argument.

## What predictive information is lost

For0<=p<.5, the original one-step raw conditional mean is(1-2p)*rho*R*X. Given onlyphi(X), the two initial sign fibers have equal posterior probability, and the raw conditional mean is0. The extra raw BayesMSE is exactly

```
2*rho^2*(1-2p)^2.
```

At the executedrho=.9,p=.25 this equals.405. Atp=.5 it vanishes, giving an explicit null where the sign is not predictively relevant.

The folded conditional density is the sum of the densities at its two preimages divided by2. Evaluated atphi(y), it is

```
0.5 * [N(y;rho*R*x,(1-rho^2)I) + N(y;-rho*R*x,(1-rho^2)I)].
```

The original conditional has weights1-p,p. Since both frame marginals have identical Gaussian differential entropy, the one-step conditional-NLL difference equals the loss of predictive mutual information. Existing likelihood/information principles thus expose the lost state distinction. This is not a new estimator.

## Limits on a research claim

The exact construction says that marginal anti-collapse plus fixed-target properness does not imply representation sufficiency. Properness is not violated: each score predicts its own encoded target correctly. No neural training failure follows solely from the existence of this lower-risk representation. Stop-gradient/EMA training has a different vector field from full-gradient global-risk minimization; existing theory already studies this distinction. Full actual methods may ground their latent in observed space, use information objectives, or have optimization biases that retain state. These are mandatory controls. The random-initialization learning study is retained separately and decides whether this example has demonstrated practical importance.
