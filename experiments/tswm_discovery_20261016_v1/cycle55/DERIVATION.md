# Scalar Gaussian specialization and its scope

This derives an existing objective, rather than proposing a new method. The native formulas are pinned in [RP-GSSM](https://github.com/gatsby-sahani/rp-gssm/tree/a75225d5edceb972bb872dc9a335608733567dd4). The factor parametrization and interior-bound construction are described in Sections3.1–3.2 and AppendixC of [Maximum-Likelihood Learning of Latent Dynamics Without Reconstruction](https://proceedings.mlr.press/v306/hromadka26a.html). The scalar argument below does not invoke the paper's full-image identifiability theorem.

Let the true observation law be Gaussian with covariance S_star=R(a_star)+r*I, where R(a)[s,t]=a^abs(s-t), and v=1+r. Its one-time marginal p0 is N(0,v). The recognition potential is f_delta(z|x)=N(z;b*x,1/lambda); the full normalized recognition factor is proportional to p(z)*f_delta(z|x), with p(z)=N(0,1).

## Exact proper model under a Gaussian reference law

The full recognition factor is N(z;lambda*b*x/(1+lambda),1/(1+lambda)). Its reference mixture over p0 is Gaussian with variance

F_var = 1/(1+lambda) + (lambda*b/(1+lambda))^2*v.

The normalized conditional p0(x)*f(z|x)/F(z) is exactly Gaussian. Its observation-noise variance and slope are

R_eff = [1/v + (lambda*b)^2/(1+lambda)]^(-1),

C_eff = R_eff*lambda*b.

Consequently, the proper RPM observation law has covariance S_theta=C_eff^2*R(a)+R_eff*I. Its expected negative log likelihood relative to the product of true one-time marginals, per timestep, is

L_proper(theta) = [logdet(S_theta) + tr(S_theta^(-1)*S_star) - T - T*log(v)]/(2*T).

The candidate true point is (a,lambda,b)=(a_star,1/r,1). At this point R_eff=r, C_eff=1 and F_var=1. These identities are algebraic and also checked numerically.

## Independent expectation of the implemented auxiliary objective

Let J(a) be the stationary AR1 prior precision and V=[J(a)+lambda*I]^(-1). The augmented smoother mean is m=L*x with L=lambda*b*V. For the ideal convention its covariance is V. For the source reporting convention, add epsilon=1e-5 to each marginal variance and recompute adjacent cross covariance as (V[t+1,t+1]+epsilon)*RTS_gain[t], matching the unmodified source. The latter detail caused the preserved version1 qualification failure; version2 includes it.

For a target marginal q=N(m,C), write U'=b*X' for an independent reference at that time, with variance tau2=b^2*v. The source pair score differs by a target-only constant from

h(m,U') = log N(U';m,C+1/lambda) - log N(U';0,1+1/lambda).

That constant cancels between the positive score and its reference normalizer. Set

k=lambda/(1+lambda*C), e=lambda^2*(C-1)/[2*(1+lambda*C)*(1+lambda)], D=1-2*e*tau2.

When D>0, the reference exponential integral is closed form. After cancellation of its target-only term, the expected positive-minus-log-reference contribution is

E_gamma = k*b*Cov(m,X_t) + e*tau2 + log(D)/2 - k^2*Var(m)*tau2/(2*D).

The factor penalty is

E_KL(q || normalized(q*f_delta)) = [lambda*C-log(1+lambda*C)+lambda^2*E(m-b*X_t)^2/(lambda+1/C)]/2.

The chain penalty uses the ordinary Gaussian Markov KL computed from the marginal/adjacent moments. All outer quadratic expectations follow from S_star. The centered population loss is(-sum(E_gamma)+sum(E_KL_factor)+E_KL_chain)/T. Centering subtracts log(N) from the finite-bank negative objective; it removes a parameter-independent constant only. Independent finite native loss, Gaussian quadrature and target sigma-point checks precede any numerical minimization.

## A limited global-optimum argument

For zero reporting jitter and finite normalizer integrals, the auxiliary objective is a lower bound to the proper model log likelihood. Therefore its centered negative population objective obeys

L_aux(theta) >= L_proper(theta) >= L_proper(theta_star).

At theta_star the reference mixture equals the prior, so the interior bound is tight. The augmented smoother is then the actual Gaussian posterior, so the remaining variational gap is also zero. Thus

L_aux(theta_star)=L_proper(theta_star)=[logdet(S_star)-T*log(v)]/(2*T).

This establishes an attainable population minimum for the ideal scalar setting. With T>=3 and a_star nonzero, the proper observation covariance also identifies a from its lag2/lag1 covariance ratio. This is a separate elementary Gaussian argument; the constant-precision recognition map does not automatically satisfy the original theorem's open-image condition.

The literal reporting-jitter implementation has an approximate posterior even at the true point and is assessed separately. Bounded numerical multi-start checks verify arithmetic and local behavior; they are not a proof over arbitrary nonlinear encoders, finite datasets or minibatch reference estimates. If the identities and source qualification hold, a claim that the observed C54 deficit must arise from an unavoidable asymptotic scalar objective bias should be rejected. Finite-reference estimation and coupled optimization remain competing explanations.
