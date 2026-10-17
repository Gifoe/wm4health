# What the gradient control measures

Let theta=(a,log(lambda),log(b)). The recognition potential is N(z;b*x,1/lambda), multiplied by the stationary unit Gaussian prior before normalization. The augmented smoother produces moments q_theta for each target sequence independently of the reference bank. C55 provides the finite loss and analytic Gaussian-reference equations; C56 copies the native covariance-jitter and adjacent-cross convention and qualifies derivatives separately.

For a target marginal with mean m and variance C, the reference-dependent part of the log kernel can be written

    H(x') = k*m*b*x' + e*(b*x')^2
    k = lambda/(1+lambda*C)
    e = lambda^2*(C-1)/(2*(1+lambda*C)*(1+lambda))

The constant terms cancel between the positive term and reference denominator. With x'~N(0,v), tau=b^2*v and D=1-2*e*tau,

    log E exp(H) = -0.5*log(D) + 0.5*k^2*m^2*tau/D.

Replacing log(mean(exp(H))) by this expression gives an integrated-reference loss on the same targets. Its exact population expectation is the C55 objective. At zero reporting jitter and the specified true parameters, its full gradient is zero. Nonzero literal jitter is computed explicitly rather than silently treated as zero.

For each independent bank i, record g_finite,i and g_integrated,i. The mean of their paired difference estimates the finite-reference gradient deviation. The control-variate estimator

    mean(g_finite - g_integrated) + grad L_population

has the same expectation as the finite-reference gradient; it reduces Monte Carlo noise without changing the mathematical training objective. It is used for diagnosis only. Gaussian sigma points integrate the quadratic target loss and its derivatives exactly in this scalar Gaussian setting. Central differences of the separately implemented C55 expression and unchanged native-source AD supply independent qualifications.

For the fixed-posterior convention, all q moments are stopped. The invariant prior marginal is N(0,1) for every a, so the Gamma and local-factor terms have no explicit a dependence. The only a derivative is the chain KL, which is identical with finite and integrated references. Therefore the paired finite-reference a-gradient deviation should cancel identically for fixed q. This structural cancellation does not imply that recognition updates are unbiased, that q stays correct during training, or that the existing EM option solves C54.

The same-bank target prefix has the same expected gradient as averaging all targets by permutation symmetry. External and leave-one-out references change the reference law and are diagnostic controls; neither is a claimed native training option. No finite-N InfoNCE ceiling alone proves parameter inconsistency here. We test the actual native-matched objective, including its KL terms.
