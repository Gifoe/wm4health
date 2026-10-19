# Restricted confidence certificate: assumptions and derivation

This is an executable application of classical Gaussian regression confidence
sets and Taylor/reachability bounds. No new general verification theorem is
claimed. `scripts/confidence_tube.py` implements the formulas below in float64.
Floating-point roundoff is checked numerically; this implementation does **not**
use outward-rounded interval arithmetic, so it is not a machine-checked formal
certificate including all numerical errors.

## Exact scope

Two-dimensional fully observed state, independently randomized training
transitions, features `phi(x,a)=[x0,x1,a,a*x0,a*x1,1]`, conditional mean
`f_W(x,a)=W^T phi(x,a)`, known isotropic Gaussian innovation scale sigma.
Training design has full column rank. Inference uses a complete observed
factual path under a=0; the alternative uses a=1. The structural sensitivity
family is one global angle theta, with independent standard-normal innovations
and innovation map `sigma R(a*theta) epsilon`.

The same theta applies across time and episodes in each SCM. The same W is used
in both branches and at all times. A query-specific interval is the union over
this same global-angle family; different interval extrema may be attained by
different globally valid SCMs. This does not allow selecting per-step rotations.

Every model in this family has the same conditional transition kernel for a
fixed W. Thus it also has the same prospective path distribution under any
non-anticipating policy. What differs is the retrospective cross-world coupling.
It is not an all-SCM class, not partially observed, not general non-Gaussian,
and not a guarantee for an estimated neural conditional mean.

## Regression event

Let Z be the N by p training design, p=6, d=2, P=(Z^T Z)^(-1), and W_hat OLS.
Under the correctly specified known-variance model, conditional on Z,

`||P^(-1/2) (W_true-W_hat)||_F^2 / sigma^2 ~ chi2(p*d)`.

Set `r = sigma sqrt(chi2_{12}(1-alpha))`. A single event of probability 1-alpha
covers the entire W matrix. It is not a separate 95% event for each query.
Conditioned on this event, every trajectory and theta is covered by the outer
bounds derived below. There is no query-count Bonferroni penalty because the
event is uniform over features and paths. The known variance assumption is
material: using an unchecked residual estimate in r voids this guarantee.

## Retrospective recurrence and exact fixed-kernel family

For observed x_f[0:H], the abducted physical innovations under W are
`eta_W[t]=x_f[t+1]-f_W(x_f[t],0)`. The counterfactual recurrence is

`z_W,theta[t+1]=f_W(z_W,theta[t],1)+R(theta) eta_W[t]`.

Because the drift is affine and theta is shared, for fixed W the whole path is
`z[t]=d[t]+c[t]cos(theta)+s[t]sin(theta)`. Three linear recurrences compute these
coefficients. The terminal first-coordinate effect interval is exactly

`d_H[0]-x_f[H,0] ± sqrt(c_H[0]^2+s_H[0]^2)`.

This is exact only within the declared family. It is an ignorance/sensitivity
range for the fitted kernel, not a finite-sample confidence interval.

## Basic norm tube

Write `DeltaW=P^(1/2) G`, `||G||_F<=r`. Let M map a state perturbation to its
features at a=1, so `Delta A1=DeltaW^T M`. Then

`q=r ||P^(1/2)M||_2`, `rho=||A1_hat||_2+q`.

At estimated state z_hat(theta,t), define whitened features
`v_c=P^(1/2)phi(z_hat,1)`, `v_f=P^(1/2)phi(x_f[t],0)`.
The linear map from vec(G) to
`DeltaW^T phi(z_hat,1)-R DeltaW^T phi(x_f,0)` has squared operator norm

`||v_c||²+||v_f||²-2 cos(theta) <v_c,v_f>`.

This follows from `R+R^T=2cos(theta) I` in two dimensions. It retains within-step
factual/counterfactual parameter sharing. A uniform bound k_t on this norm gives
`E[t+1] <= rho E[t] + r k_t`, starting from E[0]=0.

The separate tube instead uses `||v_c||+||v_f||`, a valid looser baseline.
The statistics-only comparator fixes R=I and uses the exact force norm
`||P^(1/2)(phi(z_hat,1)-phi(x_f,0))||`; it is a confidence tube for that fixed
coupling convention, not for all possible theta.

## Shared-parameter Taylor tube

Let J_t(theta) be the derivative of z_t with respect to whitened shared G at G=0.
The derivative recurrence is

`J[t+1]=A1_hat J[t]+K_t(theta)`.

The same G multiplies the entire accumulated derivative J, so its support is
evaluated **after** the temporal sum. Each J is a first harmonic in theta;
three coefficient recurrences suffice. For exact error e and linear part JG,
the remainder b obeys

`b[t+1] <= rho b[t] + q*r*sup_theta ||J_t(theta)||_2`, b[0]=0.

Proof: the only term not included in linearization is `Delta A1 * e_t`;
substitute `e_t=J_tG+remainder`, apply operator norms, and use the same q.
At the terminal readout, use `r ||row_0(J_H(theta))||_2 + b_H` around each
nominal effect, then take the union over theta. Removing b is only an ablation;
its good empirical coverage in this toy system supplies no general guarantee.

## Continuous angles, not an uncorrected grid

For a matrix-valued first harmonic F(theta)=Fd+Fc cos(theta)+Fs sin(theta),
`||F'(theta)||_2 <= ||Fc||_2+||Fs||_2`. A uniform K-point circular grid has maximum
nearest distance pi/K. The code adds this derivative bound times pi/K to each
grid extremum. The endpoint effect derivative is bounded by
`sqrt(c_H[0]^2+s_H[0]^2)`. This interpolation remainder ensures the analytic
bound covers the continuous global-angle family, rather than just K samples.

## Verification and limits

Tests independently simulate 4097 angles to check the exact family extrema,
check pointwise likelihood invariance, and draw 128 models on the exact
coefficient-confidence boundary to test both outer bounds. Fresh repetitions
also assert coverage whenever the true coefficient matrix lies in the event.
These are falsification checks alongside the algebra, not a formal proof checker.

The main prototype passed coverage and sign-utility gates but failed its width
gate. Small-sample tubes remain extremely wide. General nonlinear, unknown
variance/covariance, hidden states and learned representation uncertainty remain
unresolved here. Mature Taylor-model tools are relevant stronger baselines.
