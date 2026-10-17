"""Per-sequence decomposition of the one-factor native RP auxiliary loss.

Adapted/reindexed from rp_ssm.rpm.ConstrainedIVFreeEnergy.get_loss_terms,
gatsby-sahani/rp-gssm commit a75225d5edceb972bb872dc9a335608733567dd4.
Original authors: Samo Hromadka and Kai Biegun; CC BY-NC 4.0.
https://github.com/gatsby-sahani/rp-gssm/blob/a75225d5edceb972bb872dc9a335608733567dd4/LICENSE
Changes: expose the target-sequence axis rather than immediately summing it;
one unconditional factor only. This is the same existing objective, not a new loss.
"""
import jax,jax.numpy as jnp
from jax.scipy.special import logsumexp
from rp_ssm.datasets import TrainData
def per_target_loss(energy,params,x):
 assert len(params)==2
 prior=energy.model.prior.update(params[0])
 chain,sites,posterior=energy.get_posterior(None,prior,params[1:],TrainData(obs=(x,)))
 assert sites.params['pwm'].shape[0]==1
 kl_prior=jax.vmap(lambda q:q.kl(chain))(posterior)
 q_nat=posterior.all_param.nat_param;site=jax.tree.map(lambda a:a[0],sites);p_nat=chain.all_param.nat_param
 kl_factor=jax.vmap(jax.vmap(lambda q,f:q.kl(q+f)))(q_nat,site)
 def at_time(qs,fs,p):
  return jax.vmap(lambda q:jax.vmap(lambda f:(f+q).lognormalizer-(f+p).lognormalizer)(fs))(qs)
 scores=jax.vmap(at_time,in_axes=(1,1,0))(q_nat,site,p_nat)
 positives=jnp.diagonal(scores,axis1=-2,axis2=-1)
 association=positives-logsumexp(scores,axis=-1)
 return -(association.sum(0)-kl_factor.sum(1)-kl_prior)/x.shape[1]
