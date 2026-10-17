"""Ordinary LGSSM EM: correct native initial covariance's mean normalization only."""
import jax,jax.numpy as jnp
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
class CorrectedLinearGaussianSSM(LinearGaussianSSM):
 def m_step(self,params,props,batch_stats,m_step_state):
  updated,state=super().m_step(params,props,batch_stats,m_step_state)
  sum_x,sum_xx,N=jax.tree.map(lambda x:jnp.sum(x,axis=0),batch_stats)[0]
  mean=sum_x/N
  covariance=sum_xx/N-jnp.outer(mean,mean)
  covariance=(covariance+covariance.T)/2
  return updated._replace(initial=updated.initial._replace(cov=covariance)),state
