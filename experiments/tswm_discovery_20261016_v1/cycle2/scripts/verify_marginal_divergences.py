"""Independent integration of reference/prior divergence in the fixed example."""
import hashlib,json,pathlib
import numpy as np
from scipy.integrate import quad
from probe_rpm_tail_bound import log_normal
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 source=ROOT/'outputs/rpm_tail_probe.json';records=[]
 for item in json.loads(source.read_text())['records']:
  V=item['recognition_variance'];eps=item['reference_rare_probability']
  def parts(z):
   lp=log_normal(z,1.);lr=log_normal(z,V);lF=np.logaddexp(np.log1p(-eps)+lp,np.log(eps)+lr);F=np.exp(lF);w=np.exp(np.log(eps)+lr-lF)
   return lp,lF,F,w
  kl,kerr=quad(lambda z:parts(z)[2]*(parts(z)[1]-parts(z)[0]),-np.inf,np.inf,epsabs=1e-12,epsrel=1e-9,limit=300)
  fisher,ferr=quad(lambda z:parts(z)[2]*(parts(z)[3]*(1-1/V)*z)**2,-np.inf,np.inf,epsabs=1e-12,epsrel=1e-9,limit=300)
  klbound=eps*.5*(V-1-np.log(V));fbound=eps*(V-1)**2/V
  assert -1e-11<=kl<=klbound+1e-10
  assert -1e-11<=fisher<=fbound+1e-10
  records.append({'recognition_variance':V,'reference_rare_probability':eps,'kl_F_to_prior':float(kl),'kl_convexity_upper_bound':float(klbound),'relative_fisher_F_to_prior':float(fisher),'relative_fisher_upper_bound':float(fbound),'kl_quad_error':float(kerr),'fisher_quad_error':float(ferr),'interior_objective_gap':item['interior_objective_gap'],'ordinary_gaussian_vi_kl':item['ordinary_gaussian_vi'][-1]['kl_to_exact']})
 result={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'records':records,'scope':'Actual F||prior KL and relative Fisher divergence, not moment-matched Gaussian proxies. The conditioning event becomes rare; no contradiction to an average-case convergence theorem is claimed.'}
 (ROOT/'outputs/marginal_divergence_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps([r for r in records if r['recognition_variance']==3],indent=2))
if __name__=='__main__':main()
