"""Execute pinned native RP-GSSM precision maps and Kalman posterior path."""
import os
os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['JAX_ENABLE_X64']='true'
os.environ.setdefault('OMP_NUM_THREADS','4')
if os.name=='nt':
    import ctypes
    fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
import hashlib,importlib.metadata,json,pathlib,sys,time
import numpy as onp
import jax
import jax.numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];SRC=ROOT.parent/'upstream/rp-gssm'
sys.path.insert(0,str(SRC))
from rp_ssm.distmaps import MVNDiag
from rp_ssm.dists import LGStationaryParam,GaussianNatParam
from rp_ssm.datasets import TrainData
from rp_ssm.rpm import RPSSM,ConstrainedIVFreeEnergy

class FixedRecognition:
    def apply(self,params,x):return MVNDiag(1)(x)

def main():
    start=time.perf_counter();prior=LGStationaryParam(start_from_invariant=True,stay_at_invariant=True,opt_params=['A'],A=np.array([[.9]]))
    model=RPSSM(prior,[FixedRecognition()]);energy=ConstrainedIVFreeEnergy(model);records=[]
    for logprecision in [-20.,-5.,0.,5.,20.]:
        # Native map: [precision-weighted mean, unconstrained precision logit].
        x=np.tile(np.array([0.,logprecision]),(2,8,1));data=TrainData(obs=(x,))
        _,factors,posterior=energy.get_posterior(None,prior,[None],data)
        covs=onp.asarray(posterior.params['covs']);assert onp.isfinite(covs).all()
        p=float(MVNDiag(1)(np.array([0.,logprecision])).params['p'][0,0])
        # The upstream smoother explicitly adds 1e-5 to its returned diagonal.
        # Check the Gaussian inference constraint before that numerical padding.
        assert covs.max()<=1+1e-5+1e-10
        records.append({'precision_logit':logprecision,'native_precision':p,'max_posterior_variance':float(covs.max()),'min_posterior_variance':float(covs.min()),'upstream_covariance_padding':1e-5,'max_variance_before_padding':float(covs.max()-1e-5)})
    result={'source_revision':'a75225d5edceb972bb872dc9a335608733567dd4','scope':'unmodified precision-map and posterior-inference component, fixed oracle recognition potentials; no native model training claimed',
            'records':records,'packages':{p:importlib.metadata.version(p) for p in ['jax','jaxlib','flax','optax','numpy','dynamax','tensorflow-probability']},
            'source_sha256':{f:hashlib.sha256((SRC/f).read_bytes()).hexdigest() for f in ['rp_ssm/distmaps.py','rp_ssm/rpm.py','rp_ssm/dists_utils.py']},
            'devices':[str(d) for d in jax.devices()],'runtime_seconds':time.perf_counter()-start}
    (ROOT/'outputs/native_rp_precision.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
