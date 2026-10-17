"""Training-only inspection of failed EM numeric state and installed source."""
from common54 import *
import inspect,pickle
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from dynamax.ssm import SSM
run='em_linear_small_seed0'
h=np.load(ART/(run+'_em_objective_audit_v2.npy'));finite=np.isfinite(h)
print(json.dumps({'shape':h.shape,'finite_count':int(finite.sum()),'first_nonfinite_index':int(np.where(~finite)[0][0]) if not finite.all() else None,'first_20':[float(x) if np.isfinite(x) else str(x) for x in h[:20]],'last_5':[float(x) if np.isfinite(x) else str(x) for x in h[-5:]]}))
print('MODEL_INIT\n',inspect.getsource(LinearGaussianSSM.initialize))
print('MODEL_MSTEP\n',inspect.getsource(LinearGaussianSSM.m_step))
print('SSM_FIT_EM\n',inspect.getsource(SSM.fit_em))
