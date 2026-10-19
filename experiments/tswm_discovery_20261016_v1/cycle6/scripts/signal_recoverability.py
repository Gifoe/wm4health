"""Training-only linear recoverability certificates for the actual linear A data."""
import runtime_profile
import datetime,json,pathlib
import numpy as np
from scipy.stats import spearmanr
from native_child import ROOT
def main():
 out=ROOT/'outputs/signal_recoverability';out.mkdir(parents=True,exist_ok=True)
 split=np.load(ROOT/'outputs/child_native_A_r1/split_indices.npz');train=split['train'];valid=split['validation'];rows=[]
 for variant in ['native_A','independent_noise_A']:
  with np.load(ROOT/f'data/{variant}/A/data.npz') as data:x=data['xt'];y=data['yt'][:,3]
  for width in [1,3,5]:
   half=width//2;features=x[:,3-half:4+half].reshape(len(x),-1);features=np.c_[features,np.ones(len(features))]
   coef=np.linalg.lstsq(features[train],y[train],rcond=None)[0];prediction=features[valid]@coef;error=y[valid]-prediction
   r2=1-error.var(0)/y[valid].var(0);corr=np.array([spearmanr(y[valid,j],prediction[:,j]).statistic for j in range(7)])
   row={'variant':variant,'context_width':width,'training_trajectories':len(train),'validation_trajectories':len(valid),'r2_per_true_factor':r2.tolist(),'spearman_per_true_factor':corr.tolist(),'mean_spearman':float(corr.mean()),'mean_R2':float(r2.mean()),'normalized_residual_variance_per_factor':(error.var(0)/y[valid].var(0)).tolist(),'scope':'Supervised recoverability certificate only, not an unsupervised method or independent test comparison. Same validation split as native configuration, centered time3.'};rows.append(row)
   np.savez(out/f'{variant}_w{width}.npz',coef=coef,validation_prediction=prediction,validation_true=y[valid]);print(json.dumps(row),flush=True)
 result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'source_caution':'The pinned native leaky_ReLU helper is identity, so native A observation mixing and latent recurrence are linear. The independent-noise control changes noise indexing only. Neither is certified to satisfy every nonlinear identifiability assumption.','interpretation':'A supervised certificate can separate missing information from learning failure; it is not a deployable baseline with matched latent-label access. Finite-window latent-distribution identifiability does not imply exact realization recovery under stochastic partial observation.'}
 (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
