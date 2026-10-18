import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common81_v1 import *
from features81_v1 import *
import ordinary80_v1 as ordinary
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
def main():
 r=read(OUT/'RESULTS81_V1.json');model,weights,data=load_assets();rows=[]
 for sid in read(OUT/'COHORT81_V1.json')['development']:
  tt,xx,_=load_odebench_system(sid,data['path'],noise_sigma=.03,subsample_fraction=.5,rng=np.random.default_rng(2026810));x=xx[0];t=tt[0,:,0];nf=NativeFeatures(model,x,t)
  for kind in ['fim_readout_residual','fim_readout_zero','rff_residual','affine_residual']:
   pars=read(next(z['path'] for z in r['models'] if z['system']==sid and z['kind']==kind));phi,base=nf.phi(x,kind);z=(x-nf.m)/nf.s;ip=np.r_[np.zeros((1,phi.shape[1])),np.cumsum(.5*(phi[1:]+phi[:-1])*np.diff(t)[:,None],axis=0)];ib=np.r_[np.zeros((1,2)),np.cumsum(.5*(base[1:]+base[:-1])/nf.s*np.diff(t)[:,None],axis=0)];a1=[];b1=[];a2=[];b2=[]
   for w in [8,16,32]:
    a1.append(ip[w:]-ip[:-w]);b=z[w:]-z[:-w]
    if kind!='fim_readout_zero':b=b-(ib[w:]-ib[:-w])
    b1.append(b)
    for j in range(len(t)-w):
     sel=slice(j,j+w+1);a2.append(np.trapz(phi[sel],t[sel],axis=0));b=z[j+w]-z[j]
     if kind!='fim_readout_zero':b=b-np.trapz(base[sel]/nf.s,t[sel],axis=0)
     b2.append(b)
   A=np.concatenate(a1);B=np.concatenate(b1);A2=np.asarray(a2);B2=np.asarray(b2);scale=np.maximum(np.sqrt(np.mean(A*A,axis=0)),1e-10);Z=A/scale;ridge=pars['ridge'];normal=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T@B);aug=np.linalg.lstsq(np.r_[Z,np.sqrt(ridge)*np.eye(Z.shape[1])],np.r_[B,np.zeros((Z.shape[1],2))],rcond=None)[0];scale2=np.maximum(np.sqrt(np.mean(A2*A2,axis=0)),1e-10);Z2=A2/scale2;aug2=np.linalg.lstsq(np.r_[Z2,np.sqrt(ridge)*np.eye(Z2.shape[1])],np.r_[B2,np.zeros((Z2.shape[1],2))],rcond=None)[0];stored=np.asarray(pars['coefficient']);row=dict(system=sid,kind=kind,ridge=ridge,max_abs_coefficient=float(abs(stored).max()),min_column_rms=float(scale.min()),design_quadrature_max_abs=float(abs(A-A2).max()),response_quadrature_max_abs=float(abs(B-B2).max()),stored_vs_reconstructed_original_max_abs=float(abs(stored-normal/scale[:,None]).max()),same_design_augmented_raw_coef_max_abs=float(abs(stored-aug/scale[:,None]).max()),same_design_augmented_scaled_coef_max_abs=float(abs(stored*scale[:,None]-aug).max()),alternate_quadrature_scaled_coef_max_abs=float(abs(stored*scale[:,None]-aug2).max()),alternate_quadrature_raw_coef_max_abs=float(abs(stored-aug2/scale2[:,None]).max()),same_design_context_field_max_abs=float(abs(phi@(stored-aug/scale[:,None])*nf.s).max()));rows.append(row);print(json.dumps(row),flush=True)
 dump(OUT/'SOLVER_DIAGNOSTIC81_V1.json',dict(utc=utc(),rows=rows,numerical_experiment_modified=False))
 print(json.dumps(dict(new_ic=[{k:z[k] for k in ['system','kind','nrmse']} for z in r['rows'] if z['initial_condition_index']==1])),flush=True)
if __name__=='__main__':main()
