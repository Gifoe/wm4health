"""Independent raw-score, identity, selection and augmented-LS checks."""
import sys,pathlib,time,math
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common81_v1 import *
from features81_v1 import NativeFeatures,load_assets,qualify,old
import ordinary80_v1 as ordinary
import numpy as np
import torch
from ode_tutorial_helper import load_odebench_system
torch.set_num_threads(2);torch.set_num_interop_threads(2)

def independent_coefficient(nf,x,t,kind,ridge):
 phi,base=nf.phi(x,kind);z=(x-nf.m)/nf.s;aa=[];bb=[]
 # Direct per-window quadrature instead of the fitting code's cumulative sums.
 for width in [8,16,32]:
  for start in range(len(t)-width):
   end=start+width;sel=slice(start,end+1);aa.append(np.trapz(phi[sel],t[sel],axis=0));b=z[end]-z[start]
   if kind!='fim_readout_zero':b=b-np.trapz(base[sel]/nf.s,t[sel],axis=0)
   bb.append(b)
 A=np.asarray(aa);B=np.asarray(bb);scale=np.maximum(np.sqrt(np.mean(A*A,axis=0)),1e-10);Z=A/scale
 # Augmented least squares instead of the fitting code's normal equations.
 C=np.linalg.lstsq(np.r_[Z,np.sqrt(ridge)*np.eye(Z.shape[1])],np.r_[B,np.zeros((Z.shape[1],2))],rcond=None)[0]/scale[:,None]
 return C

def main():
 started=time.perf_counter();result=read(OUT/'RESULTS81_V1.json');cohort=read(OUT/'COHORT81_V1.json');assert not result['evaluation_opened'];assert {r['system'] for r in result['rows']}==set(cohort['development']);assert len(result['rows'])==72
 model,weights,data=load_assets();count=0;maxmetric=0.;identities=0;failures=[];data_by_system={};models={};coeffchecks=[];selectionchecks=[];qualification=[]
 for sid in cohort['development']:
  tt,xx,_=load_odebench_system(sid,data['path'],noise_sigma=.03,subsample_fraction=.5,rng=np.random.default_rng(2026810));ct,yy,_=load_odebench_system(sid,data['path'],noise_sigma=0.,subsample_fraction=1.,rng=np.random.default_rng(2026810));data_by_system[sid]=(xx[0],tt[0,:,0],ct,yy)
 for row in result['rows']:
  assert sha(row['array_path'])==row['array_sha256']
  with np.load(row['array_path'],allow_pickle=False) as a:
   x,t,ct,yy=data_by_system[row['system']];k=row['initial_condition_index'];assert np.array_equal(a['context'],x) and np.array_equal(a['context_times'],t) and np.array_equal(a['target'],yy[k]) and np.array_equal(a['time'],ct[k,:,0]);identities+=4;p=a['prediction'];y=a['target'];valid=p.shape==y.shape and np.isfinite(p).all();assert valid==row['finite_complete']
   if valid:
    err=p-y;values=dict(mse=float((err*err).mean()),rmse=float(np.sqrt((err*err).mean())),mae=float(np.abs(err).mean()),nrmse=float(np.sqrt((err*err).mean())/y.std()))
    for name,value in values.items():maxmetric=max(maxmetric,abs(value-row[name]));assert math.isclose(value,row[name],rel_tol=1e-12,abs_tol=1e-12)
    count+=p.size+y.size
   else:failures.append(dict(system=row['system'],kind=row['kind'],ic=k,solver=row['solver']))
 for item in result['models']:
  assert sha(item['path'])==item['sha256'];models[(item['system'],item['kind'])]=read(item['path'])
 for sid in cohort['development']:
  x,t,_,_=data_by_system[sid];nf=NativeFeatures(model,x,t);qualification.append(dict(system=sid,**qualify(model,data,sid)))
  for kind in ['fim_readout_residual','fim_readout_zero','rff_residual','affine_residual','ordinary_weak_dictionary']:
   rows=[r for r in result['candidates'] if r['system']==sid and r['kind']==kind];valid=[r for r in rows if r['solver']['success'] and r['validation']['finite_complete']];pars=models[(sid,kind)]
   if not valid:assert pars['all_context_candidates_failed'];continue
   if kind=='ordinary_weak_dictionary':
    best=min(valid,key=lambda r:(r['validation']['mse'],r['library'],r['ridge'],r['threshold']));assert all(best[k]==pars[k] for k in ['library','ridge','threshold'])
   else:
    best=min(valid,key=lambda r:(r['validation']['mse'],r['ridge']));assert best['ridge']==pars['ridge'];ind=independent_coefficient(nf,x,t,kind,pars['ridge']);stored=np.asarray(pars['coefficient']);diff=float(abs(ind-stored).max());assert np.allclose(ind,stored,atol=5e-7,rtol=5e-6),(sid,kind,diff);coeffchecks.append(dict(system=sid,kind=kind,independent_max_abs_coefficient=diff,features=len(ind),checked_coefficients=ind.size))
   selectionchecks.append(dict(system=sid,kind=kind,selected_ridge=pars['ridge'],candidate_count=len(rows)))
 expected_selected=sum(item['kind']!='native' and not models[(item['system'],item['kind'])].get('all_context_candidates_failed',False) for item in result['models']);expected_coeff=sum(item['kind'] in ['fim_readout_residual','fim_readout_zero','rff_residual','affine_residual'] and not models[(item['system'],item['kind'])].get('all_context_candidates_failed',False) for item in result['models']);assert len(selectionchecks)==expected_selected and len(coeffchecks)==expected_coeff and old.check_source()==70 and sha(weights['path'])==weights['sha256']
 dump(OUT/'VERIFICATION81_V1.json',dict(utc=utc(),status='PASS_RAW_SCORES_IDENTITIES_SELECTION_AND_INDEPENDENT_AUGMENTED_LS',metric_max_abs=maxmetric,prediction_target_scalars=count,source_identity_arrays_checked=identities,raw_rows_checked=len(result['rows']),fit_selection_checks=selectionchecks,independent_coefficients=coeffchecks,context_only_qualifications=qualification,failed_trajectories=failures,reserved_evaluation_opened=False,weights_unchanged=True,source_files_checked=70,wall_seconds=time.perf_counter()-started))
 print(json.dumps(dict(status='PASS',raw_rows=72,metric_max_abs=maxmetric,coefficients_checked=sum(r['checked_coefficients'] for r in coeffchecks),maximum_independent_coefficient_difference=max(r['independent_max_abs_coefficient'] for r in coeffchecks),failed_trajectories=len(failures))),flush=True)
if __name__=='__main__':main()
