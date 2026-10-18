"""Independent equations, quadrature/augmented least squares and metric replay."""
import sys,pathlib,time,math
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import run83_v1 as run
import numpy as np
from scipy.integrate import solve_ivp
def exact_rhs(y,family,setting):
 if family=='hill':
  alpha,beta=((2.,1.),(4.,.8))[setting];return alpha*(1+.05*np.sin(np.arange(7)))/(1+np.roll(y,1)**3)-beta*y
 v=y[::2];w=y[1::2];g,base=((.15,.3),(.4,.6))[setting];out=np.empty(12);out[::2]=v-v*v*v/3-w+g*(np.roll(v,1)-v)+base+.1*np.sin(np.arange(6));out[1::2]=(.15+.02*np.arange(6))*(v+.7-.8*w);return out
def reference(y0,t,family,setting,clamp):
 def rhs(now,y):
  q=y.copy()
  if clamp:q[0]=y0[0]+.5*np.sin(.8*now)
  dy=exact_rhs(q,family,setting)
  if clamp:dy[0]=.4*np.cos(.8*now)
  return dy
 s=solve_ivp(rhs,(t[0],t[-1]),y0,t_eval=t,method='DOP853',rtol=1e-9,atol=1e-11);assert s.success;p=s.y.T
 if clamp:p[:,0]=y0[0]+.5*np.sin(.8*t)
 return p
def checks(p,y,clamp,row):
 good=p.shape==y.shape and np.isfinite(p).all();assert good==row['finite_complete']
 if not good:assert row['coordinate_mean_nrmse'] is None;return 0.
 if clamp:p=p[:,1:];y=y[:,1:]
 err=p-y;mse=float(np.mean(err**2));coord=np.sqrt(np.mean(err**2,axis=0))/np.maximum(np.std(y,axis=0),1e-12);values=dict(mse=mse,nrmse=float(np.sqrt(mse)/max(np.std(y),1e-12)),coordinate_mean_nrmse=float(np.mean(coord)))
 assert np.allclose(coord,row['coordinate_nrmse'],atol=1e-12,rtol=1e-12)
 for name,value in values.items():assert math.isclose(value,row[name],abs_tol=1e-12,rel_tol=1e-12),(row['tag'],row['kind'],name)
 return max(abs(value-row[name]) for name,value in values.items())
def main():
 start=time.perf_counter();r=read(OUT/'RESULTS83_V1.json');manifest=read(OUT/'DATA_MANIFEST83_V1.json');assert len(r['rollouts'])==192 and len(r['fields'])==48;assets=run.old.assets();weight=next(x for x in assets if x['relative'].endswith('.safetensors'));model=run.load_fim_ode_local(pathlib.Path(weight['path']).parent,'cpu');model.eval();identity=0;score_error=0.;coef_rows=[];native=[];predictions={};scalar_count=0;ref_errors=[];selected=[]
 for parameter in model.parameters():parameter.requires_grad_(False)
 assert all(not parameter.requires_grad for parameter in model.parameters())
 for item in manifest['rows']:
  assert sha(item['path'])==item['sha256'];tag=item['tag']
  with np.load(item['path'],allow_pickle=False) as a:ctx={k:a[k].copy() for k in a.files}
  x=ctx['context'];t=ctx['time'];clean=ctx['clean_context'];par=item['parents']
  for j,y0 in enumerate(ctx['initial_states']):
   ref=reference(y0,t,item['family'],item['setting'],False);error=float(abs(ref-clean[j]).max());assert np.allclose(ref,clean[j],atol=1e-7,rtol=1e-7);ref_errors.append(error);identity+=1
  rng=np.random.default_rng(item['seed'])
  if item['family']=='hill':initial=rng.uniform(.5,2.,(4,7))
  else:
   initial=np.empty((4,12));initial[:,::2]=rng.normal(0,.7,(4,6));initial[:,1::2]=rng.uniform(-.5,.5,(4,6))
  noise=rng.normal(size=clean.shape);assert np.array_equal(initial,ctx['initial_states']);assert np.array_equal(x,clean+item['noise_sigma']*clean.std((0,1))*noise);assert np.allclose(np.stack([exact_rhs(z,item['family'],item['setting']) for z in ctx['queries']]),ctx['field_target'],atol=1e-12,rtol=1e-12);identity+=3
  fr=next(z for z in r['fields'] if z['tag']==tag and z['kind']=='FIM_local_composed');nf=run.Native(model,x,t,par)(ctx['queries'])
  with np.load(fr['path'],allow_pickle=False) as a:error=float(abs(nf-a['prediction']).max());assert error==0.;native.append(dict(tag=tag,max_abs=error))
  for kind in ['weak_polynomial3','weak_rff128']:
   cs=[z for z in r['candidates'] if z['tag']==tag and z['kind']==kind];assert len(cs)==3;valid=[z for z in cs if z['validation_mse'] is not None];saved=next(z for z in r['models'] if z['tag']==tag and z['kind']==kind)
   if not valid:assert saved['all_candidates_failed'];continue
   best=min(valid,key=lambda z:(z['validation_mse'],z['ridge']));assert sha(saved['path'])==saved['sha256'];pars=read(saved['path']);assert best['ridge']==pars['ridge'];selected.append(dict(tag=tag,kind=kind,ridge=pars['ridge']))
   for target,(idx,p) in enumerate(zip(par,pars['locals'])):
    xx=x[...,idx];m=xx.mean((0,1));s=np.maximum(xx.std((0,1)),1e-8);h=run.phi(xx,m,s,kind);aa=[];bb=[]
    for width in [8,16,32]:
     for path in range(4):
      for begin in range(len(t)-width):
       end=begin+width;aa.append(np.trapz(h[path,begin:end+1],t[begin:end+1],axis=0));bb.append((xx[path,end,0]-xx[path,begin,0])/s[0])
    a=np.asarray(aa);b=np.asarray(bb);scale=np.maximum(np.sqrt(np.mean(a*a,0)),1e-10);z=a/scale;ridge=p['ridge'];ind=np.linalg.lstsq(np.r_[z,np.sqrt(ridge)*np.eye(z.shape[1])],np.r_[b,np.zeros(z.shape[1])],rcond=None)[0]/scale;stored=np.asarray(p['coefficient']);err=float(abs(ind-stored).max());assert np.allclose(ind,stored,atol=1e-6,rtol=1e-5),(tag,kind,target,err);coef_rows.append(dict(tag=tag,kind=kind,target=target,parameters=len(ind),max_abs=err))
   frow=next(z for z in r['fields'] if z['tag']==tag and z['kind']==kind)
   with np.load(frow['path'],allow_pickle=False) as a:assert np.array_equal(run.ordinary_field(pars['locals'],par)(a['queries']),a['prediction']);identity+=1
 for row in r['fields']+r['rollouts']:
  assert sha(row['path'])==row['sha256']
  with np.load(row['path'],allow_pickle=False) as a:
   p=a['prediction'];y=a['target'];scalar_count+=p.size+y.size;score_error=max(score_error,checks(p,y,row.get('clamp',False),row))
   if 'clamp' in row:
    item=next(i for i in manifest['rows'] if i['tag']==row['tag']);ref=reference(a['initial_state'],a['time'],item['family'],item['setting'],row['clamp']);err=float(abs(ref-y).max());assert np.allclose(ref,y,atol=1e-7,rtol=1e-7);ref_errors.append(err);predictions[row['tag'],row['kind'],row['initial_index'],row['clamp']]=(p.copy(),y.copy());identity+=1
 for row in r['responses']:
  free,fy=predictions[row['tag'],row['kind'],row['initial_index'],False];clamp,cy=predictions[row['tag'],row['kind'],row['initial_index'],True];ok=free.shape==fy.shape and clamp.shape==cy.shape;diff=clamp-free if ok else np.empty((0,fy.shape[1]));score_error=max(score_error,checks(diff,cy-fy,True,row))
 assert len(native)==16 and run.old.check_source()==70 and sha(weight['path'])==weight['sha256'];dump(OUT/'VERIFICATION83_V1.json',dict(utc=utc(),status='PASS_INDEPENDENT_EQUATIONS_QUADRATURE_SCORES_AND_NATIVE_FIELDS',score_max_abs=score_error,prediction_target_scalars=scalar_count,identity_checks=identity,native_field_replays=native,independent_coefficients=coef_rows,selection_checks=selected,maximum_reference_error=max(ref_errors),failed_rollouts=sum(not z['finite_complete'] for z in r['rollouts']),weights_unchanged=True,source_files_checked=70,wall_seconds=time.perf_counter()-start));print(json.dumps(dict(status='PASS',scores=336,coefficient_models=len(coef_rows),coefficient_count=sum(z['parameters'] for z in coef_rows),max_coefficient_error=max(z['max_abs'] for z in coef_rows),max_reference_error=max(ref_errors))),flush=True)
if __name__=='__main__':main()
