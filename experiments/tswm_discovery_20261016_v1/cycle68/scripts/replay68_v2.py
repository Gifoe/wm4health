"""Independent generator, scores and augmented least-squares operator replay."""
import sys,pathlib,math,statistics,collections,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native68_v1 import NormWear,MetaID
from common68_v1 import ROOT,OUT,L,H,P,read,sha,dump,utc
import numpy as np
assert read(OUT/'campaign_state68_v2.json')['status']=='VALIDATION_AND_EVALUATION_TERMINAL_SUCCESS'
counts=collections.Counter();errors={}
def check(name,a,b,tol):
 err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert err<=tol,(name,err,tol);errors[name]=max(err,errors.get(name,0.))
def independent_ridge(X,Y,alpha):
 xm=X.mean(0);xs=X.std(0);xs[xs<1e-12]=1.;ym=Y.mean(0);Z=(X-xm)/xs;A=np.r_[Z,np.sqrt(alpha)*np.eye(Z.shape[1])];B=np.r_[Y-ym,np.zeros((Z.shape[1],Y.shape[1]))];w=np.linalg.lstsq(A,B,rcond=None)[0];return xm,xs,ym,w
def raw_reference(h,known,degree,alpha):
 mu=np.array([statistics.mean(x) for x in h]);sd=np.array([statistics.stdev(x)+1e-8 for x in h]);y=(h[0]-mu[0])/sd[0];u=(h[1]-mu[1])/sd[1];X=[];Y=[]
 for t in range(1,L-1):X.append([y[t],y[t-1],u[t+1]]+([y[t]**2,y[t]**3] if degree==3 else []));Y.append([y[t+1]])
 m=independent_ridge(np.array(X),np.array(Y),alpha);xm,xs,ym,w=m;past=list(y);pred=[]
 for rawu in known:
  a=past[-1];b=past[-2];v=(rawu-mu[1])/sd[1];x=np.array([a,b,v]+([a*a,a*a*a] if degree==3 else []));q=float(((x-xm)/xs)@w[:,0]+ym[0]);pred.append(mu[0]+sd[0]*q);past.append(q)
 return np.array(pred),m
manifest=read(OUT/'CASE_MANIFEST68_V1.json');ledger=read(OUT/'FORECAST_LEDGER68_V1.json');lookup={r['case']:r for r in ledger['cases']};all_rows=[]
for name in ['VALIDATION','EVALUATION']:all_rows+=read(OUT/(name+'_RESULTS68_V1.json'))['rows']
native=NormWear();meta=MetaID();first_eval=False
for case in manifest['cases']:
 assert sha(case['path'])==case['sha256'];z=np.load(case['path'],allow_pickle=False);h=z['history'];known=z['known'];target=z['target'];rng=np.random.default_rng(case['seed']);r=float(rng.uniform(.94,.98));theta=float(rng.uniform(.08,.18));gain=float(rng.uniform(.7,1.3));c=0. if case['family']=='linear' else float(rng.uniform(.0002,.001));a1=2*r*math.cos(theta);a2=-r*r;b=gain*(1-a1-a2);n=1024+L+H+2;values=rng.uniform(-1,1,math.ceil(n/16));u=[float(values[i//16]) for i in range(n)];y=[0.,0.]
 for t in range(1,n-1):y.append(a1*y[-1]+a2*y[-2]+b*u[t]-c*y[-1]**3)
 raw=np.array([y[1024:1024+L+H],u[1023:1023+L+H]]);check('regenerated_case',raw,z['raw'],1e-9);assert np.array_equal(z['raw'][:,:L],h) and np.array_equal(z['raw'][1,L:],known) and np.array_equal(z['raw'][0,L:],target);counts['independently_regenerated_cases']+=1
 rec=lookup[case['id']];assert sha(rec['path'])==rec['sha256'];archive=np.load(rec['path'],allow_pickle=False);counts['forecast_archive_hashes']+=1;rows=[q for q in all_rows if q['case']==case['id']];native.set_case(h,known);E=native.E.astype('float64');esvd=None
 sd=statistics.stdev(h[0])+1e-8
 for row in rows:
  key=row['method']+'__'+str(row.get('MC_seed',2021));p=archive[key]
  if row['status']=='NONFINITE':assert not np.isfinite(p).all();counts['retained_nonfinite_rows']+=1;continue
  d=[float(a)-float(b) for a,b in zip(p,target)];mae=math.fsum(abs(q) for q in d)/H;rmse=math.sqrt(math.fsum(q*q for q in d)/H);check('MAE',mae,row['MAE'],1e-9);check('RMSE',rmse,row['RMSE'],1e-9);check('normalized_MAE',mae/sd,row['normalized_MAE'],1e-9);check('first50',math.fsum(abs(q) for q in d[:50])/50/sd,row['first50_normalized_MAE'],1e-9);check('last50',math.fsum(abs(q) for q in d[50:])/50/sd,row['last50_normalized_MAE'],1e-9);counts['independent_metric_rows']+=1
  method=row['method']
  if method.startswith('raw_arx_'):
   expected,m=raw_reference(h,known,row['degree'],row['alpha']);check('raw_forecast_augmented_fit',p,expected,1e-6)
   for tag,value in zip(['xm','xs','ym','w'],m):check('raw_operator_'+tag,archive[method+'__fit_'+tag],value,1e-7)
   counts['raw_fits_and_rollouts']+=1
  elif method.startswith('latent_ridge_'):
   k=row['k'];alpha=row['alpha'];basis=archive[method+'__basis'];mu=archive[method+'__mean'];check('latent_center',mu,E.mean(0),1e-8);check('PCA_orthonormality',basis@basis.T,np.eye(k),1e-7)
   if esvd is None:esvd=np.linalg.svd(E-E.mean(0),full_matrices=False)
   basis2=esvd[2][:k];check('PCA_projection',((E-mu)@basis.T)@basis,((E-mu)@basis2.T)@basis2,1e-5)
   C=(E-mu)@basis.T;um=statistics.mean(h[1]);us=statistics.stdev(h[1])+1e-8;U=(h[1]-um)/us;X=np.c_[C[:-1],U.reshape(-1,P)[1:]];xm,xs,ym,w=independent_ridge(X,C[1:],alpha)
   for tag,value in zip(['xm','xs','ym','w'],[xm,xs,ym,w]):check('latent_operator_'+tag,archive[method+'__operator_'+tag],value,1e-6)
   future=(known-um)/us;future=np.r_[future,np.repeat(future[-1],(-len(future))%P)];current=C[-1];pred=[]
   for block in future.reshape(-1,P):current=((np.r_[current,block]-xm)/xs)@w+ym;pred.append(current@basis+mu)
   expected=native.decode(np.array(pred));check('latent_operator_redecoded',p,expected,1e-4);counts['latent_fits_and_decoded_rollouts']+=1
  elif method=='context_mean':check('context_mean',p,np.repeat(statistics.mean(h[0]),H),1e-10)
  elif method=='last_value':check('last_value',p,np.repeat(h[0,-1],H),0)
 if case['subset']=='EVAL' and not first_eval:
  for method in ['native_predict','bayes_indexed','bayes_same_view']:expected=native.forecast(method,2021);check('fresh_native_'+method,archive[method+'__2021'],expected,1e-4);counts['fresh_native_calls']+=1
  expected,_=meta.forecast(h,known);check('fresh_author_meta_id',archive['meta_id_author_ensemble__2021'],expected,1e-5);counts['fresh_native_calls']+=1;first_eval=True
 print(case['id'],'REPLAYED',flush=True)
selection=read(OUT/'VALIDATION_SELECTION68_V1.json');assert sha(OUT/'VALIDATION_RESULTS68_V1.json')==selection['validation_results_sha256'];assert read(OUT/'EVALUATION_RESULTS68_V1.json')['selection_sha256']==sha(OUT/'VALIDATION_SELECTION68_V1.json')
for phase in ['VALIDATION','EVALUATION']:
 r=read(OUT/(phase+'_RESULTS68_V1.json'))
 for group in r['aggregate']:
  rr=[q for q in r['rows'] if q['method']==group['method']]
  if group['status']!='FINITE':assert any(q['status']!='FINITE' for q in rr);continue
  units={}
  for q in rr:units.setdefault(q['case'],[]).append(q['normalized_MAE'])
  expected=statistics.mean(statistics.mean(values) for values in units.values());check('aggregate_normalized_MAE',expected,group['normalized_MAE'],1e-10);counts['aggregate_groups']+=1
good=[r for r in read(OUT/'VALIDATION_RESULTS68_V1.json')['aggregate'] if r['status']=='FINITE'];winner=min([r for r in good if r['method'].startswith('latent_ridge_')],key=lambda r:(r['normalized_MAE'],r['method']));assert winner['method']==selection['selected']['latent']['method'];counts['validation_only_selections']+=1
for degree in [1,3]:
 winner=min([r for r in good if r['method'].startswith(f'raw_arx_d{degree}_')],key=lambda r:(r['normalized_MAE'],r['method']));row=next(q for q in all_rows if q['method']==winner['method']);assert row['alpha']==selection['selected']['raw'][str(degree)];counts['validation_only_selections']+=1
dump(OUT/'INDEPENDENT_REPLAY68_V2.json',dict(utc=utc(),status='PASS',counts=dict(counts),maximum_absolute_errors=errors,case_manifest_sha256=sha(OUT/'CASE_MANIFEST68_V1.json'),validation_results_sha256=sha(OUT/'VALIDATION_RESULTS68_V1.json'),selection_sha256=sha(OUT/'VALIDATION_SELECTION68_V1.json'),evaluation_results_sha256=sha(OUT/'EVALUATION_RESULTS68_V1.json'),scope='Independent scalar generator and all scores; augmented ridge fits, PCA via full SVD, fresh decoded latent rollouts and four fixed native calls; no claim of independent upstream pretraining reproduction',accepted_final_directions=0));print(json.dumps(dict(status='PASS',counts=dict(counts),maximum_absolute_errors=errors)),flush=True)
