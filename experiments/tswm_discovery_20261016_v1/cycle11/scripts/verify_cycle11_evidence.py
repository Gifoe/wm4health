"""Replay saved draws, validate the score analytically and inspect all checkpoints."""
import runtime_profile
import hashlib,json,math,pathlib
import numpy as np
import torch
from scipy.stats import norm
from sklearn.linear_model import Ridge
from learn_score_audit import Model,weighted_ES
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
 oracle=read(OUT/'score_folding/results.json');learn=read(OUT/'score_learning/results.json')
 assert len(oracle['score_rows'])==60 and len(oracle['density_rows'])==15 and len(learn['rows'])==12
 sourcecounts={}
 for key in ['lens','infoldm','ac_mtm']:
  pin=read(ROOT/f'sources/{key}/SOURCE_PIN.json')
  for r in pin['files']:assert sha(ROOT/'sources'/key/r['path'])==r['sha256']
  sourcecounts[key]=len(pin['files'])
 cfg=oracle['config'];theta=cfg['rotation_radians'];rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
 rho=cfg['rho'];variance=1-rho*rho;replayed=0;maximum_error=0.;analytic_checks=[]
 def polar(x):
  angle=2*np.arctan2(x[...,1],x[...,0]);radius=np.linalg.norm(x,axis=-1)
  return np.stack([radius*np.cos(angle),radius*np.sin(angle)],axis=-1)
 def trajectory(x,e,u,p):
  states=[];state=x.copy()
  for t in range(e.shape[1]):
   state=rho*np.einsum('ij,nj->ni',rot,state)*np.where(u[:,t,None]<p,-1.,1.)+math.sqrt(variance)*e[:,t];states.append(state.copy())
  return np.stack(states,axis=1)
 def distance(a,b):return np.sqrt(np.sum((a-b)**2,axis=(1,2))/a.shape[1])
 for seed in cfg['seeds']:
  d=np.load(OUT/'score_folding'/f'design_seed{seed}.npz');x=d['context'];assert x.shape==(32768,2)
  for p in cfg['flip_probabilities']:
   a=trajectory(x,d['innovation1'],d['flip_uniform1'],p);b=trajectory(x,d['innovation2'],d['flip_uniform2'],p);fa,fb=polar(a),polar(b)
   mu=rho*np.einsum('ij,nj->ni',rot,x);positive=norm.logpdf(a[:,0],loc=mu,scale=math.sqrt(variance)).sum(-1);negative=norm.logpdf(a[:,0],loc=-mu,scale=math.sqrt(variance)).sum(-1)
   true=np.logaddexp(positive+np.log1p(-p),negative+(np.log(p) if p else -np.inf));folded=np.logaddexp(positive,negative)-np.log(2.)
   density=next(r for r in oracle['density_rows'] if r['seed']==seed and r['p']==p)
   assert abs(float(np.mean(true-folded))-density['predictive_MI_lost_by_fold']['mean'])<1e-12
   assert abs(density['analytic_raw_MSE_increase']-2*rho*rho*(1-2*p)**2)<1e-14
   if p==.5:assert density['predictive_MI_lost_by_fold']['mean']==0
   for H in cfg['horizons']:
    for normalization in ['Gaussian radial','unit circle']:
     aa,bb,ff,gg=a[:,:H].copy(),b[:,:H].copy(),fa[:,:H].copy(),fb[:,:H].copy()
     if normalization=='unit circle':
      aa/=np.linalg.norm(aa,axis=-1,keepdims=True);bb/=np.linalg.norm(bb,axis=-1,keepdims=True);ff/=np.linalg.norm(ff,axis=-1,keepdims=True);gg/=np.linalg.norm(gg,axis=-1,keepdims=True)
     row=next(r for r in oracle['score_rows'] if r['seed']==seed and r['p']==p and r['horizon']==H and r['normalization']==normalization)
     identity=.5*distance(aa,bb);fold=.5*distance(ff,gg)
     error=max(abs(identity.mean()-row['identity_oracle_ES']['mean']),abs(fold.mean()-row['fold_oracle_ES']['mean']),abs((fold-identity).mean()-row['fold_minus_identity_ES']['mean']))
     maximum_error=max(maximum_error,float(error));assert error<1e-11
     if normalization=='Gaussian radial':
      q1=a[:,:H]*d['initial_sign1'][:,None,:];q2=b[:,:H]*d['initial_sign2'][:,None,:]
      assert abs(.5*distance(q1,q2).mean()-row['raw_folded_context_oracle_ES']['mean'])<1e-11
     if H==1 and normalization=='Gaussian radial':
      prob=2*p*(1-p);expected=.5*math.sqrt(math.pi)*((1-prob)*math.sqrt(1-rho*rho)+prob*math.sqrt(1+rho*rho));se=row['identity_oracle_ES']['iid_context_standard_error'];assert abs(identity.mean()-expected)<8*se
      analytic_checks.append({'seed':seed,'p':p,'analytic_identity_Bayes_ES':expected,'MC_identity_Bayes_ES':float(identity.mean()),'MC_standard_error':se})
     replayed+=1
 # An independent known proper-score optimum checks the changed weighted-MC path.
 torch.set_num_threads(1);normal_checks=[]
 for scale in [0.,.5,1.,2.]:
  values=[]
  for seed in range(3):
   torch.manual_seed(3021+seed);noise=torch.randn(4096,8,4,2);target=torch.randn(4096,2);weights=torch.full((4096,8),1/8)
   values.append(float(weighted_ES(scale*noise,weights,target)))
  expected=math.sqrt(math.pi/2)*math.sqrt(1+scale*scale)-math.sqrt(math.pi)*scale/2
  assert abs(np.mean(values)-expected)<.035
  normal_checks.append({'forecast_std':scale,'expected_ES':expected,'MC_mean':float(np.mean(values)),'MC_seed_values':values})
 assert normal_checks[2]['MC_mean']<min(normal_checks[i]['MC_mean'] for i in [0,1,3])
 states=np.load(OUT/'score_learning/evaluation_states.npz');px=states['probe_train'];tx=states['probe_test'];checkpoint_rows=[]
 curves=read(OUT/'score_learning/training_curves.json');assert len(curves)==156
 for r in learn['rows']:
  key=f"{r['arm']}_seed{r['seed']}";lat=np.load(OUT/'score_learning'/f'latents_{key}.npz');ztrain=lat['train'];ztest=lat['test'];neg=lat['negated_test']
  assert all(np.isfinite(lat[k]).all() for k in lat.files)
  checkpoint=torch.load(OUT/'score_learning'/f'{key}.pt',map_location='cpu',weights_only=True);model=Model();model.load_state_dict(checkpoint['state_dict']);model.eval()
  with torch.no_grad():encoded=model.encoder(torch.tensor(tx,dtype=torch.float32)).numpy()
  encoding_error=float(np.max(np.abs(encoded-ztest)));assert encoding_error<1e-5
  cov=np.cov(ztest.T);fiber=np.mean(np.sum((ztest-neg)**2,axis=1))/(2*np.trace(cov));assert abs(fiber-r['sign_fiber_separation_ratio'])<1e-12
  mean=ztrain.mean(0);std=ztrain.std(0).clip(1e-6);estimator=Ridge(alpha=1e-4).fit((ztrain-mean)/std,px);pred=estimator.predict((ztest-mean)/std);mse=float(np.mean(np.sum((pred-tx)**2,axis=1)))
  assert abs(mse-r['probes']['ridge']['raw_state_MSE'])<1e-10
  assert checkpoint['config']==learn['config'] and checkpoint['row']==r
  subset=[c for c in curves if c['seed']==r['seed'] and c['arm']==r['arm']];assert len(subset)==13 and subset[-1]['update']==6000
  checkpoint_rows.append({'seed':r['seed'],'arm':r['arm'],'checkpoint_encoding_max_error_CPU_vs_GPU':encoding_error,'replayed_ridge_state_MSE':mse})
 summary={'oracle_grid_rows_replayed':replayed,'oracle_replay_max_error':maximum_error,'native_source_files_verified':sourcecounts,'analytic_identity_energy_checks':analytic_checks,'weighted_ES_independent_normal_checks':normal_checks,'checkpoints_verified':checkpoint_rows,'all_checks_passed':True,'objective_achieved':False,'scientific_jobs_complete':True,'novel_direction_accepted':False,'interpretation':'Constructed criterion-level preference is real; all tested random-initialized energy-only encoders retain the observed state. This is negative evidence for a new method on this task.'}
 (OUT/'evidence_verification.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({'all_checks_passed':True,'oracle_rows':replayed,'checkpoints':len(checkpoint_rows),'objective_achieved':False}))
if __name__=='__main__':main()
