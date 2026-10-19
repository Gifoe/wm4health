"""Native moment operator vs persistent-context and ordinary exact controls."""
import runtime_profile
import datetime,hashlib,importlib.util,json,math,pathlib,sys,time
from types import SimpleNamespace
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/operator'
HORIZONS=[1,2,4,8,16,32,64]
CELLS=[(1.,1.,.5),(.95,1.,.5),(-.8,1.,.5),(0.,1.,.5),(.95,0.,.5),(.95,1.,0.)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(matrix):
 return [matrix[0,0].reshape(1,1),matrix[1,1].reshape(1,1),matrix[0,1].reshape(1,1)]
def unpack(blocks):
 return torch.stack([torch.cat([blocks[0][0],blocks[2][0]]),torch.cat([blocks[2][0],blocks[1][0]])])
def moments(a,c,s,h,q=.01):
 powers=np.power(a,np.arange(h));sum1=float(powers.sum());sum2=float(np.square(powers).sum())
 return a**(2*h)*.05+c*c*sum1*sum1*s+q*sum2,a**(2*h)*.05+(c*c*s+q)*sum2
def main():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'results.json').exists()
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 source=ROOT/'upstream/MTS3/agent/worldModels/gaussianTransformations/gaussian_marginalization.py'
 registry=json.loads((ROOT/'outputs/source/manifest.json').read_text())
 expected=next(r for r in registry['files'] if r['path']=='agent/worldModels/gaussianTransformations/gaussian_marginalization.py')
 assert sha(source)==expected['sha256']
 spec=importlib.util.spec_from_file_location('audited_native_gaussian_marginalization',source);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 device='cuda' if torch.cuda.is_available() else 'cpu';rows=[];arrays={};start=time.perf_counter()
 cfg=SimpleNamespace(bandwidth=0,trans_covar=.01,process_noise_hidden_units=[],process_noise_hidden_activation='ReLU')
 with torch.inference_mode():
  for hierarchy in ['worker','HIPRSSM']:
   for a in [0.,.5,.95,1.,-.8]:
    for c in [0.,.4,1.]:
     for s in [0.,.05,.5]:
      model=native.Predict(1,act_dim=None,hierarchy_type=sys.intern(hierarchy),config=cfg,dtype=torch.float64).to(device).double()
      for mats,value in [(model._A,a),(model._C,c)]:
       for k,parameter in enumerate(mats):parameter.fill_(value if k in [0,3] else 0.)
      model._log_process_noise._log_process_noise.fill_(math.log(.01))
      q=float(model.get_process_noise()[0][0,0])
      mean=torch.zeros(1,2,device=device);cov=pack(.05*torch.eye(2,device=device))
      context_mean=torch.zeros_like(mean);context_cov=pack(s*torch.eye(2,device=device))
      F=torch.cat([torch.cat([a*torch.eye(2,device=device),c*torch.eye(2,device=device)],1),torch.cat([torch.zeros(2,2,device=device),torch.eye(2,device=device)],1)],0)
      joint=torch.diag(torch.tensor([.05,.05,s,s],device=device));Q=torch.diag(torch.tensor([q,q,0.,0.],device=device))
      native_sequence=[];joint_sequence=[]
      for h in range(1,65):
       mean,cov=model([mean,context_mean],[cov,context_cov]);joint=F@joint@F.T+Q
       native_matrix=unpack(cov);native_sequence.append(float(native_matrix[0,0]));joint_sequence.append(float(joint[0,0]))
       if h in HORIZONS:
        persistent,fresh=moments(a,c,s,h,q)
        actual=float(native_matrix[0,0]);reference=float(joint[0,0])
        assert abs(actual-fresh)<1e-8,(a,c,s,h,actual,fresh)
        assert abs(reference-persistent)<1e-8
        ratio=actual/persistent
        coverage=math.erf(1.959963984540054*math.sqrt(ratio)/math.sqrt(2))
        rows.append({'native_branch':hierarchy,'a':a,'c':c,'context_variance':s,'horizon':h,'process_variance':q,
                     'native_variance':actual,'persistent_variance':persistent,'fresh_context_variance':fresh,
                     'augmented_state_variance':reference,'native_over_persistent_variance':ratio,
                     'persistent_nominal95_actual_marginal_coverage':coverage,
                     'expected_NLL_excess_nats_per_coordinate':.5*(1/ratio-1+math.log(ratio))})
      key=f'{hierarchy}_a{a}_c{c}_s{s}'
      arrays[key+'_native']=np.asarray(native_sequence);arrays[key+'_joint']=np.asarray(joint_sequence)
      del model
  print(json.dumps({'native_formula_rows_verified':len(rows)}),flush=True)
 # NumPy simulation independently checks persistent vs fresh latent semantics.
 mc=[];count=50000
 for index,(a,c,s) in enumerate(CELLS):
  rng=np.random.default_rng(2021+index);theta=rng.normal(0,math.sqrt(s),(count,2))
  xp=rng.normal(0,math.sqrt(.05),(count,2));xf=xp.copy();persistent_endpoints=[];fresh_endpoints=[]
  for h in range(1,65):
   epsilon=rng.normal(0,.1,(count,2));xp=a*xp+c*theta+epsilon
   fresh_theta=rng.normal(0,math.sqrt(s),(count,2));xf=a*xf+c*fresh_theta+epsilon
   if h in HORIZONS:
    vp,vf=moments(a,c,s,h);persistent_endpoints.append(xp.astype(np.float32));fresh_endpoints.append(xf.astype(np.float32))
    actualp=np.var(xp,axis=0,ddof=1);actualf=np.var(xf,axis=0,ddof=1)
    # Each coordinate is a separate Gaussian sample variance; 50k gives SE about .63%.
    assert max(np.abs(actualp/vp-1))<.04 and max(np.abs(actualf/vf-1))<.04
    mc.append({'case':index,'a':a,'c':c,'s':s,'horizon':h,'n_particles':count,
               'persistent_empirical_variances':actualp.tolist(),'fresh_empirical_variances':actualf.tolist(),
               'persistent_expected_variance':vp,'fresh_expected_variance':vf,
               'persistent_coverage_under_native95':np.mean(np.abs(xp)<=1.959963984540054*math.sqrt(vf),axis=0).tolist(),
               'persistent_coverage_under_augmented95':np.mean(np.abs(xp)<=1.959963984540054*math.sqrt(vp),axis=0).tolist()})
  np.savez_compressed(OUT/f'mc_case{index}.npz',persistent=np.stack(persistent_endpoints),fresh=np.stack(fresh_endpoints),theta=theta,horizons=np.asarray(HORIZONS))
 np.savez_compressed(OUT/'variance_sequences.npz',**arrays)
 output={'rows':rows,'monte_carlo':mc,'native_revision':registry['revision'],'native_source_sha256':sha(source),
         'plan_sha256':sha(ROOT/'PERSISTENT_CONTEXT_PLAN.md'),'script_sha256':sha(pathlib.Path(__file__)),
         'environment':{'python':sys.version,'torch':torch.__version__,'device':device,'dtype':'float64','gpu':torch.cuda.get_device_name() if device=='cuda' else None},
         'scope':'Prescribed admissible native operator parameters; not trained models, not original paper result reproduction, not a novel filtering algorithm',
         'elapsed_seconds':time.perf_counter()-start,'scientific_discriminator_complete':True,'trained_model_relevance_established':False,
         'novel_direction_accepted':False,'objective_achieved':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
 print(json.dumps({'rows':len(rows),'mc_rows':len(mc),'seconds':output['elapsed_seconds'],'objective_achieved':False}),flush=True)
if __name__=='__main__':main()
