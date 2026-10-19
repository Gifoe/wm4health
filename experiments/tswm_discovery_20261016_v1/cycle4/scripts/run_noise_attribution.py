"""One fixed budget extension and an existing unlabeled spectral initialization.

Motivated by decreasing pilot loss, not a proposed new method. No test-set tuning.
This retains the published objective implementations and the same linear adapters.
"""
import datetime,hashlib,json,pathlib,time
import numpy as np
import torch
from run_noise_objective_pilot import ROOT,DEVICE,DIM,BATCH,SEED,dump,seed_all,build,make_data,probe
STEPS=8000
def fit_training_cca(data):
 x=data['pixels'][:,:-1].reshape(-1,DIM).double();y=data['pixels'][:,1:].reshape(-1,DIM).double()
 x=x-x.mean(0);y=y-y.mean(0)
 def ir(c):
  v,u=torch.linalg.eigh(c);return (u*v.clamp_min(1e-5).rsqrt())@u.T
 a=ir(x.T@x/len(x));b=ir(y.T@y/len(y));u,sv,_=torch.linalg.svd(a@(x.T@y/len(x))@b)
 return (a@u[:,:2]).T.float(),sv[:6].cpu().tolist()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'outputs'/'noise_attribution';out.mkdir(parents=True,exist_ok=True)
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Pilot loss decreasing and all neural variants incomplete; reject premature structural attribution','data':'same fixed neural training observation/action cohort9051; independent probe9052 and development9053','source_revision':'b3c2fb374b7c69f169de15774f371290a8736f56','regime':'independent_nuisance','seed':SEED,'steps':STEPS,'batch':BATCH,'lr':.0005,'weight_decay':.001,'runs':['lewm_native_weight_random','lewm_matched_static_random','motion_random','lewm_matched_static_trainingCCA','motion_trainingCCA'],'initialization_control':'Main encoder initialized from ordinary CCA/VAMP-style subspace estimated on identical training observations only; no state labels. Difference branch/predictor retain original initialization.','checkpoint':'fixed final step; report2000/8000 and intermediate state diagnostics only after training finishes','decision':'If budget or existing unlabeled initialization repairs the common deficit, do not claim new mechanism necessary. The noise-specific Motion gate already failed in the first pilot.'}
 dump(out/'plan.json',plan)
 gen=torch.Generator();gen.manual_seed(9001);q=torch.linalg.qr(torch.randn(DIM,DIM,generator=gen)).Q.to(DEVICE)
 training,_=make_data(q,8192,4,'independent_nuisance',9051)
 prober=make_data(q,4096,4,'independent_nuisance',9052);dev=make_data(q,8192,4,'independent_nuisance',9053)
 initial,sv=fit_training_cca(training);np.save(out/'training_cca_encoder.npy',initial.cpu().numpy());dump(out/'training_cca_info.json',{'singular_values':sv,'fitted_on_state_labels':False,'observations':8192*4,'transitions':8192*3})
 rows=[]
 for name in plan['runs']:
  run=out/name;run.mkdir(exist_ok=True)
  if (run/'result.json').exists():rows.append(json.loads((run/'result.json').read_text()));continue
  variant=name.removesuffix('_random').removesuffix('_trainingCCA');spectral=name.endswith('_trainingCCA')
  seed_all(SEED);model=build(variant)
  if spectral:model.encoder.map.weight.data.copy_(initial)
  opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
  sampler=torch.Generator(device=DEVICE);sampler.manual_seed(9061);start=time.perf_counter();snapshots=[]
  with (run/'train.jsonl').open('w',encoding='utf-8') as logfile:
   for step in range(1,STEPS+1):
    model.train();idx=torch.randint(0,8192,(BATCH,),generator=sampler,device=DEVICE);batch={k:v[idx] for k,v in training.items()}
    opt.zero_grad(set_to_none=True);_,losses=model.train_step(batch);losses['loss'].backward();assert torch.isfinite(losses['loss']);opt.step()
    if step==1 or step%500==0:
     log={'step':step,**{k:float(v.detach()) for k,v in losses.items()},'seconds':time.perf_counter()-start};logfile.write(json.dumps(log)+'\n');logfile.flush();dump(out/'status.json',{'state':'training','run':name,'step':step,'completed':len(rows),'total':len(plan['runs'])})
    # Save immutable states, do not inspect their physical probes during training.
    if step in [2000,4000,8000]:torch.save(model.state_dict(),run/f'model_step{step}.pt')
  for step in [2000,4000,8000]:
   path=run/f'model_step{step}.pt';model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True));snapshots.append({'step':step,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**probe(model,prober,dev,q)})
  result={'run':name,'seed':SEED,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'snapshots':snapshots};dump(run/'result.json',result);rows.append(result);dump(out/'results.json',rows);print(json.dumps(result),flush=True)
  del model;torch.cuda.empty_cache()
 dump(out/'status.json',{'state':'complete','completed':len(rows),'total':len(plan['runs']),'candidate_accepted':False})
if __name__=='__main__':main()
