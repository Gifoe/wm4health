"""Measure a strong conventional covariance-control operator before designing methods."""
import runtime_profile
import datetime,hashlib,importlib.util,json,pathlib,sys,time
from types import SimpleNamespace
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cost'
def measure(fn):
 with torch.inference_mode():
  for _ in range(4):fn()
  torch.cuda.synchronize();rows=[]
  for _ in range(3):
   begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   torch.cuda.synchronize();t=time.perf_counter();begin.record()
   for _ in range(5):fn()
   end.record();torch.cuda.synchronize()
   rows.append({'wall_ms_per_rollout':1000*(time.perf_counter()-t)/5,'CUDA_ms_per_rollout':begin.elapsed_time(end)/5})
  torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();output=fn();torch.cuda.synchronize()
  peak=torch.cuda.max_memory_allocated();del output
 return {'timings':rows,'baseline_allocated_bytes':base,'peak_allocated_bytes':peak,'peak_increment_bytes':peak-base}
def main():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'results.json').exists()
 torch.set_num_threads(1);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
 source=ROOT/'upstream/MTS3/agent/worldModels/gaussianTransformations/gaussian_marginalization.py'
 spec=importlib.util.spec_from_file_location('native_cost_marginal',source);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
 cfg=SimpleNamespace(bandwidth=3,trans_covar=.01,process_noise_hidden_units=[],process_noise_hidden_activation='ReLU')
 rows=[];horizon=32
 for d in [1,30,128]:
  for batch in [1,32]:
   model=native.Predict(d,act_dim=None,hierarchy_type=sys.intern('worker'),config=cfg).cuda().eval()
   state_dim=2*d;I=torch.eye(state_dim,device='cuda');perturb=torch.randn(state_dim,state_dim,device='cuda')
   A=.85*I+.02*perturb/torch.linalg.matrix_norm(perturb,ord=2);C=.1*I
   with torch.no_grad():
    for collection,full in [(model._A,A),(model._C,C)]:
     for parameter,part in zip(collection,[full[:d,:d],full[:d,d:],full[d:,:d],full[d:,d:]]):parameter.copy_(part)
   means=torch.zeros(batch,state_dim,device='cuda')
   state_cov=[torch.full((batch,d),.05,device='cuda'),torch.full((batch,d),.05,device='cuda'),torch.zeros(batch,d,device='cuda')]
   context_cov=[torch.full((batch,d),.5,device='cuda'),torch.full((batch,d),.5,device='cuda'),torch.zeros(batch,d,device='cuda')]
   F=torch.cat([torch.cat([A,C],1),torch.cat([torch.zeros_like(A),I],1)],0)
   P0=torch.diag(torch.cat([torch.full((state_dim,),.05,device='cuda'),torch.full((state_dim,),.5,device='cuda')])).expand(batch,-1,-1).clone()
   Q=torch.diag(torch.cat([torch.full((state_dim,),.01,device='cuda'),torch.zeros(state_dim,device='cuda')]))
   jm=torch.zeros(batch,2*state_dim,device='cuda')
   def native_rollout():
    mean=means;cov=state_cov
    for _ in range(horizon):mean,cov=model([mean,means],[cov,context_cov])
    return mean,cov
   def augmented_rollout():
    P=P0;mean=jm
    for _ in range(horizon):mean=mean@F.T;P=F@P@F.T+Q
    return mean,P
   n=measure(native_rollout);a=measure(augmented_rollout)
   ratio=np.median([x['wall_ms_per_rollout'] for x in a['timings']])/np.median([x['wall_ms_per_rollout'] for x in n['timings']])
   row={'latent_obs_dim':d,'fast_dim':2*d,'context_dim':2*d,'augmented_dim':4*d,'batch':batch,'horizon':horizon,
        'native_factorized':n,'ordinary_dense_augmented':a,'augmented_over_native_wall_ratio':float(ratio)}
   rows.append(row);print(json.dumps({'d':d,'batch':batch,'augmented_over_native_wall_ratio':float(ratio),'augmented_peak_increment_bytes':a['peak_increment_bytes']}),flush=True)
   del model,means,state_cov,context_cov,F,P0,Q,jm,A,C,I,perturb;torch.cuda.empty_cache()
 output={'rows':rows,'environment':{'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'float32','tf32':False},
         'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((ROOT/'ORDINARY_CONTROL_COST_PLAN.md').read_bytes()).hexdigest(),
         'scope':'Eager operator inference only; no learned-model relevance, training-memory or training-runtime claim',
         'objective_achieved':False,'novel_direction_accepted':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'results.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
 print(json.dumps({'cost_rows':len(rows),'objective_achieved':False}))
if __name__=='__main__':main()
