"""Frozen source-derived discrete forcing-interface falsifier. No learned model."""
import runtime_profile
import csv,hashlib,json,pathlib,platform,time
import numpy as np
import torch

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/forcing_audit'
DT=.01;N=8192;STEPS=640;BURN=1000
SEEDS=[20262601,20262602,20262603]
K_VALUES=[1,4,16,64]
FORCES=np.array([0,.01,-.01,.1,-.1],dtype=np.float64)
A=np.array([[-2,.2,.1],[-.2,-60,0],[-.1,0,-40]],dtype=np.float64)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def gain(k):
 f=np.eye(3)+DT*A;p=np.eye(3);g=np.zeros(3)
 for _ in range(k):g+=DT*p[:,0];p=f@p
 return g
def step(x,noise,nonlinear):
 a,b,c=x.unbind(-1)
 drift=torch.stack((-2*a+.2*b+.1*c+nonlinear*a*b,-.2*a-nonlinear*a*a-60*b,-.1*a-40*c),dim=-1)
 x=x+DT*drift
 x[...,1:]+=noise
 return x

def score(states,methods,seed,nonlinear,k):
 # All saved states are float64 [method,force,time,system,coordinate].
 reference=states[0];rows=[]
 for mi,method in enumerate(methods):
  autonomous=float(np.max(np.abs(states[mi,0]-reference[0])))
  for fi,force in enumerate(FORCES[1:],1):
   truth=reference[fi]-reference[0];prediction=states[mi,fi]-states[mi,0]
   for dims,label in [([0],'observed_x1'),([0,1,2],'full_state')]:
    target=truth[...,dims];pred=prediction[...,dims]
    rms=np.sqrt(np.mean(target[1:]**2));assert rms>0
    mean_true=target.mean(1);mean_pred=pred.mean(1)
    mean_den=np.sqrt(np.mean(mean_true[1:]**2))
    row=dict(seed=seed,nonlinear=nonlinear,k=k,h=k*DT,method=method,force=float(force),coordinates=label,
       response_path_rel_rmse=float(np.sqrt(np.mean((pred[1:]-target[1:])**2))/rms),
       response_mean_rel_rmse=float(np.sqrt(np.mean((mean_pred[1:]-mean_true[1:])**2))/mean_den),
       response_mean_endpoint_true=float(mean_true[-1,0]),response_mean_endpoint_pred=float(mean_pred[-1,0]),
       response_mean_endpoint_rel_error=float(abs(mean_pred[-1,0]-mean_true[-1,0])/abs(mean_true[-1,0])),
       paired_endpoint_error_se=float((pred[-1,:,0]-target[-1,:,0]).std(ddof=1)/np.sqrt(N)),
       variance_response_endpoint_true=float(reference[fi,-1,:,dims[0]].var(ddof=1)-reference[0,-1,:,dims[0]].var(ddof=1)),
       variance_response_endpoint_pred=float(states[mi,fi,-1,:,dims[0]].var(ddof=1)-states[mi,0,-1,:,dims[0]].var(ddof=1)),
       autonomous_path_max_abs_error=autonomous)
    rows.append(row)
 return rows

def main():
 assert torch.cuda.is_available()
 assert not OUT.exists(),'Refuse overwrite scientific results'
 OUT.mkdir(parents=True)
 start=time.monotonic();torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
 device='cuda';u=torch.as_tensor(FORCES,device=device,dtype=torch.float64)[None,:,None]
 rows=[];files=[]
 for seed in SEEDS:
  rng=np.random.default_rng(seed)
  burn_noise=rng.normal(size=(BURN,N,2))*np.array([1.2,.8]) # sqrt(.01/epsilon)=1
  future_noise=rng.normal(size=(STEPS,N,2))*np.array([1.2,.8])
  bn=torch.as_tensor(burn_noise,device=device);fn=torch.as_tensor(future_noise,device=device)
  initial=[]
  for nonlinear in [0,1]:
   x=torch.zeros((N,3),device=device,dtype=torch.float64)
   for t in range(BURN):x=step(x,bn[t],nonlinear)
   initial.append(x.detach().clone())
  inputs=OUT/f'inputs_{seed}.npz'
  np.savez_compressed(inputs,initial=torch.stack(initial).cpu().numpy(),noise=future_noise,forces=FORCES)
  files.append(dict(path=inputs.name,sha256=sha(inputs),bytes=inputs.stat().st_size))
  del bn,burn_noise
  for nonlinear in [0,1]:
   for k in K_VALUES:
    methods=['fine_reference','endpoint','symmetric','split4']+(['exact_discrete_input'] if nonlinear==0 else [])
    x=initial[nonlinear][None,None].expand(len(methods),5,N,3).clone()
    snapshots=[x.clone()];sub=max(1,k//4)
    g=torch.as_tensor(gain(k),device=device)
    for t in range(STEPS):
     if k>1 and t%k==0:x[2,:,:,0]+=.5*k*DT*u[0]
     x=step(x,fn[t],nonlinear)
     x[0,:,:,0]+=DT*u[0]
     if (t+1)%k==0:
      x[1,:,:,0]+=k*DT*u[0]
      x[2,:,:,0]+=(.5 if k>1 else 1)*k*DT*u[0]
      if nonlinear==0:x[4]+=u[0,:,:,None]*g
     if (t+1)%sub==0:x[3,:,:,0]+=sub*DT*u[0]
     if (t+1)%64==0:snapshots.append(x.clone())
    states=torch.stack(snapshots,dim=2).cpu().numpy();assert np.isfinite(states).all()
    output=OUT/f'states_i{nonlinear}_k{k}_seed{seed}.npz'
    np.savez_compressed(output,states=states,methods=np.asarray(methods),times=np.arange(11)*.64)
    files.append(dict(path=output.name,sha256=sha(output),bytes=output.stat().st_size))
    new=score(states,methods,seed,nonlinear,k);rows.extend(new)
    assert max(r['autonomous_path_max_abs_error'] for r in new)==0
    if k==1 or k==4:
     assert max(r['response_path_rel_rmse'] for r in new if r['method']=='split4')==0
    if nonlinear==0:
     assert max(r['response_path_rel_rmse'] for r in new if r['method']=='exact_discrete_input')<1e-10
    print(json.dumps(dict(seed=seed,nonlinear=nonlinear,k=k,rows=len(rows),elapsed_s=time.monotonic()-start)),flush=True)
    del states,snapshots,x
  del fn,future_noise,initial
 output=OUT/'rows.csv'
 with output.open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 files.append(dict(path=output.name,sha256=sha(output),bytes=output.stat().st_size))
 result=dict(status='completed',scope='controlled source-derived forcing-interface audit, no learned model',n_systems_per_mc=N,
    seeds=SEEDS,n_rows=len(rows),source_fine_dt=DT,source_fine_steps=STEPS,coarse_strides=K_VALUES,
    source='Falasca2506.22552v4 partial-observation triad; full-state audit extension',
    runtime_s=time.monotonic()-start,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name()),
    script_sha256=sha(pathlib.Path(__file__)),protocol_sha256=sha(ROOT/'PROTOCOL.md'),raw_files=files)
 (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='raw_files'}),flush=True)

if __name__=='__main__':
 with torch.inference_mode():main()
