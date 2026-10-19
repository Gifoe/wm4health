"""Independent CPU/NumPy replay, score and matrix-power checks; no main imports."""
import runtime_profile
import csv,hashlib,json,pathlib,time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/forcing_audit'
A=np.array([[-2,.2,.1],[-.2,-60,0],[-.1,0,-40]],dtype=np.float64)
F=np.eye(3)+.01*A;U=np.array([0,.01,-.01,.1,-.1])
IDS=[0,17,255,2048,8191]
def advance(x,z,nonlinear):
 delta=x@A.T
 delta[...,0]+=nonlinear*x[...,0]*x[...,1]
 delta[...,1]-=nonlinear*x[...,0]**2
 result=x+.01*delta
 result[...,1]+=z[...,0];result[...,2]+=z[...,1]
 return result
def replay(initial,noise,k,nonlinear,method):
 x=np.broadcast_to(initial[None],(5,len(IDS),3)).copy();saved=[x.copy()]
 interval=max(k//4,1) if method=='split4' else k
 g=np.linalg.solve(A,np.linalg.matrix_power(F,k)-np.eye(3))[:,0]
 for t in range(640):
  if method=='symmetric' and k>1 and t%k==0:x[...,0]+=.005*k*U[:,None]
  x=advance(x,noise[t],nonlinear)
  if method=='fine_reference':x[...,0]+=.01*U[:,None]
  elif method=='exact_discrete_input':
   if (t+1)%k==0:x+=U[:,None,None]*g
  elif method=='symmetric':
   if (t+1)%k==0:x[...,0]+=(.005 if k>1 else .01)*k*U[:,None]
  elif (t+1)%interval==0:x[...,0]+=.01*interval*U[:,None]
  if (t+1)%64==0:saved.append(x.copy())
 return np.stack(saved,axis=1)

def main():
 started=time.monotonic();manifest=json.loads((OUT/'results.json').read_text())
 assert manifest['status']=='completed'
 maxima=dict(replay_abs=0.,linear_response_abs=0.,score_abs=0.)
 counts=dict(hash_files=0,replayed_paths=0,analytic_values=0,score_values=0)
 for file in manifest['raw_files']:
  path=OUT/file['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==file['sha256'];counts['hash_files']+=1
 with (OUT/'rows.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 indexed={(int(r['seed']),int(r['nonlinear']),int(r['k']),r['method'],float(r['force']),r['coordinates']):r for r in rows}
 for seed in manifest['seeds']:
  data=np.load(OUT/f'inputs_{seed}.npz');initial=data['initial'];noise=data['noise']
  for nonlinear in [0,1]:
   for k in manifest['coarse_strides']:
    saved=np.load(OUT/f'states_i{nonlinear}_k{k}_seed{seed}.npz');states=saved['states'];methods=saved['methods'].tolist()
    truth=states[0]
    for mi,method in enumerate(methods):
     checked=replay(initial[nonlinear,IDS],noise[:,IDS],k,nonlinear,method)
     actual=np.take(states[mi],IDS,axis=2)
     maximum=float(np.max(np.abs(actual-checked)));maxima['replay_abs']=max(maxima['replay_abs'],maximum)
     assert maximum<3e-13,(seed,nonlinear,k,method,maximum)
     counts['replayed_paths']+=len(IDS)*5
     if nonlinear==0:
      interval=max(k//4,1) if method=='split4' else k
      if method=='fine_reference':interval=1
      transition=np.linalg.matrix_power(F,interval)
      if method in ['fine_reference','exact_discrete_input']:
       g=np.linalg.solve(A,transition-np.eye(3))[:,0]
      elif method=='symmetric' and k>1:g=.005*k*(transition+np.eye(3))[:,0]
      else:g=.01*interval*np.eye(3)[:,0]
      expected=np.zeros((5,11,3))
      current=np.zeros((5,3));saved_t=1
      for t in range(interval,641,interval):
       current=current@transition.T+U[:,None]*g
       if t%64==0:expected[:,saved_t]=current;saved_t+=1
      observed=states[mi]-states[mi,0:1]
      maximum=float(np.max(np.abs(observed-expected[:,:,None,:])))
      assert maximum<3e-13,(method,maximum)
      maxima['linear_response_abs']=max(maxima['linear_response_abs'],maximum);counts['analytic_values']+=int(observed.size)
     for fi,force in enumerate(U[1:],1):
      for dims,label in [(slice(0,1),'observed_x1'),(slice(None),'full_state')]:
       target=(truth[fi]-truth[0])[1:,:,dims];prediction=(states[mi,fi]-states[mi,0])[1:,:,dims]
       err=prediction-target;mu=prediction.mean(axis=1);muref=target.mean(axis=1)
       values=dict(response_path_rel_rmse=float(np.linalg.norm(err.ravel())/np.linalg.norm(target.ravel())),
          response_mean_rel_rmse=float(np.linalg.norm((mu-muref).ravel())/np.linalg.norm(muref.ravel())),
          response_mean_endpoint_true=float(muref[-1,0]),response_mean_endpoint_pred=float(mu[-1,0]),
          response_mean_endpoint_rel_error=float(abs(mu[-1,0]-muref[-1,0])/abs(muref[-1,0])),
          paired_endpoint_error_se=float(np.std(err[-1,:,0],ddof=1)/np.sqrt(target.shape[1])),
          variance_response_endpoint_true=float(np.var(truth[fi,-1,:,0],ddof=1)-np.var(truth[0,-1,:,0],ddof=1)),
          variance_response_endpoint_pred=float(np.var(states[mi,fi,-1,:,0],ddof=1)-np.var(states[mi,0,-1,:,0],ddof=1)),
          autonomous_path_max_abs_error=float(np.max(np.abs(states[mi,0]-truth[0]))))
       row=indexed[(seed,nonlinear,k,method,float(force),label)]
       for name,value in values.items():
        difference=abs(value-float(row[name]));maxima['score_abs']=max(maxima['score_abs'],difference)
        assert difference<1e-11,(name,difference);counts['score_values']+=1
    print(json.dumps(dict(seed=seed,nonlinear=nonlinear,k=k,elapsed_s=time.monotonic()-started)),flush=True)
 result=dict(status='passed',maxima=maxima,counts=counts,runtime_s=time.monotonic()-started,
    main_results_sha256=hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),
    verifier_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    limitation='Discrete reference only; no continuous-SDE convergence or learned-model capability claim')
 (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
