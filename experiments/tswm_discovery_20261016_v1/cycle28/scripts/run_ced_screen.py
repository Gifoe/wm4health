"""Frozen same-information full official CED split; no test selection or neural fitting."""
import runtime_profile
import argparse,csv,hashlib,json,pathlib,sys,time
import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import norm
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/ced_screen';OLD=ROOT.parent/'cycle16'
LEVELS=np.arange(1,10)/10
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def sync():torch.cuda.synchronize()
def arx_fit(y,u,order,lam):
 indices=np.arange(order,len(y));x=np.column_stack([y[indices-j] for j in range(1,order+1)]+[u[indices-j] for j in range(order)])
 mean=x.mean(0);scale=np.maximum(x.std(0),1e-8);z=np.column_stack((np.ones(len(x)),(x-mean)/scale));penalty=np.eye(z.shape[1])*lam;penalty[0,0]=0
 coef=np.linalg.solve(z.T@z/len(z)+penalty,z.T@y[indices]/len(z));residual=y[indices]-z@coef
 return dict(order=order,lam=lam,mean=mean,scale=scale,coefficient=coef,innovation_variance=float(residual@residual/max(1,len(residual)-len(coef))))
def arx_predict(fit,observed,u,h):
 order=int(fit['order']);history=list(observed);start=len(history);output=[]
 for t in range(start,start+h):
  x=np.r_[[history[t-j] for j in range(1,order+1)],[u[t-j] for j in range(order)]]
  value=float(np.r_[1,(x-fit['mean'])/fit['scale']]@fit['coefficient']);history.append(value);output.append(value)
 a=fit['coefficient'][1:1+order]/fit['scale'][:order];psi=np.zeros(h);psi[0]=1
 for t in range(1,h):psi[t]=sum(a[j-1]*psi[t-j] for j in range(1,min(t,order)+1))
 return np.array(output),float(fit['innovation_variance'])*np.cumsum(psi**2)
def scores(pred,q,target,scale):
 error=pred-target;r=dict(rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(np.abs(error))),normalized_rmse=float(np.sqrt(np.mean(error**2))/scale),finite_fraction=float(np.mean(np.isfinite(pred))))
 if q is not None:
  e=target[:,None]-q;r.update(pinball9=float(np.mean(np.maximum(LEVELS*e,(LEVELS-1)*e))),coverage10_90=float(np.mean((target>=q[:,0])&(target<=q[:,-1]))),mean_width10_90=float(np.mean(q[:,-1]-q[:,0])))
 return r
def main():
 assert not OUT.exists();OUT.mkdir();started=time.monotonic();torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats();torch.manual_seed(20262801)
 source=json.loads((ROOT/'outputs/source/pcfm_manifest.json').read_text())
 for row in source['files']:assert sha(ROOT/'upstream/PCFM'/row['path'])==row['sha256']
 oldsource=json.loads((OLD/'outputs/source/metaid/manifest.json').read_text())
 for row in oldsource['files']:assert sha(OLD/'upstream/sysid_benchmarks'/row['path'])==row['sha256']
 raw=ROOT/'outputs/data/CED/DATAUNIF.MAT';assert sha(raw)=='102c8d8012d6e850d50c13227b8c2280fe69f8bc4a845177f8bc3d761d152c89'
 physical=loadmat(raw);weight=OLD/'outputs/external/metaid/ckpt_430_fine-tuned_3100000_lower_noise.pt';assert sha(weight)=='e8324becad85ac610ec94f8ae09048938b343761a5bcd3967a86353a5bf28197'
 unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(weight));assert unsafe<={'argparse.Namespace'}
 with torch.serialization.safe_globals([argparse.Namespace]):payload=torch.load(weight,map_location='cpu',weights_only=True)
 sys.path.insert(0,str(OLD/'upstream/sysid_benchmarks'));from transformer_sim import Config,TSTransformer
 icl=TSTransformer(Config(**payload['model_args'])).cuda().eval();icl.load_state_dict(payload['model'],strict=True)
 sys.path.insert(0,str(ROOT/'upstream/PCFM'));from model import PCFMPredictor
 pcfm_weight=ROOT/'outputs/external/PCFM/model.safetensors';assert sha(pcfm_weight)=='746ff3eb69f9fcaa0e934f13b5afda17be7591a987eb64c671fa05b1d87b6755'
 pcfm=PCFMPredictor.from_pretrained(str(pcfm_weight),device='cuda');sync();load_s=time.monotonic()-started
 rows=[];predictions=[];validation=[];scaling=[];checks=[]
 for label,uk,yk in [('low_amplitude','u11','z11'),('high_amplitude','u12','z12')]:
  u=physical[uk][:,0].astype(np.float64);y=physical[yk][:,0].astype(np.float64);assert len(u)==len(y)==500
  h=90;start=410;target=y[start:];scale=float(np.std(y[:400]));window=np.stack((u,np.r_[y[:start],np.zeros(h)]))
  np.savez_compressed(OUT/f'input_{label}.npz',u=u,y=y,model_window=window,train_stop=np.array(400),forecast_start=np.array(start))
  def record(method,pred,q,elapsed,**extra):
   path=OUT/f'{label}_{method}.npz';parts=dict(mean=pred,target=target)
   if q is not None:parts['quantiles']=q
   np.savez_compressed(path,**parts);predictions.append(dict(path=path.name,sha256=sha(path)))
   rows.append(dict(record=label,method=method,forward_s=elapsed,**extra,**scores(pred,q,target,scale)))
  # Actual future outputs are never supplied to either model.
  pcfm.what_if(window,context_length=start,masked_nodes=[1],graph='CD',glue=False,return_quantiles=True);sync()
  for graph_name,graph in [('CD','CD'),('known_direction',[(0,1,0.)]),('CP','CP')]:
   for glue in [True,False]:
    sync();before=time.monotonic();q=pcfm.what_if(window,context_length=start,masked_nodes=[1],graph=graph,glue=glue,return_quantiles=True)[1,start:].cpu().numpy();sync()
    record(f'pcfm_{graph_name}_glue{int(glue)}',q[:,4],q,time.monotonic()-before)
  values={}
  for name,data in [('u',u),('y',y)]:
   t=torch.tensor(data[:400],dtype=torch.float32);values[name]=(float(t.mean()),float(t.std()+1e-6))
  scaling.append(dict(record=label,u_mean=values['u'][0],u_std=values['u'][1],y_mean=values['y'][0],y_std=values['y'][1],n_train=400))
  uz=(u-values['u'][0])/values['u'][1];yz=(y[:start]-values['y'][0])/values['y'][1]
  ctx_u=torch.tensor(uz[:400],dtype=torch.float32,device='cuda')[None,:,None];ctx_y=torch.tensor(yz[:400],dtype=torch.float32,device='cuda')[None,:,None]
  for mode,n_in,pad in [('native_padded30',30,20),('direct10',10,0)]:
   qu=np.r_[np.zeros(pad),uz[400:]];qy=np.r_[np.zeros(pad),yz[400:],np.zeros(h)]
   qu=torch.tensor(qu,dtype=torch.float32,device='cuda')[None,:,None];qy=torch.tensor(qy,dtype=torch.float32,device='cuda')[None,:,None]
   with torch.inference_mode():icl(ctx_y,ctx_u,qu,qy,n_in)
   sync();before=time.monotonic()
   with torch.inference_mode():mean,std=icl(ctx_y,ctx_u,qu,qy,n_in)[:2]
   sync();elapsed=time.monotonic()-before
   mu=mean[0,:,0].cpu().numpy()*values['y'][1]+values['y'][0];sigma=std[0,:,0].cpu().numpy()*values['y'][1];assert np.min(sigma)>0
   q=mu[:,None]+sigma[:,None]*norm.ppf(LEVELS);record('icl_'+mode,mu,q,elapsed)
   alt_y=qy.clone();alt_y[:,n_in:]=123.;alt_u=qu.clone();alt_u[:,n_in+45:]=torch.flip(alt_u[:,n_in+45:],dims=[1])
   with torch.inference_mode():
    a=icl(ctx_y,ctx_u,qu,alt_y,n_in)[:2];b=icl(ctx_y,ctx_u,alt_u,qy,n_in)[:2]
   placeholder=max(float((v-w).abs().max()) for v,w in zip((mean,std),a));prefix=max(float((v[:,:45]-w[:,:45]).abs().max()) for v,w in zip((mean,std),b));assert placeholder<2e-5 and prefix<2e-5
   np.savez_compressed(OUT/f'icl_inputs_{label}_{mode}.npz',ctx_y=ctx_y.cpu().numpy(),ctx_u=ctx_u.cpu().numpy(),qu=qu.cpu().numpy(),qy=qy.cpu().numpy(),mean=mean.cpu().numpy(),std=std.cpu().numpy(),n_in=np.array(n_in),placeholder_mean=a[0].cpu().numpy(),control_alt=alt_u.cpu().numpy(),prefix_alt_mean=b[0].cpu().numpy())
   checks.append(dict(record=label,mode=mode,placeholder_max_abs=placeholder,late_control_prefix_max_abs=prefix))
  candidates=[];before=time.monotonic()
  for order in [1,2,4,8,16,32]:
   for lam in [1e-5,1e-3,.1,10.]:
    fit=arx_fit(y[:300],u[:300],order,lam);mu,_=arx_predict(fit,y[:300],u,100);mse=float(np.mean((mu-y[300:400])**2))
    candidates.append((mse,order,lam));validation.append(dict(record=label,order=order,lam=lam,free_run_validation_mse=mse))
  selected=min(candidates);fit=arx_fit(y[:400],u[:400],selected[1],selected[2]);mu,var=arx_predict(fit,y[:start],u,h);elapsed=time.monotonic()-before
  np.savez_compressed(OUT/f'arx_{label}.npz',**fit);record('arx_train_selected',mu,mu[:,None]+np.sqrt(var)[:,None]*norm.ppf(LEVELS),elapsed,order=selected[1],lam=selected[2])
  record('persistence',np.repeat(y[start-1],h),None,0.);record('training_mean',np.repeat(y[:400].mean(),h),None,0.)
  print(json.dumps(dict(phase='record_complete',record=label,rows=len(rows))),flush=True)
 for name,table in [('rows.csv',rows),('arx_validation.csv',validation)]:
  with (OUT/name).open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in table for k in row)));w.writeheader();w.writerows(table)
 result=dict(status='completed',n_rows=len(rows),predictions=predictions,scaling=scaling,checks=checks,load_s=load_s,runtime_s=time.monotonic()-started,
  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),source_pcfm_revision=source['revision'],source_icl_revision=oldsource['revision'],data_sha256=sha(raw),
  pcfm_weight_sha256=sha(pcfm_weight),icl_weight_sha256=sha(weight),plan_sha256=sha(ROOT/'CED_SCREEN_PLAN.md'),script_sha256=sha(pathlib.Path(__file__)),
  scope='Full two official CED record test splits,10initial outputs,90free forecasts. Not independent-plant population,matched pretraining compute,or counterfactual ground truth',accepted_final_directions=0,new_neural_training=False)
 write(OUT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['predictions','scaling','checks']}),flush=True)
if __name__=='__main__':main()
