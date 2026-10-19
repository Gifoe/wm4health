"""Independent data/split, augmented-LS ARX and scores; separate native neural replay."""
import runtime_profile
import argparse,csv,hashlib,json,pathlib,sys,time
import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import norm
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/ced_screen';OLD=ROOT.parent/'cycle16';Q=np.arange(1,10)/10
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fit(y,u,k,lam):
 features=[];target=[]
 for t in range(k,len(y)):
  features.append(list(y[t-k:t][::-1])+list(u[t-k+1:t+1][::-1]));target.append(y[t])
 x=np.array(features);target=np.array(target);mean=x.mean(0);scale=np.maximum(x.std(0),1e-8);x=np.column_stack((np.ones(len(x)),(x-mean)/scale))
 penalty=np.diag(np.r_[0,np.repeat(np.sqrt(lam*len(x)),2*k)]);coef=np.linalg.lstsq(np.vstack((x,penalty)),np.r_[target,np.zeros(2*k+1)],rcond=None)[0]
 variance=float(np.sum((target-x@coef)**2)/max(1,len(target)-len(coef)))
 return dict(mean=mean,scale=scale,coefficient=coef,order=k,innovation_variance=variance)
def predict(f,y,u,h):
 k=int(f['order']);coefficient=f['coefficient'][1:]/f['scale'];intercept=f['coefficient'][0]-coefficient@f['mean'];history=list(y)
 for t in range(len(y),len(y)+h):history.append(float(intercept+coefficient@np.r_[history[t-k:t][::-1],u[t-k+1:t+1][::-1]]))
 # AR state-transition covariance,independent of the producer's impulse recursion.
 a=np.zeros((k,k));a[0]=coefficient[:k]
 if k>1:a[1:,:-1]=np.eye(k-1)
 cov=np.zeros_like(a);variances=[]
 for _ in range(h):cov=a@cov@a.T;cov[0,0]+=f['innovation_variance'];variances.append(cov[0,0])
 return np.array(history[-h:]),np.array(variances)
def main():
 started=time.monotonic();r=json.loads((OUT/'results.json').read_text());assert r['status']=='completed';assert sha(ROOT/'CED_SCREEN_PLAN.md')==r['plan_sha256']
 for manifest_path,source in [(ROOT/'outputs/source/pcfm_manifest.json',ROOT/'upstream/PCFM'),(OLD/'outputs/source/metaid/manifest.json',OLD/'upstream/sysid_benchmarks')]:
  manifest=json.loads(manifest_path.read_text())
  for entry in manifest['files']:assert sha(source/entry['path'])==entry['sha256']
 assert sha(ROOT/'outputs/external/PCFM/model.safetensors')==r['pcfm_weight_sha256']
 assert sha(OLD/'outputs/external/metaid/ckpt_430_fine-tuned_3100000_lower_noise.pt')==r['icl_weight_sha256']
 path=ROOT/'outputs/data/CED/DATAUNIF.MAT';assert sha(path)==r['data_sha256'];mat=loadmat(path)
 for row in r['predictions']:assert sha(OUT/row['path'])==row['sha256']
 with (OUT/'rows.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
 with (OUT/'arx_validation.csv').open(newline='',encoding='utf-8') as f:validation=list(csv.DictReader(f))
 assert len(rows)==22 and len(validation)==48
 maxima=dict(arx_parameter=0.,arx_prediction=0.,arx_variance=0.,score=0.,pcfm_replay=0.,icl_replay=0.);counts=dict(prediction_hashes=len(r['predictions']),validation_refits=0,scores=0,pcfm_replays=0,icl_replays=0,input_values=0)
 datasets={}
 for label,u_key,y_key in [('low_amplitude','u11','z11'),('high_amplitude','u12','z12')]:
  u=mat[u_key][:,0];y=mat[y_key][:,0];saved=dict(np.load(OUT/f'input_{label}.npz'));datasets[label]=(u,y)
  assert np.array_equal(saved['u'],u) and np.array_equal(saved['y'],y);assert int(saved['train_stop'])==400 and int(saved['forecast_start'])==410
  assert np.array_equal(saved['model_window'][0],u) and np.array_equal(saved['model_window'][1,:410],y[:410]) and np.count_nonzero(saved['model_window'][1,410:])==0;counts['input_values']+=saved['model_window'].size
  choices=[]
  for item in [x for x in validation if x['record']==label]:
   k=int(item['order']);lam=float(item['lam']);f=fit(y[:300],u[:300],k,lam);mu,_=predict(f,y[:300],u,100);loss=float(np.mean((mu-y[300:400])**2));assert np.isclose(loss,float(item['free_run_validation_mse']),rtol=1e-7,atol=1e-7)
   choices.append((loss,k,lam));counts['validation_refits']+=1
  selected=min(choices);saved_fit=dict(np.load(OUT/f'arx_{label}.npz'));assert int(saved_fit['order'])==selected[1] and float(saved_fit['lam'])==selected[2]
  f=fit(y[:400],u[:400],selected[1],selected[2])
  for key in ['mean','scale','coefficient','innovation_variance']:
   maxima['arx_parameter']=max(maxima['arx_parameter'],float(np.max(np.abs(f[key]-saved_fit[key]))));assert np.allclose(f[key],saved_fit[key],atol=1e-7,rtol=1e-6)
  mu,var=predict(f,y[:410],u,90);d=dict(np.load(OUT/f'{label}_arx_train_selected.npz'));maxima['arx_prediction']=max(maxima['arx_prediction'],float(np.max(np.abs(mu-d['mean']))));assert np.allclose(mu,d['mean'],atol=1e-7,rtol=1e-6)
  expected_q=mu[:,None]+np.sqrt(var)[:,None]*norm.ppf(Q);maxima['arx_variance']=max(maxima['arx_variance'],float(np.max(np.abs(expected_q-d['quantiles']))));assert np.allclose(expected_q,d['quantiles'],atol=1e-7,rtol=1e-6)
 for row in rows:
  u,y=datasets[row['record']];d=dict(np.load(OUT/f"{row['record']}_{row['method']}.npz"));assert np.array_equal(d['target'],y[410:]);pred=d['mean'];error=pred-y[410:]
  values=dict(rmse=float(np.linalg.norm(error)/np.sqrt(90)),mae=float(np.sum(np.abs(error))/90),normalized_rmse=float(np.linalg.norm(error)/np.sqrt(90)/np.std(y[:400])),finite_fraction=float(np.isfinite(pred).sum()/90))
  if row['method']=='persistence':assert np.array_equal(pred,np.full(90,y[409]))
  if row['method']=='training_mean':assert np.allclose(pred,np.full(90,np.mean(y[:400])),atol=1e-12,rtol=1e-12)
  if 'quantiles' in d:
   q=d['quantiles'];assert np.diff(q,axis=-1).min()>-1e-5;e=y[410:,None]-q
   values.update(pinball9=float(np.sum(np.where(e>=0,e*Q,-e*(1-Q)))/810),coverage10_90=float(np.count_nonzero((y[410:]>=q[:,0])&(y[410:]<=q[:,-1]))/90),mean_width10_90=float(np.sum(q[:,-1]-q[:,0])/90))
  for key,value in values.items():maxima['score']=max(maxima['score'],abs(value-float(row[key])));assert np.isclose(value,float(row[key]),atol=1e-7,rtol=1e-7);counts['scores']+=1
 # Same unmodified neural source and weights in a new process: numerical replay,
 # explicitly separate from the independent data/ARX/score calculations above.
 sys.path.insert(0,str(ROOT/'upstream/PCFM'));from model import PCFMPredictor
 pcfm=PCFMPredictor.from_pretrained(str(ROOT/'outputs/external/PCFM/model.safetensors'),device='cuda')
 for row in rows:
  if not row['method'].startswith('pcfm_'):continue
  label=row['record'];window=np.load(OUT/f'input_{label}.npz')['model_window'];method=row['method'];graph='CP' if 'pcfm_CP' in method else ('CD' if 'pcfm_CD' in method else [(0,1,0.)]);glue=method.endswith('glue1')
  q=pcfm.what_if(window,context_length=410,masked_nodes=[1],graph=graph,glue=glue,return_quantiles=True)[1,410:].cpu().numpy();saved=np.load(OUT/f'{label}_{method}.npz')['quantiles'];diff=float(np.max(np.abs(q-saved)));maxima['pcfm_replay']=max(maxima['pcfm_replay'],diff);assert np.allclose(q,saved,atol=2e-5,rtol=2e-5);counts['pcfm_replays']+=1
 sys.path.insert(0,str(OLD/'upstream/sysid_benchmarks'));from transformer_sim import Config,TSTransformer
 with torch.serialization.safe_globals([argparse.Namespace]):payload=torch.load(OLD/'outputs/external/metaid/ckpt_430_fine-tuned_3100000_lower_noise.pt',map_location='cpu',weights_only=True)
 model=TSTransformer(Config(**payload['model_args'])).cuda().eval();model.load_state_dict(payload['model'],strict=True)
 for scale in r['scaling']:
  label=scale['record'];u,y=datasets[label]
  for name,values in [('u',u),('y',y)]:
   assert np.isclose(scale[name+'_mean'],values[:400].mean(),atol=5e-6,rtol=1e-6);assert np.isclose(scale[name+'_std'],values[:400].std(ddof=1)+1e-6,atol=5e-6,rtol=1e-6)
  for mode,pad,n_in in [('native_padded30',20,30),('direct10',0,10)]:
   d=dict(np.load(OUT/f'icl_inputs_{label}_{mode}.npz'));uz=(u-scale['u_mean'])/scale['u_std'];yz=(y[:410]-scale['y_mean'])/scale['y_std']
   for key,expected in [('ctx_u',uz[:400]),('ctx_y',yz[:400]),('qu',np.r_[np.zeros(pad),uz[400:]]),('qy',np.r_[np.zeros(pad),yz[400:],np.zeros(90)])]:assert np.array_equal(d[key][0,:,0],expected.astype(np.float32))
   args=[torch.tensor(d[key],device='cuda') for key in ['ctx_y','ctx_u','qu','qy']]
   with torch.inference_mode():mu,std=model(*args,n_in)[:2]
   for actual,key in [(mu,'mean'),(std,'std')]:
    diff=float(np.max(np.abs(actual.cpu().numpy()-d[key])));maxima['icl_replay']=max(maxima['icl_replay'],diff);assert np.allclose(actual.cpu().numpy(),d[key],atol=2e-5,rtol=2e-5)
   mu=d['mean'][0,:,0]*scale['y_std']+scale['y_mean'];std=d['std'][0,:,0]*scale['y_std'];saved=np.load(OUT/f'{label}_icl_{mode}.npz');assert np.allclose(saved['mean'],mu,atol=1e-7,rtol=1e-7);assert np.allclose(saved['quantiles'],mu[:,None]+std[:,None]*norm.ppf(Q),atol=1e-7,rtol=1e-7);counts['icl_replays']+=1
 receipt=dict(status='passed',results_sha256=sha(OUT/'results.json'),verifier_sha256=sha(pathlib.Path(__file__)),counts=counts,maxima=maxima,runtime_s=time.monotonic()-started,
  scope='Independent official raw split,input scaling,all48 ARX validations,augmented-LS finalfit,AR covariance propagation and allscores;separate same-source full neural replay')
 (OUT/'verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
