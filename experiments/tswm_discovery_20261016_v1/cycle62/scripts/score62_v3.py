from native62 import *
from types import SimpleNamespace
import time,math
campaign=read(OUT/'campaign_state62_v3.json');assert campaign['status']=='ALL_NINE_NATIVE_FITS_SEALED' and len(campaign['completed'])==9
verify_sources();ad,models,compat=load_native();df,dp=load_data();pp=paths();tv=int(len(df)*.8);tr=int(tv*.75)
mu=df.iloc[:tr].values.mean(0);scale=df.iloc[:tr].values.std(0);z=(df.values-mu)/scale
official=np.arange(tv,len(df)-96+1);common=np.arange(tv,len(df)-384+1,96)
allrows=[];artifacts=[];source_check=[];t0=time.perf_counter()
def metrics(pred,origins,h):
 target=np.stack([df.values[o:o+h] for o in origins]);e=(pred[:,:h]-target)/scale
 assert np.isfinite(e).all()
 return {'horizon':h,'origins':len(origins),'mse':float(np.mean(e*e)),'mae':float(np.mean(np.abs(e))),'channel_mse':np.mean(e*e,axis=(0,1)).tolist(),'channel_mae':np.mean(np.abs(e),axis=(0,1)).tolist(),'finite':True}
class HistoryBatch:
 def __init__(self,origins):
  self.x=np.stack([df.iloc[o-96:o].values for o in origins]);self.ts=np.stack([df.index.values[o-96:o] for o in origins]);self.pos=0
 def make_batch(self,batch_size,win_size):
  assert win_size==96;sl=slice(self.pos,self.pos+batch_size);self.pos+=batch_size
  return {'input':self.x[sl].copy(),'time_stamps':self.ts[sl].copy()}
def predict(a,origins,h,s):
 seed(s);batch=HistoryBatch(origins);pred=[]
 while batch.pos<len(origins):pred.append(a.batch_forecast(h,batch))
 result=np.concatenate(pred);assert result.shape==(len(origins),h,7) and np.isfinite(result).all();return result
def independent_roll(a,origins,h,s):
 # Build marks from the actual known calendar and loop without native rolling helpers.
 import pandas as pd
 util=importlib.import_module('ts_benchmark.baselines.utils')
 hist=np.stack([df.iloc[o-96:o].values for o in origins]);cur=(hist-mu)/scale
 dates=np.stack([pd.date_range(df.index[o-96],periods=96+h,freq='h').values for o in origins]);marks=util.get_time_mark(dates,1,'h')
 seed(s);parts=[]
 with torch.no_grad():
  for start in range(0,h,96):
   xx=torch.tensor(cur,dtype=torch.float32,device='cuda');dec=torch.cat([xx[:,-48:],torch.zeros((len(cur),96,7),device='cuda')],dim=1)
   xm=torch.tensor(marks[:,start:start+96],dtype=torch.float32,device='cuda');ym=torch.tensor(marks[:,start+48:start+192],dtype=torch.float32,device='cuda')
   out=a.model(xx,xm,dec,ym).cpu().numpy()[:,-96:];parts.append(out);cur=np.concatenate([cur,out],axis=1)[:,-96:]
 return np.concatenate(parts,axis=1)*scale+mu
for key in campaign['completed']:
 state('scoring_state62_v1.json',{'status':'SCORING','active':key,'completed_rows':len(allrows)})
 r=read(OUT/(key+'_fit62_v1.json'));assert sha(r['selected_checkpoint'])==r['selected_sha256'] and sha(r['terminal_checkpoint'])==r['terminal_sha256']
 a=adapter(ad,models,r['arm']);a.config=ad.TransformerConfig(**r['config']);a.model=models[r['arm']](a.config).cuda().eval();a.model.load_state_dict(torch.load(r['selected_checkpoint'],map_location='cuda',weights_only=True));a.early_stopping=SimpleNamespace(check_point=None);a.scaler.fit(df.iloc[:tr].values)
 # Training-only equality probe avoids opening new evaluation targets for qualification.
 probes=[tr-600,tr-500];probe_native=predict(a,probes,384,62901);probe_ind=independent_roll(a,probes,384,62901)
 err=float(np.max(np.abs(probe_native-probe_ind)));assert np.allclose(probe_native,probe_ind,atol=2e-5,rtol=2e-5),(key,err)
 source_check.append({'key':key,'native_vs_independent_roll_max_raw_difference':err,'probe_origins_training_only':probes})
 saved={}
 for label,origins,h in [('official96',official,96),('common384',common,384)]:
  preds=[]
  for rep,rs in enumerate([620100,620101,620102]):
   pred=predict(a,origins,h,rs);preds.append(pred)
   for hh in ([96] if h==96 else [96,192,384]):allrows.append({'key':key,'arm':r['arm'],'training_seed':r['seed'],'inference_rng':rs,'view':label,**metrics(pred,origins,hh)})
  saved[label]=np.stack(preds).astype(np.float32)
 path=pp['artifact_root']/(key+'_predictions62_v1.npz');np.savez_compressed(path,official_origins=official,common_origins=common,**saved)
 artifacts.append({'key':key,'path':str(path),'sha256':sha(path),'size':path.stat().st_size})
 del a;torch.cuda.empty_cache()
# All elementary controls use the same training-only normalization and permitted history.
def control_predictions(kind,origins,h):
 ans=[]
 for o in origins:
  if kind=='persistence':p=np.repeat(z[o-1:o],h,axis=0)
  elif kind=='seasonal24':p=np.tile(z[o-24:o],(math.ceil(h/24),1))[:h]
  else:
   cur=z[o-96:o].copy();parts=[]
   for start in range(0,h,96):
    x=cur.reshape(-1);p=((x-ridge['xmean'])@ridge['coef']+ridge['ymean']).reshape(96,7);parts.append(p);cur=p
   p=np.concatenate(parts)[:h]
  ans.append(p*scale+mu)
 return np.stack(ans)
train_orig=np.arange(96,tr-96+1);val_orig=np.arange(tr,tv-96+1)
X=np.stack([z[o-96:o].reshape(-1) for o in train_orig]);Y=np.stack([z[o:o+96].reshape(-1) for o in train_orig]);xm=X.mean(0);ym=Y.mean(0);X-=xm;Y-=ym
gram=X.T@X;cross=X.T@Y;ev,V=np.linalg.eigh(gram);vx=V.T@cross
VX=np.stack([z[o-96:o].reshape(-1) for o in val_orig]);VY=np.stack([z[o:o+96].reshape(-1) for o in val_orig]);candidates=[];best=None
for alpha in [1e-4,1e-2,1.,100.]:
 coef=V@(vx/(ev[:,None]+alpha));err=(VX-xm)@coef+ym-VY;mse=float(np.mean(err*err));candidates.append({'alpha':alpha,'validation96_mse':mse})
 if best is None or mse<best[0]:best=(mse,alpha,coef)
ridge={'xmean':xm,'ymean':ym,'coef':best[2]};rp=pp['checkpoint_root']/'ridge62_v1.npz';np.savez_compressed(rp,**ridge,alpha=best[1]);dump(OUT/'RIDGE_SELECTION62_V1.json',{'candidates':candidates,'selected_alpha':best[1],'train_rows':tr,'validation_origins':len(val_orig),'checkpoint':str(rp),'sha256':sha(rp),'selection':'validation96_only'})
for kind in ['persistence','seasonal24','ridge']:
 saved={}
 for label,origins,h in [('official96',official,96),('common384',common,384)]:
  pred=control_predictions(kind,origins,h);saved[label]=pred
  for hh in ([96] if h==96 else [96,192,384]):allrows.append({'key':kind,'arm':kind,'training_seed':None,'inference_rng':None,'view':label,**metrics(pred,origins,hh)})
 path=pp['artifact_root']/(kind+'_predictions62_v1.npz');np.savez_compressed(path,official_origins=official,common_origins=common,**saved);artifacts.append({'key':kind,'path':str(path),'sha256':sha(path),'size':path.stat().st_size})
dump(OUT/'EVALUATION62_V1.json',{'utc':utc(),'rows':allrows,'artifacts':artifacts,'native_vs_independent_roll':source_check,'data_sha256':sha(dp),'protocol_sha256':sha(ROOT/'PROTOCOL62_V1.md'),'source_revision':'733d986d349e9e71ded5d9cdeeeed8d3323d83b4','official_origins':official.tolist(),'common_origins':common.tolist(),'inference_rng_streams':[620100,620101,620102],'scoring_seconds':time.perf_counter()-t0})
state('scoring_state62_v1.json',{'status':'ALL_PREDICTIONS_SCORED','rows':len(allrows),'artifacts':len(artifacts)})
print(json.dumps({'status':'ALL_PREDICTIONS_SCORED','rows':len(allrows),'artifacts':len(artifacts)}))
