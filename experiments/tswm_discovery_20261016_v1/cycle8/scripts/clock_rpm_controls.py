"""Ordinary strong controls on real data; no native-paper reproduction claim.

Known future commands are allowed by the prerecorded programmed protocol. Actual
future sample intervals are an explicitly privileged conditional comparator;
deployable arms predict them using history/known commands only.
"""
import runtime_profile
import datetime,hashlib,json,pathlib,time
import numpy as np,pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/clock_rpm_controls'
SETTINGS=['hatch','pot_1','pot_2','load_in','load_out','res_in','res_out','osr_in','osr_out','osr_1','osr_2','osr_mic','osr_upwind','osr_downwind','osr_ambient','osr_intake','v_in','v_out','v_1','v_2','v_mic']
def dump(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def q(x):return dict(zip(['p01','median','p99','max'],np.quantile(x,[.01,.5,.99,1]).tolist()))
def cfeatures(settings,previous,past_dt):
 # No current/future measured sensor or timestamp enters these features.
 d=np.abs(settings[:,:5]-previous[:,:5])
 return np.column_stack([settings,d,past_dt,np.median(past_dt,axis=1)])
def features(prev,prev2,settings,previous,dt=None):
 u=settings[:,3:5];up=previous[:,3:5];r=prev/3000.;dr=(prev-prev2)/3000.
 base=np.column_stack([r,dr,u,up,u**2,u**3,u*up,u*r,settings[:,:3]/np.array([45,255,255]),settings[:,5:]])
 if dt is not None:base=np.column_stack([base,dt,dt[:,None]*np.column_stack([r,dr,u,u**2,u**3])])
 return base
def pack(path):
 df=pd.read_csv(path);n=len(df);settings=df[SETTINGS].to_numpy(float);rpm=df[['rpm_in','rpm_out']].to_numpy(float);ts=df.timestamp.to_numpy(float);dt=np.diff(ts,prepend=ts[0]);idx=np.arange(16,n)
 assert np.isfinite(settings).all() and np.isfinite(rpm).all() and (dt[1:]>0).all()
 past=np.column_stack([dt[idx-j] for j in range(1,9)])
 return {'path':path,'frame':df,'settings':settings,'rpm':rpm,'dt':dt,'idx':idx,'C':cfeatures(settings[idx],settings[idx-1],past),'past_dt':past,'B':features(rpm[idx-1],rpm[idx-2],settings[idx],settings[idx-1]),'T':features(rpm[idx-1],rpm[idx-2],settings[idx],settings[idx-1],dt[idx]),'Y':(rpm[idx]-rpm[idx-1])/3000.}
def fit_ridge(x,y,valx,valy):
 scale=StandardScaler().fit(x);xx=scale.transform(x);vx=scale.transform(valx);candidates=[]
 for alpha in [1e-6,1e-4,.01,1.,100.,10000.]:
  model=Ridge(alpha=alpha).fit(xx,y);mse=float(np.mean((model.predict(vx)-valy)**2));candidates.append((mse,alpha,model))
 mse,alpha,model=min(candidates,key=lambda t:t[0]);return scale,model,{'alpha':alpha,'validation_increment_MSE':mse,'grid':[{'alpha':a,'MSE':v} for v,a,_ in candidates]}
def main():
 started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'plan.json').exists():raise RuntimeError('Existing plan; inspect terminal status rather than restart')
 plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'New real-data qualification/control experiment. Ordinary ridge ARX/polynomial and HGB; no new method, no native WorldTS/TAWM reproduction.','train_random_walks':list(range(2,11)),'validation_random_walks':[11,12],'test_random_walks':[13,14,15,16],'excluded_random_walk_1':'Only1016 observations versus10000 in the other recordings; excluded by inventory before model metrics.','OOD':'All10loads_hatch_mix recordings and both320kregime-jump recordings. No OOD score selects model or hyperparameters.','history':'RPM2lags; clock8pastintervals; current/previous plannedsettings. No future measured sensor enters deployable inputs.','clock_model':'HGB150iterations,learning_rate.08,max_leaf_nodes15,l2=1,seed1801,training-internal10%validation for earlystop. Clip predicted interval only to train observed min/max.','prediction_model':'Ridge ARX/polynomial, train-only standardized,6alphas selected by mean validation increment MSE. Train targets are RPM increments divided by3000.','arms':['persistence','ridge_no_clock','ridge_oracle_future_dt','ridge_predicted_dt','ridge_past_dt'],'oracle_caveat':'Actual future timestamp spacing is a privileged conditional comparator, not available in next-observation simulation. Predicted/past arms share its fitted dynamics weights.','one_step':'Every row after16; history is observed.','rollout':'16steps,roots every64rows from32; past states/clocks then entirely predicted. Known future actuator/sensor settings are programmed and allowed. No cross-recording root/window.','replication':'Deterministic fit on split recordings, no seed stability claim or confidence interval. Recordings and intervention blocks remain grouped.'}
 dump(OUT/'plan.json',plan);folder=ROOT/'data/wt_walks_v1';lookup={p.stem:p for p in folder.rglob('*.csv')}
 sets={}
 for split,ids in [('train',plan['train_random_walks']),('validation',plan['validation_random_walks'])]:sets[split]=[pack(lookup[f'actuators_random_walk_{i}']) for i in ids]
 cat=lambda split,key:np.concatenate([r[key] for r in sets[split]],axis=0)
 ctrain=cat('train','C');ytime=np.concatenate([r['dt'][r['idx']] for r in sets['train']]);clock=HistGradientBoostingRegressor(max_iter=150,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.,random_state=1801,early_stopping=True).fit(ctrain,ytime)
 lo,hi=float(ytime.min()),float(ytime.max())
 bscale,bmodel,binfo=fit_ridge(cat('train','B'),cat('train','Y'),cat('validation','B'),cat('validation','Y'))
 tscale,tmodel,tinfo=fit_ridge(cat('train','T'),cat('train','Y'),cat('validation','T'),cat('validation','Y'))
 dump(OUT/'fit.json',{'no_clock':binfo,'conditional_clock':tinfo,'clock_train_range':[lo,hi],'sklearn':sklearn.__version__,'train_rows':len(ytime),'seconds_to_fit':time.perf_counter()-started})
 joblib.dump({'clock':clock,'bscale':bscale,'bmodel':bmodel,'tscale':tscale,'tmodel':tmodel},OUT/'fitted_models.joblib')
 # Fixed dynamics weights give a direct interval-information comparison.
 def predict(prev,prev2,s,sp,dt,arm):
  if arm=='persistence':return prev.copy()
  if arm=='ridge_no_clock':return prev+3000*bmodel.predict(bscale.transform(features(prev,prev2,s,sp)))
  return prev+3000*tmodel.predict(tscale.transform(features(prev,prev2,s,sp,dt)))
 rows=[];evalnames=[f'actuators_random_walk_{i}' for i in [11,12,13,14,15,16]]+sorted(n for n in lookup if n.startswith('loads_hatch_mix_') or n.startswith('regime_jumps_'))
 for name in evalnames:
  r=pack(lookup[name]);idx=r['idx'];rp=r['rpm'];s=r['settings'];past=r['past_dt'];pred_dt=np.clip(clock.predict(r['C']),lo,hi);actual_dt=r['dt'][idx];ref_dt=past[:,0]
  split='validation' if name in ['actuators_random_walk_11','actuators_random_walk_12'] else ('test' if name.startswith('actuators_') else 'OOD')
  base={'recording':name,'split':split,'rows':len(idx),'clock_seconds':{'actual':q(actual_dt),'predicted_RMSE':float(np.sqrt(np.mean((pred_dt-actual_dt)**2))),'last_interval_RMSE':float(np.sqrt(np.mean((ref_dt-actual_dt)**2))),'mean_clock_RMSE':float(np.sqrt(np.mean((np.mean(ytime)-actual_dt)**2)))}}
  arms={};roots=np.arange(32,len(rp)-16,64);actual_path=np.stack([rp[roots+h] for h in range(1,17)],1)
  for arm in plan['arms']:
   delta=actual_dt if arm=='ridge_oracle_future_dt' else (pred_dt if arm=='ridge_predicted_dt' else ref_dt)
   pred=predict(rp[idx-1],rp[idx-2],s[idx],s[idx-1],delta,arm);one=(pred-rp[idx])**2
   prev=rp[roots].copy();prev2=rp[roots-1].copy();clock_hist=np.column_stack([r['dt'][roots-j] for j in range(8)]);paths=[];durations=[]
   for h in range(1,17):
    ii=roots+h
    if arm=='ridge_oracle_future_dt':d=r['dt'][ii]
    elif arm=='ridge_predicted_dt':d=np.clip(clock.predict(cfeatures(s[ii],s[ii-1],clock_hist)),lo,hi)
    else:d=clock_hist[:,0].copy()
    new=predict(prev,prev2,s[ii],s[ii-1],d,arm);paths.append(new);durations.append(d);prev2,prev=prev,new;clock_hist=np.column_stack([d,clock_hist[:,:-1]])
   arr=np.stack(paths,1);dur=np.stack(durations,1);real_dur=np.stack([r['dt'][roots+h] for h in range(1,17)],1)
   arms[arm]={'one_step_RPM_RMSE':np.sqrt(one.mean(0)).tolist(),'one_step_normalized_MSE':float(one.mean()/3000**2),'rollout_roots':len(roots),'rollout16_normalized_MSE':float(np.mean((arr-actual_path)**2)/3000**2),'endpoint16_RPM_RMSE':np.sqrt(np.mean((arr[:,-1]-actual_path[:,-1])**2,axis=0)).tolist(),'sum16_interval_RMSE_seconds':float(np.sqrt(np.mean((dur.sum(1)-real_dur.sum(1))**2)))}
  base['arms']=arms
  if name.startswith('regime_jumps_'):
   # Error is not attributed to latent dynamics until measurement/clock controls
   # and source timing are qualified. These remain measured RPM units.
   base['scope']='Sensor and physical interventions mixed; no physical ground-truth latent or matched cross-world trajectory is claimed.'
  rows.append(base);dump(OUT/'partial_results.json',rows);print(json.dumps({'recording':name,'clock_RMSE':base['clock_seconds']['predicted_RMSE'],'one_step_MSE':{k:v['one_step_normalized_MSE'] for k,v in arms.items()},'rollout_MSE':{k:v['rollout16_normalized_MSE'] for k,v in arms.items()}}),flush=True)
 dump(OUT/'results.json',{'plan':plan,'rows':rows,'seconds':time.perf_counter()-started,'objective_achieved':False});dump(OUT/'status.json',{'state':'complete','recordings':len(rows),'objective_achieved':False});print('Completed real-data controls',flush=True)
if __name__=='__main__':main()
