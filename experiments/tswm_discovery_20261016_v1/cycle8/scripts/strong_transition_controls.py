"""Nonlinear existing baseline to exclude a polynomial/capacity explanation."""
import runtime_profile
import datetime,json,pathlib,time
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
import clock_rpm_controls as base
ROOT=base.ROOT;OUT=ROOT/'outputs/strong_transition_controls'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'plan.json').exists():raise RuntimeError('Existing plan; inspect exact process before restart')
 plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Previous ordinary polynomial controls show OOD residual; test a stronger existing nonlinear predictor before attributing it to a world-model research gap.','scope':'New controlled real-data baseline, not native-paper reproduction or proposed new architecture.','training':list(range(2,11)),'validation':[11,12],'test':[13,14,15,16],'OOD':'Same wave/regime recordings, no test/OOD hyperparameter selection','model':'Two separate HGB direct-next-RPM regressors per information arm, max_iter200,lr.05,max_leaf_nodes31,l2=1,random_state1802,internal10%training validation earlystop.','capacity_control':'More flexible than ridge; not parameter/compute matched, intentionally an existing strong stop baseline.','arms':['HGB_no_clock','HGB_oracle_future_dt','HGB_predicted_dt'],'clock':'Reuse frozen clock predictor from prior control fit; no extra sensor outcome inputs','horizon':'Same16step autoregressive roots every64 rows. Group diagnostics exclude roots crossing a regime boundary.','acceptance':'If ordinary nonlinear prediction explains prior failure, reject a new-method claim. Unknown measurement transformation/target support remains a competing explanation even if this baseline fails.'}
 base.dump(OUT/'plan.json',plan);started=time.perf_counter();lookup={p.stem:p for p in (ROOT/'data/wt_walks_v1').rglob('*.csv')};train=[base.pack(lookup[f'actuators_random_walk_{i}']) for i in plan['training']]
 fitted=joblib.load(ROOT/'outputs/clock_rpm_controls/fitted_models.joblib');clock=fitted['clock'];info=json.loads((ROOT/'outputs/clock_rpm_controls/fit.json').read_text());lo,hi=info['clock_train_range']
 models={}
 for key in ['B','T']:
  x=np.concatenate([r[key] for r in train]);y=np.concatenate([r['rpm'][r['idx']]/3000 for r in train]);models[key]=[]
  for j in range(2):
   m=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.,random_state=1802,early_stopping=True).fit(x,y[:,j]);models[key].append(m);print('Fitted',key,j,'iterations',m.n_iter_,flush=True)
 joblib.dump(models,OUT/'fitted_models.joblib')
 inventory=json.loads((ROOT/'outputs/data_audit/inventory.json').read_text());labels={(r['file'],r['regime']):r for r in inventory['regimes']}
 def pred(prev,prev2,s,sp,d,arm):
  key='B' if arm=='HGB_no_clock' else 'T';x=base.features(prev,prev2,s,sp,None if key=='B' else d)
  return 3000*np.column_stack([m.predict(x) for m in models[key]])
 names=[f'actuators_random_walk_{i}' for i in [11,12,13,14,15,16]]+sorted(n for n in lookup if n.startswith('loads_hatch_mix_') or n.startswith('regime_jumps_'));rows=[]
 for name in names:
  r=base.pack(lookup[name]);rp=r['rpm'];s=r['settings'];idx=r['idx'];pd=np.clip(clock.predict(r['C']),lo,hi);roots=np.arange(32,len(rp)-16,64);flag=r['frame'].flag.to_numpy();roots=roots[flag[roots]==flag[roots+16]];truth=np.stack([rp[roots+h] for h in range(1,17)],1);row={'recording':name,'rows':len(idx),'arms':{}}
  for arm in plan['arms']:
   d=r['dt'][idx] if arm=='HGB_oracle_future_dt' else pd;point=pred(rp[idx-1],rp[idx-2],s[idx],s[idx-1],d,arm);err=(point-rp[idx])**2/3000**2
   prev=rp[roots].copy();prev2=rp[roots-1].copy();hist=np.column_stack([r['dt'][roots-j] for j in range(8)]);paths=[]
   for h in range(1,17):
    ii=roots+h;dd=r['dt'][ii] if arm=='HGB_oracle_future_dt' else np.clip(clock.predict(base.cfeatures(s[ii],s[ii-1],hist)),lo,hi);p=pred(prev,prev2,s[ii],s[ii-1],dd,arm);paths.append(p);prev2,prev=prev,p;hist=np.column_stack([dd,hist[:,:-1]])
   roll=np.mean((np.stack(paths,1)-truth)**2,axis=(1,2))/3000**2;result={'one_step_normalized_MSE':float(err.mean()),'rollout16_normalized_MSE':float(roll.mean()),'rollout_roots':len(roots)}
   if name.startswith('regime_jumps_'):
    groups=[]
    for f in sorted(np.unique(flag)):
     ii=flag[idx]==f;rr=flag[roots]==f;meta=labels[(name,int(f))];groups.append({'regime':int(f),'targets':meta['declared_targets'],'protocol_type':meta['protocol_type'],'rows':int(ii.sum()),'one_step_normalized_MSE':float(err[ii].mean()),'rollout16_normalized_MSE':float(roll[rr].mean())})
    result['regimes']=groups
   row['arms'][arm]=result
  rows.append(row);base.dump(OUT/'partial_results.json',rows);print(json.dumps({'recording':name,'one_step':{k:v['one_step_normalized_MSE'] for k,v in row['arms'].items()},'rollout16':{k:v['rollout16_normalized_MSE'] for k,v in row['arms'].items()}}),flush=True)
 base.dump(OUT/'results.json',{'plan':plan,'rows':rows,'seconds':time.perf_counter()-started,'objective_achieved':False});base.dump(OUT/'status.json',{'state':'complete','recordings':len(rows),'objective_achieved':False});print('Strong control complete',flush=True)
if __name__=='__main__':main()
