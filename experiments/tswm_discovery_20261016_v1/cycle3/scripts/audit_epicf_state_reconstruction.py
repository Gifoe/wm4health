"""Can target-scenario susceptible trajectories reconstruct target outcomes?

Source hypothesis: transition.py records k * newly_exposed_today as daily cases
and S declines by newly_exposed_today / population. Fit only one nonnegative
scale to each factual trajectory, then apply it unchanged to counterfactual S.
This deliberately uses the released oracle-state information, not history-only
inputs. First day is excluded because S[-1] is not in the public table.
"""
import datetime,json,pathlib
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
def scores(y,p):
 error=p-y;den=np.maximum(np.ptp(y,axis=1),1.)
 return {'rmse_cases':float(np.sqrt(np.mean(error**2))),'mae_cases':float(np.mean(np.abs(error))),'max_abs_cases':float(np.max(np.abs(error))),'nrmse_county_range':float(np.mean(np.sqrt(np.mean(error**2,axis=1))/den))}
def main():
 out=ROOT/'outputs';plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formula':'prediction_c,t = fitted_scale_c * (S_c,t-1 - S_c,t)','fit':'factual trajectory t=1..167 only, scalar least squares through origin, nonnegative','targets':'counterfactual trajectories t=1..167 only','input_information':'target-scenario simultaneous S trajectory (explicit oracle); no action input','controls':['unchanged factual S','CF S shifted 1 day','CF S of a shuffled different trajectory'],'claim_boundary':'A near-exact reconstruction would show released target-state information encodes the answer; it would not establish causal prediction ability or a new method.'}
 path=out/'state_reconstruction_plan.json'
 if not path.exists():path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
 rows=[];details=[]
 for setting in ['single','multi']:
  df=pd.read_csv(ROOT/'data'/'raw'/f'{setting}_policy_data.csv')
  suffixes=['sch=0-occ=F','sch=0-occ=CF'] if setting=='single' else ['F-F','CF-F','F-CF','CF-CF']
  ynames=['0-F','0-CF'] if setting=='single' else suffixes
  ss=[df[f'susceptible_{s}'].to_numpy().reshape(158,168) for s in suffixes]
  ys=[df[s].to_numpy().reshape(158,168) for s in ynames]
  changes=[-np.diff(s,axis=1) for s in ss]
  d=changes[0];y=ys[0][:,1:];scale=np.maximum(np.sum(d*y,axis=1)/np.maximum(np.sum(d*d,axis=1),1e-30),0)
  dfact=scale[:,None]*d
  rows.append({'setting':setting,'scenario':'factual','control':'correct_state_difference',**scores(y,dfact)})
  for i in range(1,len(ss)):
   truth=ys[i][:,1:];prediction=scale[:,None]*changes[i]
   rng=np.random.default_rng(20261016);permutation=rng.permutation(158)
   for control,p in [('correct_state_difference',prediction),('factual_state_difference',dfact),('one_day_shifted_cf_difference',scale[:,None]*np.roll(changes[i],1,axis=1)),('shuffled_county_cf_difference',scale[:,None]*changes[i][permutation])]:
    rows.append({'setting':setting,'scenario':suffixes[i],'control':control,**scores(truth,p)})
   for c in range(158):details.append({'setting':setting,'scenario':suffixes[i],'trajectory':c,'county':str(df.iloc[c*168]['county']),'factual_fitted_scale':float(scale[c]),'cf_rmse_cases':float(np.sqrt(np.mean((prediction[c]-truth[c])**2))),'cf_max_abs_cases':float(np.max(np.abs(prediction[c]-truth[c])))})
  print(json.dumps([r for r in rows if r['setting']==setting]),flush=True)
 (out/'state_reconstruction_results.json').write_text(json.dumps({'plan':plan,'results':rows,'first_day_excluded':True,'n_evaluated_days_per_trajectory':167,'scale_fitting_uses_cf_targets':False},indent=2),encoding='utf-8')
 pd.DataFrame(details).to_csv(out/'state_reconstruction_by_county.csv',index=False)
if __name__=='__main__':main()
