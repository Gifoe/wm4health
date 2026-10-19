"""Execute official preprocessing and record data/implementation information sets.

This is a source/data audit, not a new model or a causal discovery claim.
"""
import datetime,hashlib,json,pathlib,sys
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources'/'epi-cf-benchmark'/'src'))
from epicf_benchmark.preprocessing import preprocess_single,preprocess_multi
def main():
 out=ROOT/'outputs';out.mkdir(exist_ok=True,parents=True);rows=[]
 for setting,function in [('single',preprocess_single),('multi',preprocess_multi)]:
  raw=ROOT/'data'/'raw'/f'{setting}_policy_data.csv';df=pd.read_csv(raw,dtype={'county':str})
  processed=ROOT/'data'/f'{setting}_official'
  function(raw,processed)
  counts=df.groupby('county',sort=False).size()
  ids=df['county'].drop_duplicates().tolist()
  # The upstream loader assumes county-major and chronological order.
  contiguous=all((df['county']==c).sum()==len(df.loc[df['county']==c]) for c in ids) and df['county'].tolist()==[c for c in ids for _ in range(int(counts[c]))]
  chronological=all(g['day'].is_monotonic_increasing and g['day'].is_unique for _,g in df.groupby('county',sort=False))
  structural={'setting':setting,'num_counties':len(ids),'count_distribution':counts.value_counts().to_dict(),'contiguous_county_major':contiguous,'chronological':chronological,'head':df[['county','date','day']].head(8).to_dict(orient='records')}
  print(json.dumps(structural),flush=True)
  (out/f'{setting}_structural_audit.json').write_text(json.dumps(structural,indent=2),encoding='utf-8')
  assert len(df)==158*168
  blocks=[df.iloc[i*168:(i+1)*168].reset_index(drop=True) for i in range(158)]
  assert all(b['county'].nunique()==1 and b['day'].tolist()==list(range(168)) for b in blocks),'Malformed fixed-length trajectory block'
  block_ids=[b['county'].iloc[0] for b in blocks]
  duplicates=[]
  for county in counts[counts>168].index:
   positions=[i for i,c in enumerate(block_ids) if c==county]
   left,right=[blocks[i] for i in positions]
   different=[c for c in df.columns if not left[c].equals(right[c])]
   duplicates.append({'county':county,'block_indices':positions,'identical_all_columns':not different,'different_columns':different})
  structural.update(valid_158_trajectory_blocks=True,duplicate_counties=duplicates)
  (out/f'{setting}_structural_audit.json').write_text(json.dumps(structural,indent=2),encoding='utf-8')
  pd.DataFrame({'trajectory_id':list(range(158)),'county':block_ids}).to_csv(out/f'{setting}_trajectory_identity.csv',index=False)
  xf=pd.read_csv(processed/'x_train.csv').to_numpy().reshape(158,168,17)
  xcf=pd.read_csv(processed/'x_test.csv').to_numpy().reshape(-1,168,17)
  yf=pd.read_csv(processed/'y_train.csv',header=None).to_numpy().reshape(158,168)
  ycf=pd.read_csv(processed/'y_test.csv',header=None).to_numpy().reshape(-1,168)
  af=pd.read_csv(processed/'a_train.csv',header=None).to_numpy().reshape(158,168,-1)
  acf=pd.read_csv(processed/'a_test.csv',header=None).to_numpy().reshape(-1,168,af.shape[-1])
  assert all(np.isfinite(x).all() for x in [xf,xcf,yf,ycf,af,acf])
  repeats=len(xcf)//len(xf)
  state_diffs=[np.mean(np.abs(xcf[i*158:(i+1)*158,:,13:]-xf[:,:,13:]),axis=(0,1)).tolist() for i in range(repeats)]
  row={'setting':setting,'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'shape_raw':list(df.shape),'columns':df.columns.tolist(),'num_counties':len(ids),'num_trajectory_blocks':158,'duplicate_counties':duplicates,'n_days':168,'contiguous_county_major':contiguous,'chronological_by_county':chronological,'valid_fixed_length_blocks':True,'date_min':str(df['date'].min()),'date_max':str(df['date'].max()),'factual_input_shape':list(xf.shape),'counterfactual_input_shape':list(xcf.shape),'covariate_names':pd.read_csv(processed/'x_train.csv',nrows=0).columns.tolist(),'counterfactual_state_covariates_are_scenario_specific':True,'mean_abs_state_difference_vs_factual_by_scenario':state_diffs,'same_exogenous_covariates_across_scenarios':bool(np.array_equal(xcf[:,:,:13],np.tile(xf[:,:,:13],(repeats,1,1)))),'factual_policy_fraction_by_dimension':af.mean(axis=(0,1)).tolist(),'policy_switches_per_county_mean':np.abs(np.diff(af,axis=1)).sum(axis=1).mean(axis=0).tolist(),'y_stats_factual_zscore':{'mean':float(yf.mean()),'std':float(yf.std())},'y_stats_cf_zscore':{'mean':float(ycf.mean()),'std':float(ycf.std())},'outcome_stats':json.loads((processed/'outcome_stats.json').read_text())}
  df[['county','date','day']].head(336).to_csv(out/f'{setting}_row_identity_examples.csv',index=False)
  rows.append(row);print(json.dumps(row),flush=True)
 report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'data_version':'Zenodo 20762823 v1.2 (signal checksums identical to v1.1)','source_revision':'8f7df27507dc8b01422bb373417c8d0d3efffae2','settings':rows,'source_findings':{'transformer_validation_and_printed_test':'Both constructed from train files and all training indices; no independent split','transformer_attention':'Unmasked whole-sequence Transformer, not online forecasting','rnn_attention':'Bidirectional LSTM, not online forecasting','cf_dynamic_covariates':'Target-scenario S,E,R,M at every evaluation time; not restricted to pre-intervention history','kde_bandwidth_selection':'Training self-prediction, not leave-one-out validation','t_learner_routing':'Mean treatment over complete sequence thresholded at 0.5; not per-time or multi-policy regimes','multi_transformer_default_loader':'158-unit default incompatible with 474 counterfactual sequences unless adjusted; evaluate separately'},'interpretation':'Released task conditions on target-scenario endogenous state trajectories. Do not interpret scores as history-only action-conditioned simulation. This alone is a protocol finding, not a novel method gap.'}
 (out/'epicf_protocol_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
