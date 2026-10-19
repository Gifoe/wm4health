"""Audit clocks, recorded settings, finite values and protocol identities."""
import runtime_profile
import ast,datetime,hashlib,json,pathlib
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
SETTINGS=['hatch','pot_1','pot_2','load_in','load_out','res_in','res_out','osr_in','osr_out','osr_1','osr_2','osr_mic','osr_upwind','osr_downwind','osr_ambient','osr_intake','v_in','v_out','v_1','v_2','v_mic']
def quant(x):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
 return dict(zip(['min','p01','p10','median','p90','p99','max'],np.quantile(x,[0,.01,.1,.5,.9,.99,1]).tolist())) if len(x) else {}
def corr(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);good=np.isfinite(x)&np.isfinite(y)
 return float(np.corrcoef(x[good],y[good])[0,1]) if good.sum()>20 and x[good].std()>1e-8 and y[good].std()>1e-8 else None
def main():
 download=json.loads((ROOT/'outputs/data_download.json').read_text());rows=[];regimes=[]
 tree=ast.parse((ROOT/'sources/chamber/datasets/wt_walks_v1/regimes_targets.py').read_text());targets={}
 for n in tree.body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):targets[n.targets[0].id]=ast.literal_eval(n.value)
 for dataset in ['wt_walks_v1','wt_test_v1']:
  for path in sorted((ROOT/'data'/dataset).rglob('*.csv')):
   if path.name=='variables.csv':continue
   df=pd.read_csv(path);assert 'timestamp' in df
   ts=df.timestamp.to_numpy(dtype=float);dt=np.diff(ts);num=df.select_dtypes(include=[np.number]);finite=np.isfinite(num.to_numpy())
   row={'dataset':dataset,'file':path.relative_to(ROOT).as_posix(),'rows':len(df),'columns':df.columns.tolist(),'dtypes':{k:str(v) for k,v in df.dtypes.items()},'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nonfinite_numeric_values':int((~finite).sum()),'nonpositive_timestamp_differences':int((dt<=0).sum()),'duplicate_counter_values':int(df.counter.duplicated().sum()) if 'counter' in df else None,'timestamp_range_raw':[float(ts[0]),float(ts[-1])],'timestamp_note':'README variables.csv describes Unix time, but some released files use small shifted values. Treat as within-recording clock; no cross-file absolute alignment is assumed.','dt_seconds':quant(dt),'settings':{k:sorted(df[k].dropna().unique().tolist()) for k in SETTINGS if k in df and df[k].nunique()<=40}}
   rows.append(row)
   if path.stem in ['regime_jumps_single','regime_jumps_multi']:
    target_list=targets['targets_single' if path.stem.endswith('single') else 'targets_multi'];assert sorted(df.flag.unique())==list(range(32))
    baseline=df[df.flag==0].iloc[0]
    for flag,g in df.groupby('flag',sort=False):
     delta=np.diff(g.timestamp.to_numpy(float));changed={k:sorted(g[k].unique().tolist()) for k in SETTINGS if k not in ['load_in','load_out','pot_1','pot_2'] and k in g and (g[k]!=baseline[k]).any()}
     declared=target_list[int(flag)];assert set(changed)==set(declared),(path.stem,flag,changed,declared)
     clocks={'corr_dt_rpm_in_previous':corr(delta,g.rpm_in.to_numpy()[:-1]),'corr_dt_rpm_in_current':corr(delta,g.rpm_in.to_numpy()[1:]),'corr_dt_rpm_out_previous':corr(delta,g.rpm_out.to_numpy()[:-1]),'corr_dt_rpm_out_current':corr(delta,g.rpm_out.to_numpy()[1:])}
     regimes.append({'file':path.stem,'regime':int(flag),'rows':len(g),'declared_targets':declared,'actual_changed_settings':changed,'protocol_type':'reference' if not declared else ('physical-hatch' if declared==['hatch'] else ('mixed' if 'hatch' in declared else 'sensor-configuration')),'dt_seconds':quant(delta),**clocks})
   print(json.dumps({'file':path.name,'rows':len(df),'dt_median':row['dt_seconds'].get('median'),'nonfinite':row['nonfinite_numeric_values']}),flush=True)
 out=ROOT/'outputs/data_audit';out.mkdir(parents=True,exist_ok=True)
 (out/'inventory.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'regimes':regimes,'scope':'Observed protocol/clock qualification, not causal identification, new mechanism discovery or model score. No future elapsed time is authorized as known exogenous input without a separate availability audit.'},indent=2),encoding='utf-8')
 print(json.dumps({'files':len(rows),'total_rows':sum(r['rows'] for r in rows),'regime_blocks':len(regimes)}))
if __name__=='__main__':main()
