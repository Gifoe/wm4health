"""Source/timestamp/channel quality only; no model or forecast target scoring."""
import pathlib,json,hashlib,datetime,sys
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';m=json.loads((OUT/'DATA_SOURCE67_V4.json').read_text());assert m['member_CRC32_source_verified'];rows=[]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for rec in m['files']:
 assert sha(rec['path'])==rec['sha256'];df=pd.read_csv(rec['path']);assert len(df)==rec['rows'];t=pd.to_datetime(df['Timestamp'],format='mixed',dayfirst=False,errors='raise');dt=np.diff(t.astype('int64').to_numpy())/60e9;channels={}
 for name in ['Libre GL','Dexcom GL','HR','Carbs','Protein','Fat']:
  v=pd.to_numeric(df[name],errors='coerce').to_numpy(dtype='float64');ok=np.isfinite(v);ix=np.flatnonzero(ok);gaps=np.diff(t.astype('int64').to_numpy()[ix])/60e9 if len(ix)>1 else np.array([])
  channels[name]=dict(finite=int(ok.sum()),missing=int((~ok).sum()),finite_fraction=float(ok.mean()),min=float(v[ok].min()) if ok.any() else None,max=float(v[ok].max()) if ok.any() else None,positive=int(np.sum(v[ok]>0)),observed_gap_minutes_quantiles=np.quantile(gaps,[0,.5,.9,.99,1]).tolist() if len(gaps) else None)
 rows.append(dict(file=pathlib.Path(rec['path']).name,sha256=rec['sha256'],rows=len(df),columns=df.columns.tolist(),first=str(t.iloc[0]),last=str(t.iloc[-1]),timestamps_unique=bool(not t.duplicated().any()),timestamps_increasing=bool(np.all(dt>0)),row_spacing_minutes_values_counts={str(k):int(v) for k,v in zip(*np.unique(dt,return_counts=True))},channels=channels,meal_type_nonempty=int(df['Meal Type'].notna().sum()),carbs_positive_without_meal_type=int(((pd.to_numeric(df['Carbs'],errors='coerce')>0)&df['Meal Type'].isna()).sum())))
obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='RAW_NUMERIC_PROVENANCE_AND_QUALITY_AUDITED_NO_MODEL',source_receipt_sha256=sha(OUT/'DATA_SOURCE67_V4.json'),subjects=len(rows),records=rows,observational_data=True,clinical_causal_truth=False,model_execution=False,forecast_scores_computed=False,local_result_files_written=0)
with (OUT/'DATA_AUDIT67_V3.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
print(json.dumps(dict(status=obj['status'],subjects=len(rows),total_rows=sum(x['rows'] for x in rows),first_record=rows[0],channel_finite_fraction_ranges={name:[min(x['channels'][name]['finite_fraction'] for x in rows),max(x['channels'][name]['finite_fraction'] for x in rows)] for name in ['Libre GL','Dexcom GL','HR','Carbs']})),flush=True)
