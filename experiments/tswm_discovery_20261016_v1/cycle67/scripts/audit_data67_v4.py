import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common67_v1 import *
m=read(OUT/'DATA_SOURCE67_V4.json');assert m['member_CRC32_source_verified'];rows=[]
for rec in m['files']:
 t,values,meal=csv_record(rec);dt=np.diff(t.astype('int64'))/60e9;channels={}
 for name,v in values.items():
  ok=np.isfinite(v);ix=np.flatnonzero(ok);gaps=np.diff(t.astype('int64')[ix])/60e9 if len(ix)>1 else np.array([])
  channels[name]=dict(finite=int(ok.sum()),missing=int((~ok).sum()),finite_fraction=float(ok.mean()),min=float(v[ok].min()) if ok.any() else None,max=float(v[ok].max()) if ok.any() else None,positive=int(np.sum(v[ok]>0)),observed_gap_minutes_quantiles=np.quantile(gaps,[0,.5,.9,.99,1]).tolist() if len(gaps) else None)
 rows.append(dict(file=pathlib.Path(rec['path']).name,sha256=rec['sha256'],rows=len(t),columns=rec['columns'],first=str(t[0]),last=str(t[-1]),timestamps_unique=bool(len(np.unique(t))==len(t)),timestamps_increasing=bool(np.all(dt>0)),row_spacing_minutes_values_counts={str(k):int(v) for k,v in zip(*np.unique(dt,return_counts=True))},channels=channels,meal_type_nonempty=int(meal.sum()),carbs_positive_without_meal_type=int(((values['Carbs']>0)&~meal).sum())))
obj=dict(utc=utc(),status='RAW_NUMERIC_PROVENANCE_AND_QUALITY_AUDITED_NO_MODEL',source_receipt_sha256=sha(OUT/'DATA_SOURCE67_V4.json'),subjects=len(rows),records=rows,observational_data=True,clinical_causal_truth=False,model_execution=False,forecast_scores_computed=False,local_result_files_written=0,parser='stdlib CSV/ISO datetime plus NumPy; pandas absent and not installed')
dump(OUT/'DATA_AUDIT67_V4.json',obj);print(json.dumps(dict(status=obj['status'],subjects=len(rows),total_rows=sum(x['rows'] for x in rows),first_record=rows[0],channel_finite_fraction_ranges={name:[min(x['channels'][name]['finite_fraction'] for x in rows),max(x['channels'][name]['finite_fraction'] for x in rows)] for name in ['Libre GL','Dexcom GL','HR','Carbs']})),flush=True)
