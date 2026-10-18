import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common67_v1 import *
m=read(OUT/'DATA_SOURCE67_V4.json');assert read(OUT/'DATA_AUDIT67_V4.json')['subjects']==45
files=sorted(m['files'],key=lambda x:pathlib.Path(x['path']).name);permutation=np.random.default_rng(67021).permutation(45);splits={part:[pathlib.Path(files[i]['path']).name for i in permutation[start:end]] for part,start,end in [('TRAIN',0,27),('VALIDATION',27,36),('EVALUATION',36,45)]}
dump(OUT/'PARTICIPANT_SPLIT67_V1.json',dict(utc=utc(),seed=67021,split=splits,source_receipt_sha256=sha(OUT/'DATA_SOURCE67_V4.json'),protocol_sha256=sha(ROOT/'PROTOCOL67_V1.md'),forecast_scores_used=False))
items={s:[] for s in splits};data={s:collections.defaultdict(list) for s in splits};excluded=[]
for rec in files:
 name=pathlib.Path(rec['path']).name;part=next(s for s,names in splits.items() if name in names);t,v,meal=csv_record(rec);y=v['Libre GL'];u=np.stack([v[n] for n in ['Carbs','Protein','Fat']]);bad_meal=meal&(~np.isfinite(u).all(0));u=np.where(np.isfinite(u),u,0)
 origins=list(range(L,len(t)-H+1,720)) if part=='TRAIN' else [2880,5760,8640,11520]
 for o in origins:
  reason=None
  if o+H>len(t):reason='incomplete_window'
  elif not np.all(np.diff(t[o-L:o+H].astype('int64'))==60_000_000_000):reason='non_one_minute_window'
  elif bad_meal[o-L:o+H].any():reason='missing_macro_on_declared_meal'
  else:
   history_y=ffill15(y[o-L:o]);mask=np.isfinite(y[o:o+H])
   if not np.isfinite(history_y).all():reason='context_missing_after_causal_ffill15'
   elif mask.mean()<.95:reason='target_availability_below_95percent'
  if reason:excluded.append(dict(participant=name,split=part,origin=o,reason=reason));continue
  history=np.r_[history_y[None],u[:,o-L:o]].astype('float32');known=u[:,o:o+H].astype('float32');target=y[o:o+H].astype('float32');case=dict(id=name.removesuffix('.csv')+f'_o{o}',participant=name,split=part,origin=o,context_first=str(t[o-L]),forecast_first=str(t[o]),forecast_last=str(t[o+H-1]),target_points=int(mask.sum()),source_sha256=rec['sha256'])
  items[part].append(case)
  for k,val in dict(history=history,known=known,target=target,mask=mask,features=feature(history,known)).items():data[part][k].append(val)
receipts={}
for part in splits:
 assert len(items[part])>0;ap=artifact_root()/f'CASES_{part}67_V1.npz';assert not ap.exists();np.savez_compressed(ap,**{k:np.stack(v) for k,v in data[part].items()});receipts[part]=dict(path=str(ap),sha256=sha(ap),cases=items[part],count=len(items[part]),participants=len(set(x['participant'] for x in items[part])))
dump(OUT/'CASE_MANIFEST67_V1.json',dict(utc=utc(),status='SPLITS_AND_CASES_FROZEN_BEFORE_MODELS',split_receipt_sha256=sha(OUT/'PARTICIPANT_SPLIT67_V1.json'),source_receipt_sha256=sha(OUT/'DATA_SOURCE67_V4.json'),protocol_sha256=sha(ROOT/'PROTOCOL67_V1.md'),subsets=receipts,excluded=excluded,channels=['Libre GL','Carbs','Protein','Fat'],observed_context_minutes=L,prediction_minutes=H,input_and_retained_target_dtype='float32',target_masks_preserve_source_missingness=True,model_execution=False,EVAL_forecast_scores_used=False))
print(json.dumps(dict(status='SPLITS_AND_CASES_FROZEN_BEFORE_MODELS',counts={s:r['count'] for s,r in receipts.items()},participants={s:r['participants'] for s,r in receipts.items()},excluded_by_reason=dict(collections.Counter(x['reason'] for x in excluded)))),flush=True)
