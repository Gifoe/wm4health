from common62 import *
import numpy as np
e=read(OUT/'EVALUATION62_V1.json');q=read(OUT/'QUALIFICATION62_V1.json');assert sha(ROOT/'PROTOCOL62_V1.md')==e['protocol_sha256']==q['protocol_sha256']
import pandas as pd
df=pd.read_csv(PROJECT/'cycle5/data/ETTh1.csv');raw=df.drop(columns=['date']).values;tr=q['train_boundary'];scale=np.std(raw[:tr],axis=0)
index={}
for row in e['rows']:
 tag=(row['key'],row['view'],row['inference_rng'],row['horizon']);assert tag not in index;index[tag]=row
max_mse=max_mae=0.;count=0
for r in e['artifacts']:
 assert sha(r['path'])==r['sha256'];saved=np.load(r['path'])
 for view,origin_field in [('official96','official_origins'),('common384','common_origins')]:
  pred=saved[view];orig=saved[origin_field];native=pred.ndim==4
  streams=e['inference_rng_streams'] if native else [None]
  for i,stream in enumerate(streams):
   p=pred[i] if native else pred
   for h in ([96] if view=='official96' else [96,192,384]):
    # Accumulate per origin with separately indexed truths; no reuse of scorer metrics helper.
    sq=[];absolute=[];channel_sq=[];channel_ab=[]
    for j,o in enumerate(orig):
     err=np.divide(np.subtract(np.asarray(p[j,:h],dtype=np.float64),raw[o:o+h]),scale)
     assert np.isfinite(err).all();sq.append(np.einsum('tc,tc->',err,err));absolute.append(np.abs(err).sum());channel_sq.append(np.square(err).sum(0));channel_ab.append(np.abs(err).sum(0))
    mse=float(np.sum(sq,dtype=np.float64)/(len(orig)*h*7));mae=float(np.sum(absolute,dtype=np.float64)/(len(orig)*h*7));row=index[(r['key'],view,stream,h)]
    dm=abs(mse-row['mse']);da=abs(mae-row['mae']);assert dm<1e-10 and da<1e-10,(r['key'],view,h,dm,da)
    assert np.allclose(np.sum(channel_sq,0)/(len(orig)*h),row['channel_mse'],atol=1e-10,rtol=0)
    assert np.allclose(np.sum(channel_ab,0)/(len(orig)*h),row['channel_mae'],atol=1e-10,rtol=0)
    max_mse=max(max_mse,dm);max_mae=max(max_mae,da);count+=1
assert count==len(e['rows'])
fits=[]
for p in sorted(OUT.glob('*_fit62_v1.json')):
 r=read(p);assert sha(r['selected_checkpoint'])==r['selected_sha256'] and sha(r['terminal_checkpoint'])==r['terminal_sha256'];assert r['data_sha256']==e['data_sha256'];fits.append(r['key'])
assert len(fits)==8
dump(OUT/'INDEPENDENT_REPLAY62_V1.json',{'utc':utc(),'status':'PASS','metric_rows':count,'max_mse_difference':max_mse,'max_mae_difference':max_mae,'all_selected_and_terminal_checkpoint_hashes_verified':True,'fit_count':len(fits),'evaluation_sha256':sha(OUT/'EVALUATION62_V1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL62_V1.md')})
print(json.dumps({'status':'PASS','rows':count,'max_mse_difference':max_mse,'max_mae_difference':max_mae}))
