"""Independent score/data/protocol checks; use raw paths, not production scores()."""
import sys,pathlib,json,hashlib,math
R=pathlib.Path(__file__).resolve().parents[1];ROOT=R.parent
sys.path.insert(0,str(ROOT/'cycle29/scripts'));import runtime_profile
import numpy as np
from scipy.spatial.distance import pdist
O=R/'outputs';S=O/'screen_v1';assert (S/'COMPLETE.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.loads((S/'results.json').read_text());assert len(rows)==8
protocol=sha(R/'ORDINARY_SCREEN_PROTOCOL.md');maxerr=0.;cells=0;raws=0
intake=json.loads((O/'intake.json').read_text());splits=[]
for name in ['ett','jena']:
 info=intake['data'][name];z=np.load(R/'assets'/f'{name}.npz');assert sha(R/'assets'/f'{name}.npz')==info['array_sha256']
 y=z['y'];t=z['timestamps'];assert np.all(np.diff(t)==3600*10**9)
 train=info['splits']['train'];a=train['stop_exclusive'];valid=np.isfinite(y[:a]);mu=y[:a][valid].mean();sd=y[:a][valid].std()
 assert abs(mu-info['train_mean'])<1e-12 and abs(sd-info['train_std'])<1e-12
 assert np.allclose(z['normalized'],(y-mu)/sd,equal_nan=True)
 for split in ['train','val','test']:
  es=np.load(R/'assets'/f'{name}_{split}_origins.npy');spec=info['splits'][split]
  assert len(es)==spec['complete_origins'] and np.all(es-192>=spec['start']) and np.all(es+96<=spec['stop_exclusive'])
  splits.append({'data':name,'split':split,'origins':len(es),'all_history_and_targets_in_split':True})
for row in rows:
 assert row['test_scored'] is False
 D=S/row['key'];assert sha(D/'selected.pt')==row['selected_checkpoint_sha256']
 cfg=json.loads((S/'config.json').read_text());assert cfg['protocol_sha256']==protocol
 best=min(row['curves'],key=lambda x:x['selection_energy']);assert best['step']==row['selected_step']
 name=row['data'];v=np.load(R/'assets'/f'{name}.npz')['normalized'];valspec=intake['data'][name]['splits']['val']
 for ev in row['evals']:
  p=D/ev['raw'];assert sha(p)==ev['raw_sha256'];z=np.load(p);truth=z['truth'];es=z['origins']
  assert np.array_equal(truth,v[es[:,None]+np.arange(64,96)[None]].astype(np.float32))
  assert np.all(es-192>=valspec['start']) and np.all(es+96<=valspec['stop_exclusive'])
  for idx,r in enumerate([0,8,32]):
   x=z[f'r{r}'].astype(np.float64);m,n,h=x.shape;assert m==32 and h==32 and n==len(es) and np.isfinite(x).all()
   e=[];cr=[];mc=[];mae=[];cov=[]
   for j in range(n):
    b=x[:,j,:];w=truth[j].astype(np.float64)
    e.append((np.linalg.norm(b-w,axis=-1).mean()-pdist(b,'euclidean').sum()/(m*(m-1)))/math.sqrt(h))
    cr.append(np.mean([np.abs(b[:,k]-w[k]).mean()-pdist(b[:,k,None],'cityblock').sum()/(m*(m-1)) for k in range(h)]))
    q=b.mean(-1);mc.append(np.abs(q-w.mean()).mean()-pdist(q[:,None],'cityblock').sum()/(m*(m-1)))
    mae.append(np.abs(b.mean(0)-w).mean());cov.append(((w>=np.quantile(b,.1,axis=0))&(w<=np.quantile(b,.9,axis=0))).mean())
   independent={'energy':np.mean(e),'marginal_crps':np.mean(cr),'mean_crps':np.mean(mc),'mae':np.mean(mae),'coverage80':np.mean(cov)}
   for k,value in independent.items():
    err=abs(value-ev['means_by_reveal'][idx][k]);maxerr=max(maxerr,err);assert err<1e-6,(row['key'],r,k,err);cells+=1
  raws+=1
out={'status':'PASS','fits':len(rows),'raw_path_archives':raws,'independent_score_cells':cells,'max_score_difference':maxerr,'protocol_sha256':protocol,'split_checks':splits,'test_scored':False,'native_api_qualification':json.loads((O/'qualification.json').read_text())['checks'],'limits':'No independent GPU retraining; verifier reimplements raw-path statistics and data/split alignment.'}
(O/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
