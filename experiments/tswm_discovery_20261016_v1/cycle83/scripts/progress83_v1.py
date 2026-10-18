"""Read complete tasks only; partial unverified diagnostic, never model selection."""
import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import numpy as np
def main():
 root=storage('artifact')/'composition_screen_v1';rows=[]
 for item in read(OUT/'DATA_MANIFEST83_V1.json')['rows']:
  tag=item['tag'];paths=[root/f'{tag}_{kind}_ic{j}_clamp{c}.npz' for kind in ['FIM_local_composed','weak_polynomial3','weak_rff128'] for j in range(2) for c in range(2)]
  if not all(p.exists() for p in paths):continue
  for kind in ['FIM_local_composed','weak_polynomial3','weak_rff128']:
   scores=[]
   for j in range(2):
    for c in range(2):
     with np.load(root/f'{tag}_{kind}_ic{j}_clamp{c}.npz',allow_pickle=False) as a:
      p=a['prediction'];y=a['target'];ok=p.shape==y.shape and np.isfinite(p).all()
      if ok:
       p=p[:,1:] if c else p;y=y[:,1:] if c else y;scores.append(float(np.mean(np.sqrt(np.mean((p-y)**2,0))/np.maximum(y.std(0),1e-12))))
   rows.append(dict(tag=tag,kind=kind,complete=len(scores),expected=4,finite_median=float(np.median(scores)) if scores else None))
 print(json.dumps(dict(utc=utc(),status='PARTIAL_UNVERIFIED_NOT_FOR_SELECTION',rows=rows)),flush=True)
if __name__=='__main__':main()
