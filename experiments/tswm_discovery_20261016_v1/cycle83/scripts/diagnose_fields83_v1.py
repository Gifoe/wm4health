"""Partial field-only readout; no fitting, selection or model changes."""
import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import numpy as np
def main():
 root=storage('artifact')/'composition_screen_v1';rows=[]
 for item in read(OUT/'DATA_MANIFEST83_V1.json')['rows']:
  tag=item['tag']
  if not (root/f'{tag}_weak_rff128_ic1_clamp1.npz').exists():continue
  for kind in ['FIM_local_composed','weak_polynomial3','weak_rff128']:
   pth=root/f'{tag}_{kind}_field.npz'
   with np.load(pth,allow_pickle=False) as a:
    p=a['prediction'];y=a['target'];ok=p.shape==y.shape and np.isfinite(p).all()
    if not ok:rows.append(dict(tag=tag,kind=kind,status='FAILED_FIELD'));continue
    rmse=np.sqrt(np.mean((p-y)**2,0));nrmse=rmse/np.maximum(y.std(0),1e-12);groups=[('fast_v',list(range(0,12,2))),('slow_w',list(range(1,12,2)))] if item['family']=='fhn' else [('all',list(range(7)))];bygroup=[dict(group=name,mean_coordinate_nrmse=float(np.mean(nrmse[idx])),mean_rmse=float(np.mean(rmse[idx])),mean_target_std=float(np.mean(y.std(0)[idx]))) for name,idx in groups];rows.append(dict(tag=tag,kind=kind,mean_coordinate_nrmse=float(nrmse.mean()),bygroup=bygroup))
 print(json.dumps(dict(utc=utc(),status='PARTIAL_UNVERIFIED_FIELD_DIAGNOSTIC',rows=rows)),flush=True)
if __name__=='__main__':main()
