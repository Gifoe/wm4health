"""Independent NumPy/SciPy scoring from saved EVAL trajectories, no training."""
import pathlib,sys,json,hashlib,datetime
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
import numpy as np
from scipy.spatial.distance import cdist
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
O=R/'outputs';out=O/'quality_verification_v1.json';assert not out.exists();summary=read(O/'quality_v1/quality_summary.json');assert summary['status']=='PASS' and summary['models']==16
checks=[]
for row in summary['rows']:
 raw=O/'quality_v1'/row['model']/'quality_arrays.npz';assert sha(raw)==row['raw_arrays_sha256'];z=np.load(raw);paths=z['paths'].astype(np.float64);targets=z['targets'].astype(np.float64);assert paths.shape==(64,64,16,64) and targets.shape==(64,16,64) and np.isfinite(paths).all();es=[];mses=[];coverage=[]
 for x,y in zip(paths,targets):
  flat=x.reshape(64,-1);target=y.ravel();dist=cdist(flat,flat,'euclidean');np.fill_diagonal(dist,0.);es.append((np.mean(np.sqrt(np.sum((flat-target)**2,axis=1)))-np.sum(dist)/(2*64*63))/np.sqrt(1024));mses.append(np.mean((x.mean(0)-y)**2));lo,hi=np.quantile(x,[.05,.95],axis=0);coverage.append(np.mean((y>=lo)&(y<=hi)))
 energy_error=float(np.max(np.abs(np.asarray(es)-row['joint_energy_by_origin'])));mse_error=float(np.max(np.abs(np.asarray(mses)-row['MSE_by_origin'])));coverage_error=float(np.max(np.abs(np.asarray(coverage)-row['coverage_by_origin'])));assert energy_error<1e-5 and mse_error<1e-5 and coverage_error==0,(row['model'],energy_error,mse_error,coverage_error)
 checks.append({'model':row['model'],'energy_max_abs':energy_error,'MSE_max_abs':mse_error,'coverage_max_abs':coverage_error,'independent_joint_energy_mean':float(np.mean(es))})
result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':len(checks),'origins':len(checks)*64,'checks':checks,'quality_summary_sha256':sha(O/'quality_v1/quality_summary.json'),'source_sha256':sha(pathlib.Path(__file__)),'numerics':'float64 NumPy + SciPy cdist, independent of torch scoring'};out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'status':'PASS','models':len(checks),'origins':len(checks)*64,'max_energy_abs':max(x['energy_max_abs'] for x in checks)}))
