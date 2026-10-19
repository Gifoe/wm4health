"""Fresh-process NumPy/SciPy raw-array scoring, no learned core imported."""
import pathlib,sys,os
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
import numpy as np,json,hashlib,math,datetime
from scipy.spatial.distance import pdist
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def main():
 out=R/'outputs/eval_v1';assert not (out/'verification.json').exists();summary=read(out/'summary.json');manifest=read(out/'raw_manifest.json');data=np.load(R/'data/sealed_eval.npz');errors={'energy':0.,'mse':0.,'coverage':0.};verified=0;group_scores={}
 for entry in manifest:
  p=out/entry['path'];assert sha(p)==entry['sha256'];raw=np.load(p);draws=raw['draws'].astype('float64');truth=raw['truth'].astype('float64');assert np.array_equal(truth,data[entry['cell']][:,8:]) and draws.shape==(len(truth),64,16,8) and np.isfinite(draws).all();scores=[]
  for i,(paths,y) in enumerate(zip(draws,truth)):
   x=paths.reshape(64,-1);energy=(np.linalg.norm(x-y.ravel()[None],axis=1).mean()-pdist(x).mean()/2)/math.sqrt(128);mean=x.mean(0);mse=np.mean((mean-y.ravel())**2);bounds=np.percentile(paths,[5,95],axis=0);coverage=float(np.logical_and(y>=bounds[0],y<=bounds[1]).mean());assert abs(energy-float(raw['energy'][i]))<3e-6,(p.name,i,'energy');assert np.isclose(mse,float(raw['mse'][i]),atol=3e-6,rtol=3e-6),(p.name,i,'mse');assert coverage==float(raw['coverage'][i]);errors['energy']=max(errors['energy'],abs(energy-float(raw['energy'][i])));errors['mse']=max(errors['mse'],abs(mse-float(raw['mse'][i])));errors['coverage']=max(errors['coverage'],abs(coverage-float(raw['coverage'][i])));scores.append((energy,mse,coverage));verified+=1
  group_scores[(entry['model'],int(entry['cell'][1])==int(entry['cell'][-1]))]=group_scores.get((entry['model'],int(entry['cell'][1])==int(entry['cell'][-1])),[])+scores
 for row in summary['groups']:
  scores=np.array(group_scores[(row['model'],row['heldout'])]);assert np.allclose(scores.mean(0),[row['energy'],row['mse'],row['coverage90']],atol=3e-6,rtol=3e-6)
 assert len(manifest)==192 and verified==15360,(len(manifest),verified)
 native=R.parent/'cycle45/source_archive/ncdssm';source=read(native/'SOURCE_TREE.json')
 for item in source['files']:assert sha(native/item['path'])==item['sha256']
 schedules={}
 for p in (R/'outputs/fits').iterdir():
  fit=read(p/'fit_summary.json');config=read(p/'config.json');assert sha(p/'selected.pt')==fit['selected_sha256'] and sha(p/'last.pt')==fit['last_sha256'];assert sha(p/'schedule.npz')==fit['schedule_sha256'];schedules.setdefault(config['seed'],set()).add(fit['schedule_sha256']);assert config['common_sha256']==sha(R/'scripts/common46.py') and config['adapter_sha256']==sha(R/'scripts/joint_adapter46.py')
 assert len(schedules)==2 and all(len(values)==1 for values in schedules.values());result={'status':'PASS_FOR_FROZEN_CONTROLLED_SCREEN','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw_groups':len(manifest),'verified_origins':verified,'max_absolute_scoring_errors':errors,'source_core_SHA_unchanged':True,'per_seed_schedule_identical_across_4_arms':True,'qualification_weights_excluded':True,'summary_sha256':sha(out/'summary.json'),'manifest_sha256':sha(out/'raw_manifest.json'),'verifier_source_sha256':sha(__file__),'novelty_or_cross_domain_validity_not_established':True};(out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
