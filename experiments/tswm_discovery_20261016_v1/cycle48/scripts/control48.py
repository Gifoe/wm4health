"""Complete-history linear ordinary controls, selected on the same VAL only."""
from common48_v3 import *
def features(data):return np.concatenate((data['history'].numpy()[:,:,0],np.sin(2*np.pi*data['phase'].numpy())[:,None],np.cos(2*np.pi*data['phase'].numpy())[:,None],np.ones((len(data['history']),1))),axis=1).astype('float64')
def main():
 out=R/'outputs/control_v1';assert not out.exists();out.mkdir(parents=True);results=[];manifest=[]
 for case in CASES:
  train=load(case,'train');val=load(case,'val');x=features(train);y=train['future'].numpy()[:,:,0].astype('float64');vx=features(val);vy=val['future'].numpy()[:,:,0].astype('float64');candidates=[];weights=[];gram=x.T@x;cross=x.T@y
  for penalty in [1e-4,1e-2,1.,100.,10000.]:
   regularizer=np.eye(x.shape[1])*penalty;regularizer[-1,-1]=0.;w=np.linalg.solve(gram+regularizer,cross);error=float(np.mean((vx@w-vy)**2));candidates.append({'ridge':penalty,'val_mse':error});weights.append(w)
  index=int(np.argmin([c['val_mse'] for c in candidates]));path=out/(case+'_ridge.npz');np.savez_compressed(path,weights=weights[index]);manifest.append({'case':case,'path':path.name,'sha256':sha(path)});results.append({'case':case,'selected':candidates[index],'candidates':candidates,'nominal_fitted_coefficients':int(weights[index].size),'train_origins':len(x),'val_origins':len(vx),'EVAL_opened':False})
 dump(out/'selection.json',{'status':'ALL_RIDGE_SELECTIONS_COMPLETE_EVAL_UNOPENED','utc':now(),'cases':results,'files':manifest,'data_manifest_sha256':sha(R/'data/MANIFEST.json'),'source_sha256':sha(__file__)});print(json.dumps(results),flush=True)
if __name__=='__main__':main()
