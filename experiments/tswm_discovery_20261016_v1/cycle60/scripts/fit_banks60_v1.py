"""Validation-select immutable ordinary banks for the independent mechanical system."""
from common60 import *
import sys,numpy as np
sys.path.insert(0,str(PROJECT/'cycle59/scripts'))
from baselines59_v1 import features,fit,predict_arx
def main():
    data=read(OUT/'DATA60_V2.json');ds=data['systems']['pick_place'];parts={}
    for name in ['train','validation']:
        row=next(r for r in ds['splits'] if r['split']==name);assert sha(row['file'])==row['sha256']
        with np.load(row['file']) as a:parts[name]={'b':a['u'][:,:,0].astype('float64'),'h':a['y'][:,:,0].astype('float64')}
    tr,va=parts['train'],parts['validation'];Ft=features(tr['b']);Fv=features(va['b']);p=100;ridges=[1e-4,.01,1.,100.];ars=[0.,.5,.9,.99,.999,.9999]
    folder=paths()['checkpoint_root']/'pick_place_banks_v1';folder.mkdir(exist_ok=False);models=[];candidates=[]
    for method in ['static','play','ema','hybrid','arx']:
        if method=='arx':
            xt=np.concatenate([Ft['static'],np.ones((*Ft['static'].shape[:2],1))],-1);xv=np.concatenate([Fv['static'],np.ones((*Fv['static'].shape[:2],1))],-1);choices=[]
            for aa in ars:
                target=tr['h'][:,p:]-aa*tr['h'][:,p-1:-1]
                for reg in ridges:
                    beta=fit(xt[:,p:],target,reg);pred=predict_arx(xv,beta,aa,va['h'],p);loss=float(np.mean((pred-va['h'][:,p:])**2));choices.append((loss,reg,aa,beta));candidates.append({'method':method,'ridge':reg,'a':aa,'validation_mse':loss})
        else:
            xt=Ft[method]-Ft[method][:,:p].mean(1,keepdims=True);xv=Fv[method]-Fv[method][:,:p].mean(1,keepdims=True);target=tr['h'][:,p:]-tr['h'][:,:p].mean(1,keepdims=True);choices=[]
            for reg in ridges:
                beta=fit(xt[:,p:],target,reg);pred=va['h'][:,:p].mean(1,keepdims=True)+xv[:,p:]@beta;loss=float(np.mean((pred-va['h'][:,p:])**2));choices.append((loss,reg,None,beta));candidates.append({'method':method,'ridge':reg,'a':None,'validation_mse':loss})
        best=min(enumerate(choices),key=lambda t:(t[1][0],t[0]))[1];loss,reg,aa,beta=best;assert np.isfinite(loss)
        path=folder/(method+'.npz')
        with path.open('xb') as f:np.savez(f,beta=beta)
        models.append({'system':'pick_place','method':method,'prefix':p,'ridge':reg,'a':aa,'validation_mse':loss,'checkpoint':str(path),'checkpoint_sha256':sha(path),'coefficients':len(beta)})
    assert len(candidates)==40 and len(models)==5
    obj={'utc':utc(),'status':'ALL_FIVE_PICK_PLACE_ORDINARY_CONTROLS_VALIDATION_SELECTED_TEST_UNSCORED','models':models,'validation_candidates':candidates,'data_manifest_sha256':sha(OUT/'DATA60_V2.json'),'feature_source_sha256':sha(PROJECT/'cycle59/scripts/baselines59_v1.py'),'protocol_sha256':sha(ROOT/'PROTOCOL60_V1.md'),'script_sha256':sha(__file__),'no_test_target_reads':True,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'BANK_SEAL60_V1.json',obj);print(json.dumps({'status':obj['status'],'models':[{k:r[k] for k in ['method','validation_mse','ridge','a']} for r in models]}),flush=True)
if __name__=='__main__':main()
