"""Train/validation-select ordinary causal controls, seal them, then score full heldout rows."""
from common59 import *
import numpy as np,time
from scipy.stats import t as student_t

RIDGES=[1e-4,.01,1.,100.]
AS=[0.,.5,.9,.99,.999,.9999]
METHODS=['static','play','ema','hybrid','arx']

def features(b):
    n,L=b.shape
    static=np.stack([b,b**3,b**5,np.tanh(b),np.diff(b,axis=1,prepend=b[:,:1])],axis=-1)
    radii=np.geomspace(.02,2.,16);state=np.broadcast_to(b[:,:1],(n,len(radii))).copy();play=np.empty((n,L,len(radii)))
    taus=np.array([4,16,64,256,1024,4096,16384,65536],dtype=float);alpha=np.exp(-1/taus);estate=np.broadcast_to(b[:,:1],(n,len(taus))).copy();ema=np.empty((n,L,len(taus)))
    for k in range(L):
        x=b[:,k:k+1];state=np.minimum(np.maximum(state,x-radii),x+radii);play[:,k]=(x-state)/radii
        estate=alpha*estate+(1-alpha)*x;ema[:,k]=x-estate
    return {'static':static,'play':np.concatenate([static,play],-1),'ema':np.concatenate([static,ema],-1),'hybrid':np.concatenate([static,play,ema],-1)}

def fit(x,y,reg):
    x=x.reshape(-1,x.shape[-1]);y=y.reshape(-1)
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-8);xn=x/scale
    beta=np.linalg.solve(xn.T@xn/len(y)+reg*np.eye(x.shape[-1]),xn.T@y/len(y))/scale
    return beta

def predict_arx(f,beta,a,h,p):
    pred=np.empty_like(h[:,p:]);prev=h[:,p-1].copy()
    drive=np.einsum('ntd,d->nt',f[:,p:],beta)
    for k in range(pred.shape[1]):prev=a*prev+drive[:,k];pred[:,k]=prev
    return pred

def interval(v):
    v=np.asarray(v);se=v.std(ddof=1)/np.sqrt(len(v));q=student_t.ppf(.975,len(v)-1)
    return [float(v.mean()-q*se),float(v.mean()+q*se)]

def main():
    started=time.monotonic();audit=read(OUT/'DATA_AUDIT59_V2.json');download=read(OUT/'DOWNLOAD59_V2.json')
    assert audit['shape_B_H']==[128,5008] and audit['duplicate_B_H_T_rows']==0
    for f in download['files']:assert sha(f['server_path'])==f['sha256']
    B=np.loadtxt(data_root()/'N87_5_B.csv.gz',delimiter=',');H=np.loadtxt(data_root()/'N87_5_H.csv.gz',delimiter=',');T=np.loadtxt(data_root()/'N87_5_T.csv.gz',delimiter=',')
    groups={}
    for i in range(len(B)):
        key=hashlib.sha256(B[i].astype('<f8').tobytes()+np.asarray(T[i],dtype='<f8').tobytes()).hexdigest();groups.setdefault(key,[]).append(i)
    keys=sorted(groups);perm=np.random.default_rng(5900).permutation(len(keys));a=int(.6*len(keys));b=a+int(.2*len(keys))
    splits={name:sorted(i for p in part for i in groups[keys[p]]) for name,part in zip(['train','validation','test'],[perm[:a],perm[a:b],perm[b:]])}
    assert min(len(v) for v in splits.values())>=10 and len(set(sum(splits.values(),[])))==len(B)
    split={'utc':utc(),'status':'FULL_RAW_ROW_GROUPS_SEALED_BEFORE_FITTING','seed':5900,'groups':len(groups),'rows':splits,'grouping':'Exact raw B+T, independent of H targets','source_data_audit_sha256':sha(OUT/'DATA_AUDIT59_V2.json'),'protocol_sha256':sha(ROOT/'BASELINE_SCREEN59_V1.md'),'prior_metadata_reads_included_test_arrays':True}
    dump(OUT/'SPLIT59_V1.json',split)
    tr=np.array(splits['train']);va=np.array(splits['validation']);te=np.array(splits['test'])
    bscale=float(B[tr].std());hscale=float(H[tr].std());assert bscale>0 and hscale>0
    bn=B/bscale;hn=H/hscale;F=features(bn)
    # Causal bank execution must not depend on unobserved future input suffixes.
    check=bn[:1].copy();cut=2500;check[:,cut:]+=.37
    Fc=features(check)
    assert all(np.array_equal(Fc[k][:,:cut],F[k][:1,:cut]) for k in F)
    folder=data_root()/'baseline_screen_v1';folder.mkdir(exist_ok=False)
    candidates=[];selected=[];memories={}
    for p in [100,1252]:
        for method in METHODS:
            fs=F['static'] if method=='arx' else F[method]
            if method=='arx':
                # Intercept is an ordinary fixed feature in ARX (not a future-response statistic).
                fs=np.concatenate([fs,np.ones((*fs.shape[:2],1))],axis=-1)
                choices=[]
                for aa in AS:
                    target=hn[tr,p:]-aa*hn[tr,p-1:-1]
                    for reg in RIDGES:
                        beta=fit(fs[tr,p:],target,reg);prediction=predict_arx(fs[va],beta,aa,hn[va],p)
                        loss=float(np.mean((prediction-hn[va,p:])**2));assert np.isfinite(loss)
                        choices.append((loss,reg,aa,beta));candidates.append({'prefix':p,'method':method,'ridge':reg,'a':aa,'validation_mse':loss})
                best=min(enumerate(choices),key=lambda t:(t[1][0],t[0]))[1];loss,reg,aa,beta=best
            else:
                centered=fs-fs[:,:p].mean(axis=1,keepdims=True);target=hn[tr,p:]-hn[tr,:p].mean(axis=1,keepdims=True);choices=[]
                for reg in RIDGES:
                    beta=fit(centered[tr,p:],target,reg)
                    prediction=hn[va,:p].mean(axis=1,keepdims=True)+np.einsum('ntd,d->nt',centered[va,p:],beta)
                    loss=float(np.mean((prediction-hn[va,p:])**2));assert np.isfinite(loss)
                    choices.append((loss,reg,None,beta));candidates.append({'prefix':p,'method':method,'ridge':reg,'a':None,'validation_mse':loss})
                best=min(enumerate(choices),key=lambda t:(t[1][0],t[0]))[1];loss,reg,aa,beta=best
            tag=f'{method}_prefix{p}';path=folder/(tag+'.npz')
            with path.open('xb') as stream:np.savez(stream,beta=beta,bscale=bscale,hscale=hscale,a=np.nan if aa is None else aa)
            selected.append({'tag':tag,'method':method,'prefix':p,'ridge':reg,'a':aa,'validation_mse':loss,'coefficient_count':len(beta),'model_file':str(path),'model_sha256':sha(path)})
            memories[tag]=(fs,beta,aa)
            print(json.dumps({'status':'VALIDATION_SELECTED_NO_TEST_SCORE','tag':tag,'validation_mse':loss}),flush=True)
    assert len(candidates)==80 and len(selected)==10
    seal={'utc':utc(),'status':'ALL_10_DETERMINISTIC_CONTROLS_SELECTED_AND_SEALED_BEFORE_TEST_SCORING','models':selected,'validation_candidates':candidates,'source_split_sha256':sha(OUT/'SPLIT59_V1.json'),'script_sha256':sha(__file__),'bscale_train_only':bscale,'hscale_train_only':hscale,'no_future_H_feature_or_normalization':True,'no_test_score_selection':True,'causal_feature_prefix_check_passed':True}
    dump(OUT/'MODEL_SEAL59_V1.json',seal)
    receipt={'utc':utc(),'status':'FIRST_TEST_PREDICTIVE_SCORE_AFTER_FIXED_MODEL_SEAL','model_seal_sha256':sha(OUT/'MODEL_SEAL59_V1.json'),'metadata_audit_previously_read_all_raw_arrays':True,'test_rows':len(te),'post_test_model_selection':False}
    dump(OUT/'TEST_SCORE_OPEN59_V1.json',receipt)
    scores=[];errors={}
    for row in selected:
        tag=row['tag'];fs,beta,aa=memories[tag];p=row['prefix']
        assert sha(row['model_file'])==row['model_sha256']
        if aa is None:
            centered=fs[te,p:]-fs[te,:p].mean(axis=1,keepdims=True)
            prediction=hn[te,:p].mean(axis=1,keepdims=True)+np.einsum('ntd,d->nt',centered,beta)
        else:prediction=predict_arx(fs[te],beta,aa,hn[te],p)
        error=np.square(prediction-hn[te,p:]).mean(axis=1);errors[tag]=error
        nrmse=np.sqrt(error/np.maximum(np.square(hn[te,p:]).mean(axis=1),1e-12))
        path=folder/(tag+'_test_predictions.npz')
        with path.open('xb') as stream:np.savez(stream,rows=te,prediction=prediction*hscale,truth=H[te,p:],row_mse_train_scale=error,row_nrmse=nrmse)
        scores.append({**row,'test_mse_train_H_variance':float(error.mean()),'test_mse_row_interval95':interval(error),'test_mean_row_nrmse':float(nrmse.mean()),'test_median_row_nrmse':float(np.median(nrmse)),'test_row_mse':error.tolist(),'test_rows':te.tolist(),'prediction_file':str(path),'prediction_sha256':sha(path)})
    comparisons=[]
    for p in [100,1252]:
        for method in METHODS[1:]:
            d=errors[f'static_prefix{p}']-errors[f'{method}_prefix{p}']
            comparisons.append({'prefix':p,'static_minus_method':method,'paired_row_mean_improvement':float(d.mean()),'paired_row_interval95':interval(d)})
    result={'utc':utc(),'status':'TEN_ORDINARY_CONTROLS_SCORED_ON_FIXED_HELDOUT_ROWS','models':scores,'paired_comparisons':comparisons,'split_counts':{k:len(v) for k,v in splits.items()},'model_seal_sha256':sha(OUT/'MODEL_SEAL59_V1.json'),'test_score_receipt_sha256':sha(OUT/'TEST_SCORE_OPEN59_V1.json'),'candidate_fits':80,'selected_deterministic_fits':10,'neural_training_updates':0,'seconds':time.monotonic()-started,'scope':'One real tutorial material, one temperature, one group-disjoint row split. Not cross-system generalization, no identified noise floor, unknown reset/prehistory, no proof all strong controls exhausted. No new method.','overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'BASELINE_RESULTS59_V1.json',result)
    print(json.dumps({'status':result['status'],'counts':result['split_counts'],'seconds':result['seconds'],'scores':[{k:r[k] for k in ['tag','test_mse_train_H_variance','test_mean_row_nrmse']} for r in scores]}),flush=True)
if __name__=='__main__':main()
