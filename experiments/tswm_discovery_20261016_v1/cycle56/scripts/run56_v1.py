"""Fixed-bank-count true-point gradient audit; no learned parameter updates."""
from common56 import *
import time
import numpy as np
import jax,jax.numpy as j
from scipy.stats import t as student_t
from math56 import sample_loss,population_loss,corr

def summarize(x,alpha=.05):
    x=np.asarray(x);mean=x.mean(0);se=x.std(0,ddof=1)/np.sqrt(len(x));q=student_t.ppf(1-alpha/2,len(x)-1)
    return {'mean':mean.tolist(),'mc_se':se.tolist(),'interval_low':(mean-q*se).tolist(),'interval_high':(mean+q*se).tolist(),'confidence':1-alpha,'independent_banks':len(x)}

def main():
    started=time.time();pins=source_check();qual=read(OUT/'QUALIFICATION56_V1.json')
    assert qual['status']=='PASS_NATIVE_GRADIENTS_AND_EXACT_REFERENCE_EXPECTATION'
    assert qual['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    storage=read(PROJECT/'STORAGE_LOCATION_V1.json')
    artifact=pathlib.Path(storage['artifact_root'])/'cycle56/gradient_v1'
    artifact.mkdir(parents=True,exist_ok=False)
    records=[];laws=[];nrep=256;chunk=16;a=.95
    for case,(r,T) in enumerate([(.3,6),(.3,24),(3.,6),(3.,24)]):
        rng=np.random.default_rng(5601+case)
        S=np.asarray(corr(a,T))+r*np.eye(T);R=np.linalg.cholesky(S)
        bank=rng.normal(size=(nrep,512,T))@R.T
        external=rng.normal(size=(nrep,512,T))@R.T
        raw=artifact/f'banks_case{case}.npz'
        with raw.open('xb') as f:np.savez(f,bank=bank,external=external,S=S)
        laws.append({'case':case,'a_star':a,'r':r,'T':T,'seed':5601+case,'path':str(raw),'sha256':sha(raw)})
        theta=j.array([a,np.log(1/r),0.])
        settings=[(B,1e-5) for B in [4,16,32,128,512]]+[(32,0.)]
        for B,jitter in settings:
            M=min(8,B);targets=bank[:,:M];reference=bank[:,:B];ext=external[:,:B]
            arrays={};outcomes=[]
            for fixed in [False,True]:
                pg=np.asarray(jax.grad(lambda z:population_loss(z,a,r,T,fixed,jitter))(theta))
                functions={mode:jax.jit(jax.vmap(jax.value_and_grad(lambda z,x,y,mode=mode:sample_loss(z,x,y,1+r,mode,fixed,jitter),argnums=0),in_axes=(None,0,0))) for mode in ['integrated','self','external','loo']}
                values={};grads={}
                for mode,fn in functions.items():
                    vs=[];gs=[];refs=ext if mode=='external' else reference
                    for first in range(0,nrep,chunk):
                        v,g=fn(theta,j.asarray(targets[first:first+chunk]),j.asarray(refs[first:first+chunk]))
                        vs.append(np.asarray(v));gs.append(np.asarray(g))
                    values[mode]=np.concatenate(vs);grads[mode]=np.concatenate(gs)
                    arrays[f'fixed{int(fixed)}_{mode}_gradient']=grads[mode]
                    arrays[f'fixed{int(fixed)}_{mode}_loss']=values[mode]
                for mode in ['self','external','loo']:
                    deviation=grads[mode]-grads['integrated'];cv=deviation+pg
                    primary=bool(B==32 and jitter==1e-5 and mode=='self')
                    row={'case':case,'a_star':a,'r':r,'T':T,'B':B,'targets':M,'jitter':jitter,'fixed_q':fixed,'reference':mode,'actual_reference_size':B-1 if mode=='loo' else B,'parameter_order':['a','log_lambda','log_b'],'population_gradient':pg.tolist(),'raw_gradient':summarize(grads[mode]),'integrated_target_gradient':summarize(grads['integrated']),'finite_reference_deviation':summarize(deviation),'gradient_control_variate':summarize(cv),'primary_family_member':primary}
                    if primary:row['primary_bonferroni_deviation']=summarize(deviation,.05/8)
                    outcomes.append(row);records.append(row)
                arrays[f'fixed{int(fixed)}_population_gradient']=pg
                arrays[f'fixed{int(fixed)}_self_minus_external']=grads['self']-grads['external']
            dest=artifact/f'gradients_case{case}_B{B}_j{jitter:.0e}.npz'
            with dest.open('xb') as f:np.savez(f,**arrays)
            receipt={'utc':utc(),'case':case,'B':B,'jitter':jitter,'path':str(dest),'sha256':sha(dest),'rows':outcomes}
            dump(OUT/f'case{case}_B{B}_j{jitter:.0e}.json',receipt)
            print(json.dumps({'status':'CELL_COMPLETE','case':case,'B':B,'jitter':jitter,'elapsed_seconds':time.time()-started}),flush=True)
    assert source_check()==pins
    result={'utc':utc(),'status':'FINITE_REFERENCE_GRADIENT_AUDIT_COMPLETE_PENDING_INTERPRETATION','records':records,'laws':laws,'n_banks':nrep,'numerical_training_updates':0,'accepted_final_directions':0,'qualification_sha256':sha(OUT/'QUALIFICATION56_V1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'script_sha256':sha(__file__),'math_sha256':sha(ROOT/'scripts/math56.py'),'native_source_sha256':pins,'elapsed_seconds':time.time()-started,'overall_goal':'ACTIVE_UNACHIEVED'}
    dump(OUT/'GRADIENT56_V1.json',result)
    print(json.dumps({'status':result['status'],'rows':len(records),'elapsed_seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
