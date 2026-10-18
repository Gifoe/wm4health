from common90_v1 import *
import numpy as np,time
import csdi_core89_v1 as core
from ordinary_control90_v1 import rollout
from eval_persistent89_v1 import weighted_score
from eval_csdi89_v1 import distance
from plan_data88_v1 import advance

def dense_bank(actions,family):
    values=np.linspace(.4,2.,161);n=len(actions);a=np.repeat(actions,len(values),axis=0);state=np.zeros((len(a),3));theta=np.tile([1.,.35,1.,.5,3.],(len(a),1));theta[:,0]=np.tile(values,n);control=np.zeros((len(a),2));out=[]
    for t in range(8,72):control[:,0]=a[:,t];state=advance(state,control,theta,family);out.append(state[:,:2].copy())
    return np.stack(out,1).reshape(n,161,64,2)

def main():
    assert not (O/'ORDINARY_CONFIRMATION90_V1.json').exists();started=time.perf_counter()
    ordinary=read(O/'ORDINARY_CONTROL90_V1.json');library=read(O/'LIBRARY90_V1.json');assert ordinary['status']=='CAL_COMPETENCE_PASS' and library['status']=='CAL_LIBRARY_PASS'
    for rec in [ordinary,library]:assert sha(rec['checkpoint_path'])==rec['checkpoint_sha256']
    rng=np.random.default_rng(20269080);arrays={};queries=[];joint=[];metadata=[];maxexact=0.
    for family in [0,1]:
        a=core.draw_actions(rng,64);truth=np.stack([core.simulate(a,w,family)[:,8:] for w in range(9)],axis=1);dense=dense_bank(a,family)
        arrays[f'f{family}_actions']=a;arrays[f'f{family}_reference']=truth;arrays[f'f{family}_dense_reference']=dense
        for arm,receipt in [('unsupervised_pruned',ordinary),('supervised_library',library)]:
            with np.load(receipt['checkpoint_path']) as f:
                scale=f['state_scale'].copy() if arm=='unsupervised_pruned' else f['scale'].copy();center=f['state_center'].copy() if arm=='unsupervised_pruned' else f['center'].copy()
                beta=f[f'f{family}_beta'].copy();fs=f[f'f{family}_feature_scale'].copy();pi=f[f'f{family}_pi' if arm=='unsupervised_pruned' else f'f{family}_prior'].copy()
            prediction=rollout(beta,fs,a,scale);assert np.isfinite(prediction).all()
            pnorm=((prediction-center)/scale).transpose(0,1,3,2);tnorm=((truth-center)/scale).transpose(0,1,3,2);dnorm=((dense-center)/scale).transpose(0,1,3,2)
            arrays[f'f{family}_{arm}_predictions']=prediction;arrays[f'f{family}_{arm}_pi']=pi;cost=[];labels=[]
            for q in range(64):
                score=weighted_score(pnorm[q],pi,tnorm[q]);labels.append(score['component_nearest_mechanism']);cost.append(distance(pnorm[q],dnorm[q])**2)
                pe=pi@np.stack([np.max(abs(prediction[q,:,:,0]),1)>.75,prediction[q,:,-1,0]>.25],axis=1)
                te=core.PRIOR@np.stack([np.max(abs(truth[q,:,:,0]),1)>.75,truth[q,:,-1,0]>.25],axis=1)
                queries.append(dict(arm=arm,family=family,query=q,group=q//4,**score,event_probability=pe.tolist(),true_event_probability=te.tolist(),event_absolute_error=abs(pe-te).tolist()))
            cost=np.stack(cost);labels=np.array(labels)
            for group in range(16):
                inds=group*4+np.arange(4)
                for K in [2,3,4]:
                    ix=inds[:K];same=np.all(labels[ix]==labels[ix[:1]],axis=0)
                    freq=np.stack([np.bincount(labels[i],weights=pi,minlength=9) for i in ix]);common=np.min(cost[ix].mean(0),axis=-1);separate=np.min(cost[ix],axis=-1).mean(0)
                    assert (common>=separate-1e-12).all()
                    joint.append(dict(arm=arm,family=family,group=group,K=K,nearest_reference_bin_shared_agreement=float(pi@same),nearest_reference_bin_independent_agreement=float(np.prod(freq,axis=0).sum()),common_k_rms=float(pi@np.sqrt(common)),separate_k_rms=float(pi@np.sqrt(separate)),common_minus_separate_mse=float(pi@(common-separate)),single_query_marginal_change=0.))
            # Evaluator qualification on every exact shared physical mechanism.
            for group in range(16):
                ix=group*4+np.arange(4);exactcost=np.stack([distance(tnorm[i],dnorm[i])**2 for i in ix]);gap=np.min(exactcost.mean(0),axis=-1)-np.min(exactcost,axis=-1).mean(0);maxexact=max(maxexact,float(abs(gap).max()))
            metadata.append(dict(arm=arm,family=family,components=len(pi),checkpoint_sha256=receipt['checkpoint_sha256']))
    summary=[]
    for arm in ['unsupervised_pruned','supervised_library']:
        for family in [0,1]:
            rows=[r for r in queries if r['arm']==arm and r['family']==family];j=[r for r in joint if r['arm']==arm and r['family']==family and r['K']==4]
            delta=np.array([r['energy_minus_deterministic'] for r in rows]).reshape(16,4).mean(1);boot=np.random.default_rng(20269081).integers(0,16,(10000,16));means=delta[boot].mean(1)
            ratio=float(np.mean([r['off_manifold_ratio'] for r in rows]));energy=float(np.mean([r['energy_score'] for r in rows]))
            summary.append(dict(arm=arm,family=family,energy_score=energy,reference_population_energy=float(np.mean([r['exact_population_energy_score'] for r in rows])),energy_minus_deterministic=float(delta.mean()),energy_delta_group_bootstrap95=[float(np.quantile(means,.025)),float(np.quantile(means,.975))],off_manifold_ratio=ratio,competence_screen_passed=delta.mean()<0 and ratio<=.5,event_probability_MAE=np.mean([r['event_absolute_error'] for r in rows],axis=0).tolist(),K4_nearest_bin_shared_agreement=float(np.mean([r['nearest_reference_bin_shared_agreement'] for r in j])),K4_nearest_bin_independent_agreement=float(np.mean([r['nearest_reference_bin_independent_agreement'] for r in j])),K4_common_k_rms=float(np.mean([r['common_k_rms'] for r in j])),K4_separate_k_rms=float(np.mean([r['separate_k_rms'] for r in j])),K4_common_minus_separate_mse=float(np.mean([r['common_minus_separate_mse'] for r in j]))))
    assert maxexact<1e-12,maxexact
    dest=Path(STORAGE['artifact_root'])/'cycle90/ordinary_confirmation_v1';dest.mkdir(parents=True,exist_ok=True);p=dest/'fresh_queries_predictions.npz'
    with p.open('xb') as f:np.savez_compressed(f,**arrays)
    result=dict(utc=utc(),status='BASELINE_ONLY_INDEPENDENT_CONFIRMATION_COMPLETE',summary=summary,queries=queries,joint=joint,models=metadata,payload_path=str(p),payload_sha256=sha(p),fresh_query_seed=20269080,sealed_native_eval_seed2026905_generated=False,exact_shared_physics_grid_gap_max_abs=maxexact,protocol_sha256=sha(C/'ORDINARY_CONFIRMATION90_V1.md'),known_polynomial_markov_bias=True,supervised_library_uses_train_group_identities=True,new_method_claim=False,full_goal_achieved=False,accepted_directions=0,elapsed_seconds=time.perf_counter()-started)
    write(O/'ORDINARY_CONFIRMATION90_V1.json',result);log(dict(stage='ordinary_confirmation_closed',summary=summary,elapsed_seconds=result['elapsed_seconds']))

if __name__=='__main__':main()
