from common90_v1 import *
import numpy as np,time
import csdi_core89_v1 as core
from persistent_mixture89_v1 import polynomial
from eval_persistent89_v1 import weighted_score

def rollout(beta,feature_scale,actions,scale):
    state=np.zeros((len(actions),len(beta),2),np.float64);out=[]
    for t in range(core.CONTEXT,core.LENGTH):
        inp=np.concatenate([state,np.broadcast_to(actions[:,None,t,None],(*state.shape[:2],1))],axis=-1)
        with np.errstate(over='raise',invalid='raise'):
            phi=polynomial(inp)/feature_scale;state=state+np.einsum('nkf,kfd->nkd',phi,beta,optimize=True)*scale
        assert np.isfinite(state).all();out.append(state.copy())
    return np.stack(out,axis=2)

def main():
    assert not (O/'ORDINARY_CONTROL90_V1.json').exists();started=time.perf_counter()
    fit=read(ROOT/'cycle89/outputs/PERSISTENT_MIXTURE_FIT89_V1.json');assert sha(fit['checkpoint_path'])==fit['checkpoint_sha256']
    manifest=read(ROOT/'cycle89/outputs/CSDI_DATA89_V1.json');cal=next(r for r in manifest['files'] if r['split']=='cal');assert sha(cal['path'])==cal['sha256']
    with np.load(cal['path']) as f:actions=f['actions'].copy();families=f['family'].copy()
    with np.load(fit['checkpoint_path']) as f:
        scale=f['state_scale'].copy();center=f['state_center'].copy();models=[{k:f[f'f{i}_{k}'].copy() for k in ['beta','pi','var','feature_scale']} for i in [0,1]]
    summaries=[];queries=[];saved={};arrays={}
    for family,m in enumerate(models):
        keep=m['pi']*4096>=1.;removed=float(m['pi'][~keep].sum());assert removed<=1e-4
        pi=m['pi'][keep].copy();pi/=pi.sum();beta=m['beta'][keep].copy()
        a=actions[families==family];all_pred=rollout(beta,m['feature_scale'],a,scale)
        prednorm=((all_pred[:32]-center)/scale).transpose(0,1,3,2)
        bank=np.stack([core.simulate(a[:32],w,family)[:,core.CONTEXT:] for w in range(9)],axis=1)
        banknorm=((bank-center)/scale).transpose(0,1,3,2)
        rows=[]
        for q in range(32):rows.append(dict(query=q,family=family,**weighted_score(prednorm[q],pi,banknorm[q])))
        delta=float(np.mean([r['energy_minus_deterministic'] for r in rows]));ratio=float(np.mean([r['off_manifold_ratio'] for r in rows]))
        summaries.append(dict(family=family,removed_components=np.flatnonzero(~keep).tolist(),removed_probability=removed,retained_components=int(keep.sum()),finite_cal_queries=128,energy_score=float(np.mean([r['energy_score'] for r in rows])),energy_minus_deterministic=delta,off_manifold_ratio=ratio,nearest_mechanism_tv=float(np.mean([r['nearest_mechanism_tv'] for r in rows])),passed=delta<0 and ratio<=.5))
        queries.extend(rows);arrays[f'f{family}_predictions']=prednorm;arrays[f'f{family}_bank']=banknorm
        saved.update({f'f{family}_beta':beta,f'f{family}_feature_scale':m['feature_scale'],f'f{family}_pi':pi,f'f{family}_original_indices':np.flatnonzero(keep)})
    dest=Path(STORAGE['checkpoint_root'])/'cycle90/ordinary_control_v1';dest.mkdir(parents=True,exist_ok=True);p=dest/'pruned_selected.npz'
    with p.open('xb') as f:np.savez_compressed(f,**saved,state_scale=scale,state_center=center)
    artifact=Path(STORAGE['artifact_root'])/'cycle90/ordinary_control_v1';artifact.mkdir(parents=True,exist_ok=True);ap=artifact/'cal_predictions.npz'
    with ap.open('xb') as f:np.savez_compressed(f,**arrays)
    result=dict(utc=utc(),status='CAL_COMPETENCE_PASS' if all(r['passed'] for r in summaries) else 'CAL_COMPETENCE_FAIL_DO_NOT_OPEN_EVAL',families=summaries,queries=queries,checkpoint_path=str(p),checkpoint_sha256=sha(p),payload_path=str(ap),payload_sha256=sha(ap),protocol_sha256=sha(C/'PROTOCOL90_V1.md'),original_failed_checkpoint_sha256=fit['checkpoint_sha256'],coefficients_unchanged=True,empty_component_rule_train_only=True,eval_accessed=False,full_goal_achieved=False,elapsed_seconds=time.perf_counter()-started)
    write(O/'ORDINARY_CONTROL90_V1.json',result);log(dict(stage='ordinary_control_cal_closed',status=result['status'],families=summaries))

if __name__=='__main__':main()
