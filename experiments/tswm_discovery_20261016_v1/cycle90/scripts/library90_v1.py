from common90_v1 import *
import numpy as np,time
import csdi_core89_v1 as core
from persistent_mixture89_v1 import statistics,residual,polynomial

def recognize(states,actions,beta,feature_scale,scale,var,prior):
    future=states[:,core.CONTEXT:];previous=np.concatenate([states[:,core.CONTEXT-1:core.CONTEXT],future[:,:-1]],axis=1)
    phi=polynomial(np.concatenate([previous,actions[:,core.CONTEXT:,None]],-1))/feature_scale
    y=(future-previous)/scale
    predicted=np.einsum('ntf,kfd->nktd',phi,beta,optimize=True)
    sse=np.sum((predicted-y[:,None])**2,axis=(2,3))
    score=np.log(prior)[None]-64*np.log(2*np.pi*var)[None]-sse/(2*var[None])
    assert np.isfinite(score).all()
    return score.argmax(1)

def main():
    assert not (O/'LIBRARY90_V1.json').exists();started=time.perf_counter();manifest=read(ROOT/'cycle89/outputs/CSDI_DATA89_V1.json');paths={r['split']:r for r in manifest['files']}
    data={};permutation=np.random.default_rng(20269061).permutation(9)
    for sp in ['train','cal']:
        assert sha(paths[sp]['path'])==paths[sp]['sha256']
        with np.load(paths[sp]['path']) as f:data[sp]={k:f[k].copy() for k in ['states','actions','world','family','scale','center']}
    arrays={};rows=[];scale=data['train']['scale']
    for family in [0,1]:
        train={k:data['train'][k][data['train']['family']==family] for k in ['states','actions','world']};cal={k:data['cal'][k][data['cal']['family']==family] for k in ['states','actions','world']}
        stats=statistics(train['states'],train['actions'],scale);labels=permutation[train['world']];mass=np.bincount(labels,minlength=9).astype(np.float64);beta=[]
        for c in range(9):
            selected=labels==c
            xx=stats['xx'][selected].sum(0)+np.eye(56)*(1e-6*mass[c]*64)
            xy=stats['xy'][selected].sum(0);beta.append(np.linalg.solve(xx,xy))
        beta=np.stack(beta);sse=residual(stats,beta);var=np.array([max(sse[labels==c,c].sum()/(mass[c]*64*2),1e-7) for c in range(9)]);prior=mass/mass.sum()
        pred=recognize(cal['states'],cal['actions'],beta,stats['feature_scale'],scale,var,prior);target=permutation[cal['world']]
        accuracy=float(np.mean(pred==target));byclass=[float(np.mean(pred[target==c]==c)) if np.any(target==c) else None for c in range(9)];macro=float(np.mean([v for v in byclass if v is not None]))
        rows.append(dict(family=family,accuracy=accuracy,macro_accuracy=macro,per_class_accuracy=byclass,train_class_counts=mass.tolist(),passed=accuracy>=.9 and macro>=.8))
        arrays.update({f'f{family}_beta':beta,f'f{family}_feature_scale':stats['feature_scale'],f'f{family}_var':var,f'f{family}_prior':prior})
    dest=Path(STORAGE['checkpoint_root'])/'cycle90/library_v1';dest.mkdir(parents=True,exist_ok=True);p=dest/'selected.npz'
    with p.open('xb') as f:np.savez_compressed(f,**arrays,scale=scale,center=data['train']['center'],permutation=permutation)
    result=dict(utc=utc(),status='CAL_LIBRARY_PASS' if all(r['passed'] for r in rows) else 'CAL_LIBRARY_FAIL',families=rows,checkpoint_path=str(p),checkpoint_sha256=sha(p),protocol_sha256=sha(C/'LIBRARY_CONTROL90_V1.md'),eval_accessed=False,physical_parameter_values_used=False,known_polynomial_markov_bias=True,mechanism_group_supervision=True,elapsed_seconds=time.perf_counter()-started)
    write(O/'LIBRARY90_V1.json',result);log(dict(stage='ordinary_library_closed',status=result['status'],families=rows,elapsed_seconds=result['elapsed_seconds']))

if __name__=='__main__':main()
