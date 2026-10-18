from common90_v1 import *
import torch,numpy as np,time
import csdi_core89_v1 as core
from eval_csdi89_v1 import scores

def main():
    seed=int(sys.argv[1]);assert not (O/f'CAL_SCREEN90_{seed}_V1.json').exists();started=time.perf_counter()
    fit=read(O/f'FIT90_{seed}_V1.json');assert sha(fit['checkpoint_path'])==fit['checkpoint_sha256']
    manifest=read(ROOT/'cycle89/outputs/CSDI_DATA89_V1.json');cal=next(r for r in manifest['files'] if r['split']=='cal');assert sha(cal['path'])==cal['sha256']
    with np.load(cal['path']) as f:actions=f['actions'][np.r_[0:32,128:160]].copy();x=f['X'][np.r_[0:32,128:160]].copy();center=f['center'].copy();scale=f['scale'].copy()
    model,meta=core.new_model(seed);saved=torch.load(fit['checkpoint_path'],map_location='cuda',weights_only=False);model.load_state_dict(saved['model'])
    rows=[];arrays=[];banks=[]
    for q in range(64):
        family=q//32;bank=np.stack([core.simulate(actions[q:q+1],w,family)[0,core.CONTEXT:] for w in range(9)])
        bank=((bank-center)/scale).transpose(0,2,1)
        sample=core.sample(model,np.repeat(x[q:q+1],128,axis=0),20269040+q);arrays.append(sample);banks.append(bank)
        row=dict(query=q,family=family,**scores(sample,bank));rows.append(row)
        if q%8==7:log(dict(stage='full_budget_cal_screen',seed=seed,query=q,elapsed_seconds=time.perf_counter()-started))
    summary=[]
    for family in [0,1]:
        subset=[r for r in rows if r['family']==family];delta=float(np.mean([r['energy_minus_deterministic'] for r in subset]));ratio=float(np.mean([r['off_manifold_ratio'] for r in subset]))
        summary.append(dict(family=family,energy_score=float(np.mean([r['energy_score'] for r in subset])),energy_minus_deterministic=delta,off_manifold_ratio=ratio,passed=delta<0 and ratio<=.5))
    dest=Path(STORAGE['artifact_root'])/f'cycle90/native_seed{seed}_v1';dest.mkdir(parents=True,exist_ok=True);p=dest/'cal_samples.npz'
    with p.open('xb') as f:np.savez_compressed(f,samples=np.stack(arrays),normalized_oracle_bank=np.stack(banks))
    result=dict(utc=utc(),seed=seed,status='CAL_COMPETENCE_PASS' if all(r['passed'] for r in summary) else 'CAL_COMPETENCE_FAIL_DO_NOT_OPEN_EVAL',families=summary,queries=rows,payload_path=str(p),payload_sha256=sha(p),fit_receipt_sha256=sha(O/f'FIT90_{seed}_V1.json'),protocol_sha256=sha(C/'PROTOCOL90_V1.md'),eval_accessed=False,elapsed_seconds=time.perf_counter()-started)
    write(O/f'CAL_SCREEN90_{seed}_V1.json',result);log(dict(stage='full_budget_cal_screen_closed',seed=seed,families=summary,status=result['status']))

if __name__=='__main__':main()
