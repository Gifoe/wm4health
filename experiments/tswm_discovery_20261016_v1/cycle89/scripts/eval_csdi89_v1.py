from csdi_core89_v1 import *
import time
from scipy.spatial.distance import cdist

def distance(a,b):
    a=np.asarray(a,np.float64).reshape(len(a),-1);b=np.asarray(b,np.float64).reshape(len(b),-1)
    return cdist(a,b)/np.sqrt(a.shape[1])

def scores(samples,bank):
    dxy=distance(samples,bank);dxx=distance(samples,samples);dyy=distance(bank,bank)
    n=len(samples);energy=float((dxy@PRIOR).mean()-.5*dxx.sum()/(n*(n-1)))
    exact=float(.5*PRIOR@dyy@PRIOR)
    mean=(bank*PRIOR[:,None,None]).sum(0,keepdims=True)
    deterministic=float(distance(mean,bank)[0]@PRIOR)
    spread=float(np.sqrt(np.sum(PRIOR*distance(bank,mean)[:,0]**2)))
    nearest=dxy.argmin(1);freq=np.bincount(nearest,minlength=9)/n
    # Scalar terminal state x; exact prior-weighted CRPS outcome expectation.
    sx=samples[:,0,-1];by=bank[:,0,-1]
    crps=float(np.mean(np.abs(sx[:,None]-by[None,:])@PRIOR)-.5*np.abs(sx[:,None]-sx[None,:]).sum()/(n*(n-1)))
    return dict(energy_score=energy,exact_population_energy_score=exact,deterministic_mean_energy_score=deterministic,energy_minus_deterministic=energy-deterministic,mean_prediction_rms=float(distance(samples.mean(0,keepdims=True),mean)[0,0]),reference_rms_spread=spread,nearest_oracle_rms=float(dxy.min(1).mean()),off_manifold_ratio=float(dxy.min(1).mean()/max(spread,1e-12)),nearest_mechanism_frequency=freq.tolist(),nearest_mechanism_tv=float(.5*abs(freq-PRIOR).sum()),terminal_x_crps=crps)

def bootstrap_group_mean(values,seed):
    v=np.asarray(values).reshape(8,4).mean(1);rng=np.random.default_rng(seed)
    means=v[rng.integers(0,8,(10000,8))].mean(1)
    return dict(mean=float(v.mean()),low=float(np.quantile(means,.025)),high=float(np.quantile(means,.975)),independent_query_groups=8)

def main():
    started=time.perf_counter();assert not (O/'CSDI_EVAL89_V1.json').exists()
    fit=read(O/'CSDI_PILOT89_V1.json');assert sha(fit['checkpoint_path'])==fit['checkpoint_sha256']
    data=read(O/'CSDI_DATA89_V1.json');path=next(r for r in data['files'] if r['split']=='eval_queries');assert sha(path['path'])==path['sha256']
    with np.load(path['path']) as f:
        x=f['X'].copy();bank=f['bank'][:,:,CONTEXT:].copy();families=f['family'].copy();scale=f['scale'].copy();center=f['center'].copy()
    bank=((bank-center)/scale).transpose(0,1,3,2)
    model,meta=new_model(20268913);saved=torch.load(fit['checkpoint_path'],map_location='cuda',weights_only=False);model.load_state_dict(saved['model']);model.eval()
    draws=[];rows=[];rng=np.random.default_rng(20268916)
    for q in range(len(x)):
        tiled=np.repeat(x[q:q+1],128,axis=0)
        predictions=sample(model,tiled,20268917+q,batch=64);draws.append(predictions)
        s=scores(predictions,bank[q]);oracle=scores(bank[q,rng.choice(9,128,p=PRIOR)],bank[q])
        row=dict(query=q,family=int(families[q]),group=int((q%32)//4),learned=s,matched_budget_exact_sampler=oracle)
        rows.append(row)
        log(dict(stage='csdi_eval_query',query=q,family=row['family'],energy=s['energy_score'],delta_to_deterministic=s['energy_minus_deterministic'],off_manifold_ratio=s['off_manifold_ratio'],elapsed_seconds=time.perf_counter()-started))
    grouped=[]
    for family in [0,1]:
        subset=[r for r in rows if r['family']==family]
        metrics={k:float(np.mean([r['learned'][k] for r in subset])) for k in ['energy_score','exact_population_energy_score','deterministic_mean_energy_score','energy_minus_deterministic','mean_prediction_rms','reference_rms_spread','nearest_oracle_rms','off_manifold_ratio','nearest_mechanism_tv','terminal_x_crps']}
        metrics['energy_delta_paired_group_bootstrap']=bootstrap_group_mean([r['learned']['energy_minus_deterministic'] for r in subset],20268918)
        metrics['matched_budget_exact_sampler_energy']=float(np.mean([r['matched_budget_exact_sampler']['energy_score'] for r in subset]))
        passed=metrics['energy_minus_deterministic']<0 and metrics['off_manifold_ratio']<=.5
        grouped.append(dict(family=family,competence_screen_passed=passed,**metrics))
    payload=Path(STORAGE['artifact_root'])/'cycle89/csdi_v1';payload.mkdir(parents=True,exist_ok=True);p=payload/'heldout_native_samples.npz'
    with p.open('xb') as f:np.savez_compressed(f,samples=np.stack(draws),normalized_oracle_bank=bank,family=families)
    qualified=all(r['competence_screen_passed'] for r in grouped)
    receipt=dict(utc=utc(),status='GENERATOR_QUALIFIED_FOR_NEXT_COUPLING_TEST' if qualified else 'GENERATOR_NOT_QUALIFIED_DO_NOT_ATTRIBUTE_COUPLING_FAILURE',family_summary=grouped,queries=rows,samples_per_query=128,payload_path=str(p),payload_sha256=sha(p),checkpoint_sha256=sha(fit['checkpoint_path']),fit_receipt_sha256=sha(O/'CSDI_PILOT89_V1.json'),eval_data_sha256=path['sha256'],source_files_unchanged=all(sha(Path(meta['root'])/p)==h for p,h in meta['sha256'].items()),protocol_sha256=sha(C/'CSDI_COUPLING_PROTOCOL89_V1.md'),competence_addendum_sha256=sha(C/'CSDI_COMPETENCE_ADDENDUM89_V1.md'),elapsed_seconds=time.perf_counter()-started,optimizer_seeds=1,conditional_sample_joint_reweighting_executed=False,accepted_research_directions=0)
    write(O/'CSDI_EVAL89_V1.json',receipt);log(dict(stage='csdi_evaluation_closed',status=receipt['status'],families=grouped,elapsed_seconds=receipt['elapsed_seconds']))

if __name__=='__main__':main()
