from persistent_mixture89_v1 import *
from eval_csdi89_v1 import scores,distance,bootstrap_group_mean

def weighted_score(samples,weights,bank):
    ab=distance(samples,bank);aa=distance(samples,samples);bb=distance(bank,bank)
    mean=(bank*PRIOR[:,None,None]).sum(0,keepdims=True);spread=float(np.sqrt(np.sum(PRIOR*distance(bank,mean)[:,0]**2)))
    energy=float(weights@ab@PRIOR-.5*weights@aa@weights);deterministic=float(distance(mean,bank)[0]@PRIOR)
    nearest=ab.argmin(1);freq=np.bincount(nearest,weights=weights,minlength=9)
    return dict(energy_score=energy,exact_population_energy_score=float(.5*PRIOR@bb@PRIOR),deterministic_mean_energy_score=deterministic,energy_minus_deterministic=energy-deterministic,nearest_oracle_rms=float(weights@ab.min(1)),off_manifold_ratio=float(weights@ab.min(1)/max(spread,1e-12)),nearest_mechanism_frequency=freq.tolist(),nearest_mechanism_tv=float(.5*abs(freq-PRIOR).sum()),component_nearest_mechanism=nearest.tolist())

def main():
    started=time.perf_counter();assert not (O/'PERSISTENT_MIXTURE_EVAL89_V1.json').exists()
    fit=read(O/'PERSISTENT_MIXTURE_FIT89_V1.json');assert sha(fit['checkpoint_path'])==fit['checkpoint_sha256']
    with np.load(fit['checkpoint_path']) as f:
        scale=f['state_scale'].copy();center=f['state_center'].copy()
        models=[{k:f[f'f{i}_{k}'].copy() for k in ['beta','pi','var','feature_scale']} for i in [0,1]]
    rng=np.random.default_rng(20268923);all_actions=[];all_banks=[];all_x=[];families=[];predicted=[]
    for family in [0,1]:
        a=draw_actions(rng,32);bank=np.stack([simulate(a,w,family) for w in range(9)],1)
        try:pred=rollout(models[family],a,scale)
        except (FloatingPointError,OverflowError) as exc:
            write(O/'PERSISTENT_MIXTURE_EVAL89_V1.json',dict(utc=utc(),status='ORDINARY_BASELINE_ROLLOUT_FAILURE_NOT_EVIDENCE_OF_RESEARCH_GAP',failure_family=family,error=str(exc),fit_receipt_sha256=sha(O/'PERSISTENT_MIXTURE_FIT89_V1.json'),accepted_research_directions=0))
            raise
        all_actions.append(a);all_banks.append(bank);predicted.append(pred);families.extend([family]*32)
        all_x.append(features(np.zeros((32,LENGTH,2)),a,family,center,scale))
    bank=np.concatenate(all_banks);pred=np.concatenate(predicted);a=np.concatenate(all_actions);x=np.concatenate(all_x);families=np.array(families)
    dp=Path(STORAGE['data_root'])/'cycle89/persistent_mixture_v1';dp.mkdir(parents=True,exist_ok=True);datafile=dp/'fresh_eval_queries.npz'
    with datafile.open('xb') as f:np.savez_compressed(f,bank=bank,actions=a,X=x,family=families,center=center,scale=scale,prior=PRIOR)
    banknorm=((bank[:,:,CONTEXT:]-center)/scale).transpose(0,1,3,2);prednorm=((pred[:,:,CONTEXT:]-center)/scale).transpose(0,1,3,2)
    csdi=read(O/'CSDI_PILOT89_V1.json');assert sha(csdi['checkpoint_path'])==csdi['checkpoint_sha256']
    model,meta=new_model(20268913);cp=torch.load(csdi['checkpoint_path'],map_location='cuda',weights_only=False);model.load_state_dict(cp['model'])
    rows=[];draws=[]
    for q in range(64):
        family=int(families[q]);p=weighted_score(prednorm[q],models[family]['pi'],banknorm[q])
        samples=sample(model,np.repeat(x[q:q+1],128,axis=0),20268925+q);draws.append(samples)
        c=scores(samples,banknorm[q]);rows.append(dict(query=q,family=family,group=(q%32)//4,ordinary_mixture=p,frozen_csdi=c))
        if q%8==7:log(dict(stage='persistent_mixture_fresh_eval',query=q,elapsed_seconds=time.perf_counter()-started))
    summary=[];coupling=[]
    for family in [0,1]:
        selected=[r for r in rows if r['family']==family];arms={}
        for arm in ['ordinary_mixture','frozen_csdi']:
            metrics={k:float(np.mean([r[arm][k] for r in selected])) for k in ['energy_score','exact_population_energy_score','deterministic_mean_energy_score','energy_minus_deterministic','off_manifold_ratio','nearest_mechanism_tv']}
            metrics['competence_screen_passed']=metrics['energy_minus_deterministic']<0 and metrics['off_manifold_ratio']<=.5
            metrics['energy_delta_group_bootstrap']=bootstrap_group_mean([r[arm]['energy_minus_deterministic'] for r in selected],20268918)
            arms[arm]=metrics
        summary.append(dict(family=family,**arms))
        for group in range(8):
            group_rows=[r for r in selected if r['group']==group];pi=models[family]['pi']
            for K in [2,3,4]:
                labels=np.array([r['ordinary_mixture']['component_nearest_mechanism'] for r in group_rows[:K]])
                shared=float(pi@np.all(labels==labels[:1],axis=0))
                freqs=np.array([r['ordinary_mixture']['nearest_mechanism_frequency'] for r in group_rows[:K]])
                independent=float(np.prod(freqs,axis=0).sum())
                coupling.append(dict(family=family,group=group,K=K,shared_component_nearest_mechanism_agreement=shared,independent_component_nearest_mechanism_agreement=independent,learned_single_query_marginal_change=0.,property='shared component preserves the complete learned trajectory marginal by construction',true_shared_physics_certified=False))
    ap=Path(STORAGE['artifact_root'])/'cycle89/persistent_mixture_v1';ap.mkdir(parents=True,exist_ok=True);payload=ap/'fresh_forecasts.npz'
    with payload.open('xb') as f:np.savez_compressed(f,ordinary_component_trajectories=prednorm,csdi_samples=np.stack(draws),normalized_oracle_bank=banknorm,family=families,mixture_pi=np.stack([m['pi'] for m in models]))
    receipt=dict(utc=utc(),status='FRESH_QUERY_ORDINARY_PERSISTENT_LATENT_CONTROL_COMPLETE',family_summary=summary,queries=rows,coupling=coupling,payload_path=str(payload),payload_sha256=sha(payload),fresh_data_path=str(datafile),fresh_data_sha256=sha(datafile),fit_receipt_sha256=sha(O/'PERSISTENT_MIXTURE_FIT89_V1.json'),csdi_checkpoint_sha256=csdi['checkpoint_sha256'],protocol_sha256=sha(C/'PERSISTENT_MIXTURE_CONTROL89_V1.md'),fresh_query_seed=20268923,hidden_mechanism_labels_read_by_fit=False,calibration_selection_only=True,optimizer_seeds_per_model=1,known_polynomial_markov_bias=True,new_method_claim=False,accepted_research_directions=0,elapsed_seconds=time.perf_counter()-started)
    write(O/'PERSISTENT_MIXTURE_EVAL89_V1.json',receipt);log(dict(stage='persistent_mixture_fresh_eval_closed',families=summary,elapsed_seconds=receipt['elapsed_seconds']))

if __name__=='__main__':main()
