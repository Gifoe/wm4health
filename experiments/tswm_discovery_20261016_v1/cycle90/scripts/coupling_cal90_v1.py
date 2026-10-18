from common90_v1 import *
import torch,numpy as np,time
import csdi_core89_v1 as core
from library90_v1 import recognize
from eval_csdi89_v1 import distance
from scipy.special import logsumexp

def main():
    assert not (O/'COUPLING_CAL90_V1.json').exists();started=time.perf_counter();seed=2026901
    screen=read(O/f'CAL_SCREEN90_{seed}_V1.json');library=read(O/'LIBRARY90_V1.json')
    assert screen['status']=='CAL_COMPETENCE_PASS' and library['status']=='CAL_LIBRARY_PASS'
    assert sha(screen['payload_path'])==screen['payload_sha256'] and sha(library['checkpoint_path'])==library['checkpoint_sha256']
    with np.load(screen['payload_path']) as f:independent=f['samples'].copy();banks=f['normalized_oracle_bank'].copy()
    with np.load(library['checkpoint_path']) as f:
        scale=f['scale'].copy();center=f['center'].copy();models=[{k:f[f'f{i}_{k}'].copy() for k in ['beta','feature_scale','var','prior']} for i in [0,1]]
    manifest=read(ROOT/'cycle89/outputs/CSDI_DATA89_V1.json');cal=next(r for r in manifest['files'] if r['split']=='cal')
    with np.load(cal['path']) as f:actions=f['actions'][np.r_[0:32,128:160]].copy();x=f['X'][np.r_[0:32,128:160]].copy()
    fit=read(O/f'FIT90_{seed}_V1.json');assert sha(fit['checkpoint_path'])==fit['checkpoint_sha256']
    model,meta=core.new_model(seed);model.load_state_dict(torch.load(fit['checkpoint_path'],map_location='cuda',weights_only=False)['model'])
    common=[];labels=[];true_labels=[];common_true=[];events=[];truths=[];distances=[];sample_distances=[];sorts=[]
    for q in range(64):
        family=q//32;group=(q%32)//4
        s=core.sample(model,np.repeat(x[q:q+1],128,axis=0),20269070+family*8+group);common.append(s)
        physical=independent[q].transpose(0,2,1)*scale+center
        state=np.concatenate([np.zeros((128,8,2)),physical],axis=1);a=np.repeat(actions[q:q+1],128,axis=0);m=models[family]
        lab=recognize(state,a,m['beta'],m['feature_scale'],scale,m['var'],m['prior']);labels.append(lab);sorts.append(np.argsort(lab,kind='stable'))
        d=distance(independent[q],banks[q]);distances.append(d);sample_distances.append(distance(independent[q],independent[q]));true_labels.append(d.argmin(1));common_true.append(distance(s,banks[q]).argmin(1))
        oracle=banks[q].transpose(0,2,1)*scale+center
        events.append(np.stack([np.max(abs(physical[:,:,0]),axis=1)>.75,physical[:,-1,0]>.25],axis=1).astype(np.float64))
        truths.append(core.PRIOR@np.stack([np.max(abs(oracle[:,:,0]),axis=1)>.75,oracle[:,-1,0]>.25],axis=1))
        if q%8==7:log(dict(stage='raw_trajectory_coupling_pools',query=q,elapsed_seconds=time.perf_counter()-started))
    labels=np.stack(labels);true_labels=np.stack(true_labels);common_true=np.stack(common_true);sorts=np.stack(sorts)
    per_query=[];groups=[];weight_records=[];uniform=np.full(128,1/128.)
    for family in [0,1]:
        for group in range(8):
            anchor_weights={}
            for K in [2,3,4]:
                indices=family*32+group*4+np.arange(K)
                frequency=np.array([np.bincount(labels[q],minlength=9)/128. for q in indices]);prior=models[family]['prior']
                with np.errstate(divide='ignore'):logmass=np.log(frequency).sum(0)-(K-1)*np.log(prior)
                assert np.isfinite(logmass).any(),'No shared class support: explicit coupling failure'
                mass=np.exp(logmass-logsumexp(logmass));weighted_agreement=0.
                for c in np.flatnonzero(mass>0):
                    physicalfreq=np.array([np.bincount(true_labels[q][labels[q]==c],minlength=9)/np.sum(labels[q]==c) for q in indices])
                    weighted_agreement+=mass[c]*np.prod(physicalfreq,axis=0).sum()
                independentfreq=np.array([np.bincount(true_labels[q],minlength=9)/128. for q in indices])
                ecc=np.stack([true_labels[q][sorts[q]] for q in indices]);cn=common_true[indices]
                row=dict(family=family,group=group,K=K,independent_true_bin_agreement=float(np.prod(independentfreq,axis=0).sum()),common_noise_true_bin_agreement=float(np.mean(np.all(cn==cn[:1],axis=0))),rank_reordering_true_bin_agreement=float(np.mean(np.all(ecc==ecc[:1],axis=0))),class_Bayes_reweight_true_bin_agreement=float(weighted_agreement),rank_reordering_empirical_marginals_exactly_preserved=True,true_physical_identity_certified=False)
                groups.append(row)
                for branch,q in enumerate(indices):
                    count=np.bincount(labels[q],minlength=9);w=np.divide(mass[labels[q]],count[labels[q]])
                    assert abs(w.sum()-1)<1e-12
                    d=distances[q];pair=sample_distances[q];base_energy=float(uniform@d@core.PRIOR-.5*uniform@pair@uniform);new_energy=float(w@d@core.PRIOR-.5*w@pair@w)
                    base_risk=uniform@events[q];new_risk=w@events[q];truth=truths[q]
                    per_query.append(dict(family=family,group=group,K=K,branch=branch,query=int(q),base_empirical_energy=base_energy,reweighted_empirical_energy=new_energy,energy_delta=new_energy-base_energy,sample_atom_tv=float(.5*abs(w-uniform).sum()),base_event_probabilities=base_risk.tolist(),reweighted_event_probabilities=new_risk.tolist(),true_event_probabilities=truth.tolist(),base_expected_brier=(base_risk**2-2*base_risk*truth+truth).tolist(),reweighted_expected_brier=(new_risk**2-2*new_risk*truth+truth).tolist()))
                    weight_records.append((family,group,K,int(q),w))
                    if branch==0:anchor_weights[K]=w
                row['anchor_tv_K2_to_K4']=None
            drift=float(.5*abs(anchor_weights[2]-anchor_weights[4]).sum())
            for row in groups[-3:]:row['anchor_tv_K2_to_K4']=drift
    summary=[]
    for family in [0,1]:
        rowset=[r for r in groups if r['family']==family];queries=[r for r in per_query if r['family']==family and r['K']==4]
        summary.append(dict(family=family,by_K=[dict(K=k,**{key:float(np.mean([r[key] for r in rowset if r['K']==k])) for key in ['independent_true_bin_agreement','common_noise_true_bin_agreement','rank_reordering_true_bin_agreement','class_Bayes_reweight_true_bin_agreement','anchor_tv_K2_to_K4']}) for k in [2,3,4]],K4_mean_energy_delta=float(np.mean([r['energy_delta'] for r in queries])),K4_mean_brier_delta=np.mean([np.array(r['reweighted_expected_brier'])-r['base_expected_brier'] for r in queries],axis=0).tolist(),K4_mean_sample_atom_tv=float(np.mean([r['sample_atom_tv'] for r in queries]))))
    dest=Path(STORAGE['artifact_root'])/'cycle90/coupling_cal_v1';dest.mkdir(parents=True,exist_ok=True);p=dest/'pools_labels_weights.npz'
    with p.open('xb') as f:np.savez_compressed(f,independent_pools=independent,common_noise_pools=np.stack(common),reference_banks=banks,learned_labels=labels,reference_nearest_labels=true_labels,common_reference_nearest_labels=common_true,sorts=sorts,weight_keys=np.array([r[:4] for r in weight_records]),weights=np.stack([r[4] for r in weight_records]))
    result=dict(utc=utc(),status='CAL_ONLY_STRUCTURED_INTERPRETER_COUPLING_DIAGNOSIS_COMPLETE',summary=summary,groups=groups,queries=per_query,payload_path=str(p),payload_sha256=sha(p),protocol_sha256=sha(C/'COUPLING_DIAGNOSTIC90_V1.md'),library_protocol_sha256=sha(C/'LIBRARY_CONTROL90_V1.md'),library_receipt_sha256=sha(O/'LIBRARY90_V1.json'),cal_screen_sha256=sha(O/f'CAL_SCREEN90_{seed}_V1.json'),eval_accessed=False,known_polynomial_markov_bias=True,weak_MLP_result_not_replaced=True,native_OneWorld_reproduction=False,energy_metric='conditional empirical law V-statistic for paired pool comparison, not unbiased U-statistic',full_goal_achieved=False,accepted_directions=0,elapsed_seconds=time.perf_counter()-started)
    write(O/'COUPLING_CAL90_V1.json',result);log(dict(stage='raw_trajectory_coupling_closed',summary=summary,elapsed_seconds=result['elapsed_seconds']))

if __name__=='__main__':main()
