from common90_v1 import *
import numpy as np,torch,time
from verify_csdi89_v1 import replay

def main():
    assert not (O/'VERIFICATION90_V1.json').exists();started=time.perf_counter();seed=2026901
    fit=read(O/f'FIT90_{seed}_V1.json');screen=read(O/f'CAL_SCREEN90_{seed}_V1.json');control=read(O/'ORDINARY_CONTROL90_V1.json');library=read(O/'LIBRARY90_V1.json');classifier=read(O/'CLASSIFIER90_V2.json')
    for receipt in [fit,control,library,classifier]:assert sha(receipt['checkpoint_path'])==receipt['checkpoint_sha256']
    for receipt in [screen,control]:assert sha(receipt['payload_path'])==receipt['payload_sha256']
    selected=min(fit['trace'],key=lambda r:r['cal_denoising_mse']);assert selected['step']==fit['selected_step']
    cp=torch.load(fit['checkpoint_path'],map_location='cpu',weights_only=False);assert cp['step']==fit['selected_step']
    for name,h in fit['source']['sha256'].items():assert sha(Path(fit['source']['root'])/name)==h
    assert fit['updates']==20000 and fit['eval_accessed'] is False and screen['eval_accessed'] is False
    with np.load(screen['payload_path']) as f:samples=f['samples'];banks=f['normalized_oracle_bank']
    assert samples.shape==(64,128,2,64) and banks.shape==(64,9,2,64)
    assert np.isfinite(samples).all() and np.isfinite(banks).all()
    maxerr=0.;count=0;prior=np.array([.4,.2,.1,.08,.07,.05,.04,.035,.025])
    for i,row in enumerate(screen['queries']):
        for key,value in replay(samples[i],banks[i],prior).items():maxerr=max(maxerr,abs(value-row[key]));count+=1
    assert maxerr<1e-7,maxerr
    for summary in screen['families']:
        rows=[r for r in screen['queries'] if r['family']==summary['family']]
        ratio=float(np.mean([r['off_manifold_ratio'] for r in rows]));delta=float(np.mean([r['energy_minus_deterministic'] for r in rows]))
        assert abs(ratio-summary['off_manifold_ratio'])<1e-12 and abs(delta-summary['energy_minus_deterministic'])<1e-12
        assert summary['passed']==(delta<0 and ratio<=.5)
    oldfit=read(ROOT/'cycle89/outputs/PERSISTENT_MIXTURE_FIT89_V1.json')
    with np.load(oldfit['checkpoint_path']) as old,np.load(control['checkpoint_path']) as new:
        for family in [0,1]:
            keep=old[f'f{family}_pi']*4096>=1;assert np.array_equal(new[f'f{family}_original_indices'],np.flatnonzero(keep))
            assert np.array_equal(new[f'f{family}_beta'],old[f'f{family}_beta'][keep])
            expected=old[f'f{family}_pi'][keep];expected/=expected.sum();assert np.array_equal(new[f'f{family}_pi'],expected)
            assert abs(old[f'f{family}_pi'][~keep].sum()-control['families'][family]['removed_probability'])<1e-15
    assert classifier['selected_step']==min(classifier['trace'],key=lambda r:r['cross_entropy'])['step']
    result=dict(utc=utc(),status='NATIVE_FULL_BUDGET_CAL_AND_CONTROL_PROVENANCE_REPLAYED',metric_values_replayed=count,independent_gram_distance_max_abs=maxerr,native_source_unchanged=True,checkpoint_selection_replayed=True,ordinary_repair_coefficients_exactly_unchanged=True,ordinary_repair_support_train_only=True,failed_MLP_selection_replayed=True,all_checked_payload_hashes_verified=True,first_seed_cal_status=screen['status'],full_goal_achieved=False,accepted_directions=0,eval_accessed=False,elapsed_seconds=time.perf_counter()-started)
    write(O/'VERIFICATION90_V1.json',result);log(dict(stage='c90_bounded_verification',**{k:v for k,v in result.items() if k!='utc'}))

if __name__=='__main__':main()
