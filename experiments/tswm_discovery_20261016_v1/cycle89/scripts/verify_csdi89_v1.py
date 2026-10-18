from common89_v1 import *
import numpy as np,torch,time

def independent_dist(a,b):
    a=np.asarray(a,np.float64).reshape(len(a),-1);b=np.asarray(b,np.float64).reshape(len(b),-1)
    square=np.sum(a*a,1)[:,None]+np.sum(b*b,1)[None,:]-2*a@b.T
    return np.sqrt(np.maximum(square,0.)/a.shape[1])

def replay(x,bank,pi):
    ab=independent_dist(x,bank);aa=independent_dist(x,x);bb=independent_dist(bank,bank);np.fill_diagonal(aa,0.);np.fill_diagonal(bb,0.)
    n=len(x);mean=np.einsum('w,wct->ct',pi,bank)[None]
    deterministic=float((independent_dist(mean,bank)*pi).sum())
    spread=float(np.sqrt((pi*independent_dist(bank,mean)[:,0]**2).sum()))
    energy=float((ab*pi).sum()/n-aa.sum()/(2*n*(n-1)))
    sx=x[:,0,-1];by=bank[:,0,-1]
    return dict(energy_score=energy,exact_population_energy_score=float(.5*np.einsum('i,ij,j->',pi,bb,pi)),deterministic_mean_energy_score=deterministic,energy_minus_deterministic=energy-deterministic,reference_rms_spread=spread,nearest_oracle_rms=float(ab.min(1).mean()),off_manifold_ratio=float(ab.min(1).mean()/max(spread,1e-12)),terminal_x_crps=float(np.mean(np.sum(np.abs(sx[:,None]-by[None])*pi,1))-.5*np.sum(abs(sx[:,None]-sx[None]))/(n*(n-1))))

def main():
    started=time.perf_counter();assert not (O/'VERIFICATION89_V1.json').exists()
    ev=read(O/'CSDI_EVAL89_V1.json');fit=read(O/'CSDI_PILOT89_V1.json');data=read(O/'CSDI_DATA89_V1.json');pi=np.array([.4,.2,.1,.08,.07,.05,.04,.035,.025])
    for f in data['files']:assert sha(f['path'])==f['sha256']
    assert sha(ev['payload_path'])==ev['payload_sha256'];assert sha(fit['checkpoint_path'])==fit['checkpoint_sha256']
    cp=torch.load(fit['checkpoint_path'],map_location='cpu',weights_only=False)
    assert fit['selected_step']==cp['step']==min(fit['trace'],key=lambda r:r['cal_denoising_mse'])['step']
    assert fit['eval_queries_accessed'] is False
    for name,h in fit['source']['sha256'].items():assert sha(Path(fit['source']['root'])/name)==h
    with np.load(ev['payload_path']) as f:x=f['samples'];banks=f['normalized_oracle_bank'];families=f['family']
    assert x.shape==(64,128,2,64) and banks.shape==(64,9,2,64)
    assert np.isfinite(x).all() and np.isfinite(banks).all()
    maxerr=0.;count=0
    for i,row in enumerate(ev['queries']):
        assert row['query']==i and row['family']==families[i]
        replayed=replay(x[i],banks[i],pi)
        for k,v in replayed.items():maxerr=max(maxerr,abs(v-row['learned'][k]));count+=1
    assert maxerr<1e-7,maxerr
    for summary in ev['family_summary']:
        rows=[r for r in ev['queries'] if r['family']==summary['family']]
        for key in ['energy_score','energy_minus_deterministic','off_manifold_ratio']:
            assert abs(np.mean([r['learned'][key] for r in rows])-summary[key])<1e-12
        passed=summary['energy_minus_deterministic']<0 and summary['off_manifold_ratio']<=.5
        assert passed==summary['competence_screen_passed']
    # Exact two-point law: mean forecaster score1, population mixture score.5;
    # the expectation of the sample U-statistic over all four ordered N2 draws is.5.
    support=np.array([-1.,1.]);expected=[]
    for a in support:
        for b in support:expected.append(np.mean(abs(np.array([a,b])[:,None]-support[None]))-.5*abs(a-b))
    assert abs(np.mean(expected)-.5)<1e-12
    ipf=read(O/'MARGINAL_CONTROL89_V1.json')
    assert len(ipf['rows'])==27 and len(ipf['checkpoint_replay'])==9
    assert all(r['population_bce_max_abs']==0 for r in ipf['checkpoint_replay'])
    assert sum(r['converged'] for r in ipf['rows'])==9
    receipt=dict(utc=utc(),status='RAW_PAYLOAD_METRICS_AND_CHECKPOINT_SELECTION_REPLAYED',metric_values_recomputed=count,independent_gram_vs_cdist_max_abs=maxerr,shape=list(x.shape),native_source_hashes_unchanged=True,all_data_and_checkpoint_hashes_verified=True,metric_two_point_analytic_control_passed=True,selected_without_eval=True,ipf_rows=27,ipf_strictly_converged_rows=9,ipf_all_converged=False,eval_receipt_sha256=sha(O/'CSDI_EVAL89_V1.json'),fit_receipt_sha256=sha(O/'CSDI_PILOT89_V1.json'),elapsed_seconds=time.perf_counter()-started,novelty_verified=False,full_goal_achieved=False)
    write(O/'VERIFICATION89_V1.json',receipt);log(dict(stage='csdi_metric_replay_closed',**{k:v for k,v in receipt.items() if k!='utc'}))

if __name__=='__main__':main()
