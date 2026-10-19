"""Finite exact distribution + fixed-target native-loss witness, not training."""
import pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'cycle29/scripts'))
import runtime_profile
import numpy as np
import torch
import json,hashlib,types,importlib,time
UP=ROOT/'upstream/worldts_loss';OUT=ROOT/'outputs/compatibility_witness_v1'
SEED=20263201
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def risk(pred,target,reg):
    # pred [2,2], target [40,2], with context order A20 then B20.
    repeated=np.repeat(pred,20,axis=0)
    l1=float(np.abs(repeated-target).mean())
    unitp=repeated/np.maximum(np.linalg.norm(repeated,axis=-1,keepdims=True),1e-8)
    unitt=target/np.maximum(np.linalg.norm(target,axis=-1,keepdims=True),1e-8)
    cosine=float(1-(unitp*unitt).sum(-1).mean())
    def vc(z):
        centered=z-z.mean(0,keepdims=True)
        stdloss=np.maximum(1-np.sqrt((centered**2).mean(0)+1e-4),0).mean()
        cov=centered.T@centered/(len(z)-1)
        off=cov-np.diag(np.diag(cov))
        return float(stdloss+.04*(off**2).sum()/2)
    regloss=.5*(vc(repeated)+vc(target))
    return 10*l1+15*cosine+reg*regloss,dict(mae=l1,cosine=cosine,regularizer=regloss)

def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    start=time.perf_counter();torch.set_num_threads(1)
    manifest=json.loads((UP/'manifest.json').read_text())
    for row in manifest['files']:assert sha(UP/row['path'])==row['sha256']
    # Set only package paths; execute the three already-pinned native modules.
    for package in ['ts_benchmark','ts_benchmark.baselines','ts_benchmark.baselines.worldts','ts_benchmark.baselines.worldts.losses']:
        m=types.ModuleType(package);m.__path__=[str(UP)];sys.modules[package]=m
    native=importlib.import_module('ts_benchmark.baselines.worldts.losses.latent_state_loss').LatentStateLoss
    yvalues=np.array([-1.,0.,1.]);prob=np.array([[.45,.4,.15],[.6,.1,.3]])
    labels=np.concatenate([np.repeat(np.arange(3),[9,8,3]),np.repeat(np.arange(3),[12,2,6])])
    context=np.repeat(np.arange(2),20);y=yvalues[labels]
    M=100.;S=3.;phi=np.array([[M,M],[M+S,M],[M,M+S]])
    z=phi[labels];median=np.full((2,2),M)
    recovered=(z[:,0]+2*z[:,1]-3*M)/S-1
    assert np.array_equal(recovered,y)
    cov=np.cov(z,rowvar=False);eig=np.linalg.eigvalsh(cov)
    std=z.std(0);assert min(std)>1 and min(eig)>0
    rawmed=np.array([0.,-1.])
    bayesrisk=np.array([np.mean(np.abs(y[context==c]-rawmed[c])) for c in range(2)])
    shared_risk=np.array([np.abs(y-a).mean() for a in yvalues])
    assert np.allclose(bayesrisk,[.6,.7]) and abs(shared_risk.min()-.7)<1e-14
    rawbayes=float(bayesrisk.mean());assert abs(rawbayes-.65)<1e-14
    rng=np.random.default_rng(SEED)
    perturb=np.concatenate([rng.normal(size=(256,2,2))*s for s in np.logspace(-5,3,9)])
    candidates=median[None]+perturb
    native_rows=[];sample_rows=[];certificates=[]
    tensors=torch.from_numpy(z[:,None,:])
    for reg in [0.,.1]:
        loss=native(latent_mse_weight=10,latent_alignment_loss='mae',latent_cosine_weight=15,
                    latent_sigreg_weight=reg,latent_detach_target=True,latent_regularizer_type='legacy_vc')
        def nativevalue(p):
            pred=torch.from_numpy(np.repeat(p,20,axis=0)[:,None,:])
            return float(loss(pred,tensors)['loss'])
        baseline,components=risk(median,z,reg)
        base_native=nativevalue(median);assert abs(baseline-base_native)<1e-11
        # Population mixture: L1 margin >= .5*(d_A+d_B). Cosine range <=30.
        # Pred regularizer maximum possible reduction <=reg/2; target fixed.
        # Thus every minimizer lies in total L1 ball radius R below.
        R=(30+reg/2)/.5
        normlower=np.sqrt(2)*M-R
        coslip=7.5/normlower
        # sqrt(v+eps)-sqrt(eps) <= sqrt(v), and paired-group std=|A-B|/2.
        reglip=reg/8
        margin=.5-coslip-reglip
        assert normlower>0 and margin>0
        certificates.append({'reg':reg,'L1_joint_margin':.5,'outside_ball_radius':R,
             'inside_min_norm':float(normlower),'cosine_joint_Lipschitz':float(coslip),
             'regularizer_joint_Lipschitz':reglip,'strict_joint_margin':float(margin),
             'unique_global_minimizer':median.tolist(),
             'assumptions':'fixed phi; balanced two-context exact population batch; H1; original native loss weights'})
        gaps=[];maxerror=abs(base_native-baseline);maxrelative=0.
        for i,p in enumerate(candidates):
            value,_=risk(p,z,reg);gap=value-baseline;distance=np.abs(p-median).sum()
            assert gap>=-1e-10
            if distance<R:assert gap+1e-10>=margin*distance
            gaps.append(gap)
            # Independent native component calls over prespecified logarithmic
            # perturbation subset, including near-kink and large-norm regions.
            if i%36==0:
                nv=nativevalue(p);error=abs(nv-value);maxerror=max(maxerror,error)
                relative=error/max(1.,abs(value));maxrelative=max(maxrelative,relative)
                assert error<2e-10+2e-12*abs(value)
                native_rows.append({'reg':reg,'candidate_index':i,'numpy':value,'native':nv,'abs_error':error,'relative_error':relative})
        sample_rows.append({'reg':reg,'candidates':len(candidates),'minimum_gap':float(min(gaps)),
                           'native_formula_max_abs':maxerror,'native_formula_max_relative':maxrelative,'baseline':baseline,'components':components})
    means=prob@phi
    p0=(means[:,0]-M)/S;p1=(means[:,1]-M)/S;pm1=1-p0-p1
    reconstructed_prob=np.stack([pm1,p0,p1],axis=-1)
    assert np.allclose(reconstructed_prob,prob,atol=1e-14)
    mean_feature_link=yvalues[(np.cumsum(reconstructed_prob,axis=1)>=.5).argmax(1)]
    assert np.array_equal(mean_feature_link,rawmed)
    # Protected direct raw-coordinate median is an established anchor control;
    # it is not included in the coupled cosine/regularizer witness above.
    raw_anchor=np.array([np.median(y[context==c]) for c in range(2)])
    assert np.array_equal(raw_anchor,rawmed)
    report={'status':'passed_fixed_target_compatibility_witness','seed':SEED,'source_revision':manifest['revision'],
      'source_manifest_sha256':sha(UP/'manifest.json'),'script_sha256':sha(__file__),
      'plan_sha256':sha(ROOT/'SURROGATE_COMPATIBILITY_PLAN.md'),
      'identity':'native component transfer + exact finite-distribution diagnostic; no trained WorldTS failure/new method',
      'probabilities':prob.tolist(),'raw_values':yvalues.tolist(),'phi':phi.tolist(),
      'exact_linear_reconstruction_max_abs':float(np.abs(recovered-y).max()),
      'target_std':std.tolist(),'target_covariance_eigenvalues':eig.tolist(),
      'raw_context_medians':rawmed.tolist(),'shared_raw_median':float(yvalues[shared_risk.argmin()]),
      'best_shared_decoder_MAE':float(shared_risk.min()),'context_Bayes_MAE':rawbayes,
      'unavoidable_excess_MAE':float(shared_risk.min()-rawbayes),'certificates':certificates,
      'perturbation_checks':sample_rows,'native_checks':native_rows,
      'ordinary_controls':{'protected_raw_anchor_medians':raw_anchor.tolist(),
         'squared_latent_conditional_means':means.tolist(),'refitted_MAE_link_medians':mean_feature_link.tolist(),
         'Bayes_MAE':rawbayes,'unprotected_anchor_cos_reg_guarantee_claimed':False},
      'limitations':['fixed engineered code, not learned encoder optimum','two contexts/H1 not real benchmark',
         'known calibrated-surrogate theory; not claiming new theorem','no novel method or performance evidence'],
      'accepted_new_directions':0,'seconds':time.perf_counter()-start}
    np.savez_compressed(OUT/'arrays.npz',context=context,y=y,phi_samples=z,phi_table=phi,
                        candidates=candidates,raw_context_medians=rawmed,conditional_means=means)
    report['arrays_sha256']=sha(OUT/'arrays.npz')
    (OUT/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','target_std','raw_context_medians','best_shared_decoder_MAE',
        'context_Bayes_MAE','unavoidable_excess_MAE','perturbation_checks','seconds']}),flush=True)
if __name__=='__main__':main()
