"""Read-only fixed-horizon Gaussian marginal audit of the same16 frozen fits."""
from common54 import *
import csv, pickle
from scipy.stats import chi2
from evaluate54_v2 import components, propagate
from gaussian_belief_math54_v1 import scores

def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))

def main():
    source_check()
    reg=read(OUT/'GAUSSIAN_BELIEF_REGISTRATION54_V1.json')
    assert reg['heldout_opened_at_registration'] is False
    for rel,value in reg['files_sha256'].items():assert sha(ROOT/rel)==value
    assert read(OUT/'postprocess54_v3.json')['status']=='COMPLETE_PRIMARY12_AND_FOUR_MATCHED_EXISTING_EM_OPTION_CONTROLS'
    risk_path=OUT/'causal_risk_interpretation_v1/summary.json';risk=read(risk_path)
    assert risk['status']=='COMPLETE_READ_ONLY_CAUSAL_RISK_INTERPRETATION_OF_ALL16_FROZEN_FITS'
    main_path=OUT/'evaluation54.json';extra_path=OUT/'em_option_learning_v1/evaluation/summary.json'
    primary=read(main_path);extra=read(extra_path)
    assert extra['all_16_models_and_16_readouts_frozen_before_first_heldout_score'] is True
    scored=primary['records']+extra['records'];assert len(scored)==16
    seals=[OUT/'SELECTION_SEAL54_V1.json',OUT/'em_option_learning_v1/SELECTION_SEAL54_EM_TRUE_V1.json']
    maps=[OUT/'READOUT_SEAL54_V1.json',OUT/'em_option_learning_v1/evaluation/READOUT_SEAL_EM_TRUE54_V1.json']
    fits={r['run']:r for p in seals for r in read(p)['fits']}
    readouts={r['run']:r for p in maps for r in read(p)['readouts']}
    assert len(fits)==len(readouts)==16
    before={};provenance=[]
    def track(p,logical):
        p=pathlib.Path(p);value=sha(p)
        if p in before:assert before[p]==value
        else:before[p]=value;provenance.append({'input':logical,'sha256':value})
        return value
    preflight_path=OUT/'preflight54.json';preflight=read(preflight_path)
    for p in [main_path,extra_path,risk_path,*seals,*maps,OUT/'REPORT_MANIFEST54_V2.json',preflight_path]:
        track(p,p.relative_to(ROOT).as_posix())
    for run,f in fits.items():assert track(f['checkpoint'],'checkpoint:'+run)==f['checkpoint_sha256']
    for run,r in readouts.items():assert track(r['file'],'training_readout:'+run)==r['sha256']
    target=OUT/'gaussian_belief_audit_v1';assert not target.exists();target.mkdir()
    raw_root=ART/'gaussian_belief_audit_v1';assert not raw_root.exists();raw_root.mkdir()
    rows=[];pairs=[];raw=[];checks=[];max_mean_error=0.;by_run={}
    for task in PLAN['tasks']:
        name=task['name'];K=task['latent_dim'];hs=PLAN['horizons'];cutoff=float(chi2.ppf(.95,K))
        data_file=data_path(name)
        assert track(data_file,'original_dataset:'+name)==next(d['sha256'] for d in preflight['datasets'] if d['task']==name)
        with np.load(data_file) as d:hx=d['heldout_x'][:,:PLAN['prefix_length']]
        oracle_path=ART/(name+'_oracle.npz')
        assert track(oracle_path,'scored_oracle:'+name)==risk['input_hashes_unchanged'][str(oracle_path)]
        with np.load(oracle_path) as o:om=o['state_mean'];oc=o['state_cov'];truth=o['state_truth']
        assert om.shape==truth.shape==(50,len(hs),K) and oc.shape==(50,len(hs),K,K)
        for r in [r for r in scored if r['task']==name]:
            run=r['run'];f=fits[run];rd=readouts[run]
            with np.load(rd['file']) as d:W=d['Wz'];b=d['bz']
            params=pickle.loads(pathlib.Path(f['checkpoint']).read_bytes())
            c=components('em' if r['method']=='em' else 'rp',name,r['seed'],params,hx)
            lm,lc=propagate(c['smooth_mean'][:,-1],c['smooth_cov'][:,-1],c['A'],c['b'],c['Q'],max(hs))
            mean=lm[:,hs]@W+b;cov=W.T@lc[:,hs]@W
            assert track(r['artifact'],'original_scored_predictions:'+run)==r['artifact_sha256']
            with np.load(r['artifact']) as d:old_mean=d['state_mean']
            error=float(np.max(abs(mean-old_mean)));assert error<1e-7
            max_mean_error=max(max_mean_error,error)
            values=scores(mean,cov,om,oc,truth,cutoff);by_run[run]=values
            p=raw_root/(run+'.npz');assert not p.exists()
            np.savez_compressed(p,state_mean=mean,state_cov=cov,**values)
            raw.append({'run':run,'storage_relative_path':'cycle54/gaussian_belief_audit_v1/'+p.name,'sha256':sha(p)})
            condition=float(np.linalg.cond(W))
            checks.append({'run':run,'native_gaussian_check':c['check'],'scored_mean_max_abs_error':error,'state_map_rank':int(np.linalg.matrix_rank(W)),'state_map_condition':condition if np.isfinite(condition) else None})
            for j,h in enumerate(hs):
                valid=values['valid_covariance'][:,j];n=int(valid.sum())
                row={'run':run,'task':name,'method':r['method'],'seed':r['seed'],'h':h,'sequences':50,'valid_covariances':n,'minimum_covariance_eigenvalue':float(values['minimum_covariance_eigenvalue'][:,j].min()),'ellipsoid95_cutoff':cutoff}
                for key in ['gaussian_kl','mean_kl','covariance_kl','learned_nll','oracle_nll','realized_nll_excess','ellipsoid_covered','mahalanobis_squared']:
                    row[key]=float(values[key][:,j].mean()) if n==50 else None
                rows.append(row)
            print(json.dumps({'run':run,'all_covariances_positive':bool(np.all(values['valid_covariance'])),'h10_gaussian_kl':next(q['gaussian_kl'] for q in rows if q['run']==run and q['h']==10)}),flush=True)
        bp=next(x for x in risk['raw_bootstrap_indices'] if pathlib.Path(x['path']).name==name+'_bootstrap_indices.npy')
        assert track(bp['path'],'registered_sequence_resampling:'+name)==bp['sha256']
        ids=np.load(bp['path']);assert ids.shape==(2048,50)
        for r in [r for r in scored if r['task']==name and r['method']=='rp_emtrue']:
            left=by_run[r['run']];right=by_run[r['matched_false_run']]
            pair={'task':name,'seed':r['seed'],'true_run':r['run'],'false_run':r['matched_false_run'],'horizons':[]}
            for j,h in enumerate(hs):
                valid=left['valid_covariance'][:,j]&right['valid_covariance'][:,j]
                delta=left['gaussian_kl'][:,j]-right['gaussian_kl'][:,j]
                good=bool(np.all(valid));boots=delta[ids].mean(1) if good else None
                pair['horizons'].append({'h':h,'all_50_covariances_valid':good,'gaussian_kl_delta_true_minus_false':float(delta.mean()) if good else None,'pointwise95_sequence_interval':np.percentile(boots,[2.5,97.5]).tolist() if good else None})
            pairs.append(pair)
    assert len(rows)==96 and len(pairs)==4
    for p,value in before.items():assert sha(p)==value,'Existing source result or selected model changed'
    with (target/'gaussian_scores.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'utc':utc(),'status':'COMPLETE_ALL16_FIXED_HORIZON_GAUSSIAN_BELIEF_AUDIT','records':rows,'matched_pairs':pairs,'native_checks':checks,'max_existing_mean_recomputation_error':max_mean_error,'raw_scores':raw,'input_hashes_unchanged':provenance,'additional_optimizer_updates':0,'model_or_hyperparameter_selection':False,'covariance_recalibration':False,'units':'KL and NLL are nats per full latent-state vector, not per dimension','interval_scope':'Reuse registered2048 complete-sequence bootstrap indices, pointwise95 percentiles, conditional on fixed fits; no multiplicity correction or initialization/domain uncertainty.','scope':'Diagnostic mapped Gaussian surrogate marginals at six horizons; not proper-RPM inference, joint-path calibration, state-label-free deployment or parameter uncertainty.','protocol_sha256':sha(ROOT/'GAUSSIAN_BELIEF_AUDIT54_V1.md'),'registration_sha256':sha(OUT/'GAUSSIAN_BELIEF_REGISTRATION54_V1.json'),'script_sha256':sha(__file__),'accepted_final_directions':0}
    dump(target/'summary.json',summary)
    text='# C54 fixed-horizon Gaussian-belief audit\n\nThe same16 frozen models and existing training-only maps were used. No extra fit, calibration or selection was performed. All native causal-prefix means were checked against the previously scored predictions.\n\n'
    text+='| Task | Method | Seed | h | Gaussian KL | Mean KL | Covariance KL | Realized NLL excess | 95% ellipsoid coverage | Valid covariances |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    def fmt(x):return 'undefined' if x is None else f'{x:.6g}'
    for r in rows:text+=f"|{r['task']}|{r['method']}|{r['seed']}|{r['h']}|{fmt(r['gaussian_kl'])}|{fmt(r['mean_kl'])}|{fmt(r['covariance_kl'])}|{fmt(r['realized_nll_excess'])}|{fmt(r['ellipsoid_covered'])}|{r['valid_covariances']}/50|\n"
    text+='\nKL is oracle-to-model conditional Gaussian KL, in nats per state vector. Its mean and covariance components separate location from uncertainty mismatch. Realized NLL excess can be negative on finite sequences without surpassing the Bayes oracle. Coverage is descriptive over50 dependent-horizon sequences, not proof of calibration or significance. Undefined covariances are reported rather than repaired or removed. All six horizons and every initialization are retained.\n\nThe native reported RP covariance includes the documented1e-5 I term. Affine state maps use training latent labels and remain diagnostics. The Gaussian auxiliary posterior is not asserted to equal the proper normalized RPM posterior. This audit does not certify joint-path laws, uncertainty in fitted parameters, deployment without latent labels or a new world-model research gap. All prior scored artifacts, checkpoints and maps were hash-checked unchanged. Exact four-pair KL deltas, pointwise sequence intervals, native arithmetic checks and raw-score hashes are in outputs/gaussian_belief_audit_v1/summary.json.\n'
    report=ROOT/'GAUSSIAN_BELIEF_REPORT54_V1.md';assert not report.exists();report.write_text(text,encoding='utf-8')
    print(json.dumps({'status':summary['status'],'rows':96,'pairs':4,'report_sha256':sha(report),'accepted_final_directions':0}),flush=True)

if __name__=='__main__':main()
