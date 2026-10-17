"""Read-only, sealed-result Bayes-risk interpretation; never fits/selects a model."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import csv,ctypes,datetime,hashlib,json,pathlib
import numpy as np
from risk_math54_v1 import decompose,paired_interval,ratio
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stamp(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))

def main():
    registration=read(OUT/'CAUSAL_RISK_REGISTRATION54_V1.json');assert registration['heldout_opened_at_registration'] is False
    for name,value in registration['files_sha256'].items():assert sha(ROOT/name)==value,name
    assert read(OUT/'postprocess54_v3.json')['status']=='COMPLETE_PRIMARY12_AND_FOUR_MATCHED_EXISTING_EM_OPTION_CONTROLS'
    main=read(OUT/'evaluation54.json');extra=read(OUT/'em_option_learning_v1/evaluation/summary.json');assert len(main['records'])==12 and len(extra['records'])==4
    assert extra['all_16_models_and_16_readouts_frozen_before_first_heldout_score'] is True
    f1=read(OUT/'SELECTION_SEAL54_V1.json');f2=read(OUT/'em_option_learning_v1/SELECTION_SEAL54_EM_TRUE_V1.json')
    r1=read(OUT/'READOUT_SEAL54_V1.json');r2=read(OUT/'em_option_learning_v1/evaluation/READOUT_SEAL_EM_TRUE54_V1.json')
    assert max(stamp(f1['utc']),stamp(f2['utc']))<=stamp(r2['utc'])<=stamp(r1['utc'])<=stamp(main['utc'])<=stamp(extra['utc'])
    inputs={};paths=[OUT/'evaluation54.json',OUT/'em_option_learning_v1/evaluation/summary.json',OUT/'REPORT_MANIFEST54_V2.json']
    for f in f1['fits']+f2['fits']:assert sha(f['checkpoint'])==f['checkpoint_sha256']
    for r in r1['readouts']+r2['readouts']:assert sha(r['file'])==r['sha256']
    for p in paths:inputs[str(p)]=sha(p)
    storage=read(ROOT.parent/'STORAGE_LOCATION_V1.json');plan=read(ROOT/'plan54_v1.json');hs=plan['horizons'];prefix=plan['prefix_length']
    artifact_root=pathlib.Path(storage['artifact_root'])/'cycle54';data_root=pathlib.Path(storage['data_root'])/'cycle54'
    target=OUT/'causal_risk_interpretation_v1';assert not target.exists();target.mkdir()
    rng=np.random.default_rng(5491);records=[];comparisons=[];oracle_energy=[];trace={};max_identity=0.;max_saved=0.;raw_artifacts=[]
    raw_output=artifact_root/'causal_risk_interpretation_v1';raw_output.mkdir(exist_ok=True)
    for task in plan['tasks']:
        name=task['name'];op=artifact_root/(name+'_oracle.npz');inputs[str(op)]=sha(op)
        with np.load(op) as a:mu=a['state_mean'];truth=a['state_truth'];noise_recorded=a['state_sequence_mse']
        assert mu.shape==truth.shape and mu.shape[:2]==(50,len(hs));N,H,K=mu.shape
        with np.load(data_root/(name+'.npz')) as a:A=a['true_A'];b=a['true_b'];m=a['true_m1'].copy()
        priors=[]
        for t in range(prefix+max(hs)):
            if t:m=A@m+b
            if t in [prefix-1+h for h in hs]:priors.append(m.copy())
        prior=np.stack(priors);assert prior.shape==(H,K)
        energy=((mu-prior[None])**2).mean(-1);ids=rng.integers(0,N,size=(2048,N));tp=raw_output/(name+'_bootstrap_indices.npy');assert not tp.exists();np.save(tp,ids);raw_artifacts.append({'path':str(tp),'sha256':sha(tp)})
        oracle_energy.append({'task':name,'horizons':[{'h':h,'predictable_state_energy':float(energy[:,j].mean()),'realized_oracle_noise_mse':float(noise_recorded[:,j].mean())} for j,h in enumerate(hs)]})
        task_rows=[r for r in main['records']+extra['records'] if r['task']==name];assert len(task_rows)==8
        for row in task_rows:
            path=pathlib.Path(row['artifact']);assert sha(path)==row['artifact_sha256'];inputs[str(path)]=row['artifact_sha256']
            with np.load(path) as a:p=a['state_mean'];saved_factual=a['state_sequence_mse'];saved_excess=a['conditional_mean_sequence_mse']
            values,error=decompose(p,mu,truth);max_identity=max(max_identity,error)
            saved_error=max(float(np.max(abs(values['factual']-saved_factual))),float(np.max(abs(values['excess_conditional']-saved_excess))),float(np.max(abs(values['oracle_noise']-noise_recorded))))
            assert saved_error<1e-10;max_saved=max(max_saved,saved_error);trace[row['run']]=values
            for j,h in enumerate(hs):
                e=float(energy[:,j].mean());rec={'run':row['run'],'task':name,'method':row['method'],'seed':row['seed'],'h':h,**{k:float(v[:,j].mean()) for k,v in values.items()},'predictable_state_energy':e,'normalized_excess_conditional':ratio(float(values['excess_conditional'][:,j].mean()),e)};records.append(rec)
        def compare(label,left,right):
            le=np.stack([trace[r['run']]['excess_conditional'] for r in left]).mean(0);re=np.stack([trace[r['run']]['excess_conditional'] for r in right]).mean(0)
            ci=paired_interval(le-re,energy,ids)
            comparisons.append({'task':name,'comparison':label,'left_runs':[r['run'] for r in left],'right_runs':[r['run'] for r in right],'interpretation':'Negative delta means lower conditional-mean discrepancy for the left side. Fixed models; uncertainty over heldout sequences only.','horizons':[{'h':h,**v} for h,v in zip(hs,ci)]})
        false=[r for r in task_rows if r['method']=='rp'];em=[r for r in task_rows if r['method']=='em'];true=[r for r in task_rows if r['method']=='rp_emtrue'];assert len(false)==len(em)==3 and len(true)==2
        for r in true:compare('existing True minus matched False, seed'+str(r['seed']),[r],[next(f for f in false if f['run']==r['matched_false_run'])])
        compare('mean of3 False model errors minus mean of3 corrected EM errors',false,em)
        compare('mean of2 True model errors minus mean of2 corrected EM errors',true,[r for r in em if r['seed'] in [0,1]])
    for p,value in inputs.items():assert sha(p)==value,'Existing scored artifact/report changed during read-only interpretation'
    with (target/'risk_by_model_horizon.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary={'utc':utc(),'status':'COMPLETE_READ_ONLY_CAUSAL_RISK_INTERPRETATION_OF_ALL16_FROZEN_FITS','records':records,'oracle_predictable_energy':oracle_energy,'comparisons':comparisons,'max_sample_risk_identity_error':max_identity,'max_saved_error_recomputation_difference':max_saved,'bootstrap_replicates':2048,'bootstrap_seed':5491,'interval_scope':'Pointwise95 percentiles over complete heldout sequences, conditional on fixed models/data; no multiplicity correction, no initialization/domain uncertainty, no selection.','raw_bootstrap_indices':raw_artifacts,'input_hashes_unchanged':inputs,'model_or_hyperparameter_selection':False,'additional_optimizer_updates':0,'accepted_new_directions':0,'protocol_sha256':sha(ROOT/'CAUSAL_RISK_INTERPRETATION54_V1.md'),'registration_sha256':sha(OUT/'CAUSAL_RISK_REGISTRATION54_V1.json'),'script_sha256':sha(__file__),'scope':'Two linear synthetic tasks, supervised diagnostic state maps only. Standard risk decomposition/bootstrap, not a new metric/method, causal intervention test or full probability-law calibration.'}
    (target/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    text='# C54 causal state-risk interpretation\n\nAll16 scored models were fixed before any heldout score. This analysis reads their existing predictions; no new model is fit or selected.\n\n'
    text+='The realized error decomposes exactly into realized oracle error, conditional-mean discrepancy and a finite-sample cross term. Only the cross term\'s conditional expectation is zero. Predictable energy and normalized excess are reported at every original horizon, together with raw errors. See outputs/causal_risk_interpretation_v1/risk_by_model_horizon.csv.\n\n'
    text+='| Task | Existing-source or ordinary comparison | Horizon | Conditional-error delta | Pointwise95 sequence interval | Normalized delta |\n|---|---|---:|---:|---|---:|\n'
    for c in comparisons:
        for h in c['horizons']:
            normalized='undefined' if h['normalized_delta'] is None else f"{h['normalized_delta']:.6g}"
            text+=f"|{c['task']}|{c['comparison']}|{h['h']}|{h['delta_mean']:.6g}|[{h['pointwise95_interval'][0]:.6g}, {h['pointwise95_interval'][1]:.6g}]|{normalized}|\n"
    text+='\nIntervals condition on frozen fits and cover only complete-sequence sampling; they are pointwise and do not establish multiple-comparison significance or real-domain generalization. Mean per-model errors are compared, not prediction ensembles. Both exact source-option pairs and competent ordinary EM controls remain visible. A residual model-specific linear failure does not establish a new world-model gap, and prior-mean success at nearly unpredictable horizons does not establish useful dynamics. No direction is accepted automatically.\n\nAll existing scored artifacts/reports were hash-checked unchanged. Full risk components, normalized errors, intervals, raw resampling-index hashes and scope are in the machine-readable summary.\n'
    report=ROOT/'CAUSAL_RISK_REPORT54_V1.md';assert not report.exists();report.write_text(text,encoding='utf-8')
    print(json.dumps({'status':summary['status'],'models':16,'rows':len(records),'comparisons':len(comparisons),'max_identity_error':max_identity,'report_sha256':sha(report),'accepted_new_directions':0}),flush=True)
if __name__=='__main__':main()
