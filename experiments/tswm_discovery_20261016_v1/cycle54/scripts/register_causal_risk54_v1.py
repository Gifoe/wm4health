"""Timestamp/hash the fixed read-only analysis before any heldout scoring."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    target=OUT/'CAUSAL_RISK_REGISTRATION54_V1.json';assert not target.exists()
    assert not (OUT/'evaluation54.json').exists() and not (OUT/'READOUT_SEAL54_V1.json').exists()
    q=OUT/'QUALIFICATION_RISK_MATH54_V1.json';v=json.loads(q.read_text());assert v['status']=='PASS_INDEPENDENT_DUMMY_BAYES_RISK_AND_SEQUENCE_BOOTSTRAP_ARITHMETIC' and v['formal_model_reads']==v['formal_data_or_label_reads']==0
    assert sha(ROOT/'scripts/risk_math54_v1.py')==v['module_sha256']
    names=['CAUSAL_RISK_INTERPRETATION54_V1.md','scripts/risk_math54_v1.py','scripts/qualify_risk_math54_v1.py','scripts/analyze_causal_risk54_v1.py','scripts/watch_causal_risk54_v1.py']
    obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'REGISTERED_BEFORE_ANY_HELDOUT_SCORE','heldout_opened_at_registration':False,'files_sha256':{n:sha(ROOT/n) for n in names},'qualification_sha256':sha(q),'horizons':[0,1,5,10,25,50],'sequence_bootstrap_replicates':2048,'sequence_bootstrap_seed':5491,'interval_interpretation':'Pointwise95 descriptive sequence intervals; conditional on fixed models, no multiplicity correction or method selection.','changes_to_existing_models_or_scoring_pipeline':False,'additional_optimizer_updates':0,'model_or_data_reads':0,'scope':'Read-only interpretation after existing all16-fit/all16-map seals and completed scoring, not a new method, metric or accepted direction.'}
    target.write_text(json.dumps(obj,indent=2),encoding='utf-8');print(json.dumps(obj),flush=True)
if __name__=='__main__':main()
