"""Publish only the prescoring risk protocol/code/registration and closed dummy checks."""
import datetime,hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    assert not (OUT/'evaluation54.json').exists() and not (OUT/'READOUT_SEAL54_V1.json').exists()
    dest=OUT/'risk_protocol_publication_snapshot_v9';assert not dest.exists();dest.mkdir()
    reg=json.loads((OUT/'CAUSAL_RISK_REGISTRATION54_V1.json').read_text());assert reg['heldout_opened_at_registration'] is False
    q=json.loads((OUT/'QUALIFICATION_RISK_MATH54_V1.json').read_text());assert q['formal_model_reads']==q['formal_data_or_label_reads']==0
    for name,digest in reg['files_sha256'].items():assert sha((ROOT/name).read_bytes())==digest
    names=['CAUSAL_RISK_INTERPRETATION54_V1.md','scripts/risk_math54_v1.py','scripts/qualify_risk_math54_v1.py','scripts/analyze_causal_risk54_v1.py','scripts/watch_causal_risk54_v1.py','scripts/register_causal_risk54_v1.py','scripts/export_risk_protocol54_v9.py','outputs/CAUSAL_RISK_REGISTRATION54_V1.json','outputs/QUALIFICATION_RISK_MATH54_V1.json']
    candidates={name:ROOT/name for name in names}
    watch=json.loads((OUT/'watch_causal_risk54_v1.json').read_text());assert watch['status']=='WAITING_FOR_EXISTING_ALL16_POSTPROCESS_COMPLETION_NO_EXTRA_TRAINING'
    snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'READ_ONLY_RISK_INTERPRETATION_REGISTERED_BEFORE_SCORING','watcher_state_observed':watch,'heldout_score_exists':False,'formal_risk_artifact_reads':0,'additional_optimizer_updates':0,'existing_training_and_scoring_scripts_changed':False,'accepted_directions':0,'overall_goal':'ACTIVE_UNACHIEVED','local_result_files_written':0}
    p=dest/'RISK_PROTOCOL_PROGRESS54_V9.json';p.write_text(json.dumps(snapshot,indent=2),encoding='utf-8');candidates[p.name]=p
    rows=[]
    for name,p in sorted(candidates.items()):
        original=p.read_bytes();content=original.decode('utf-8-sig');b=content.encode('utf-8');target=dest/'files'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        rows.append({'path':name,'bytes':len(b),'sha256':sha(b),'original_sha256':sha(original),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'parts':math.ceil(len(content)/3500),'server_snapshot_file':str(target)})
    manifest={'utc':snapshot['utc'],'status':'IMMUTABLE_PRESCORING_READ_ONLY_ANALYSIS_REGISTRATION','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'scope':'Prospective protocol, implementation, registration and independently constructed qualification fixtures only. No scored formal artifacts, raw signals, weights, venvs or private connection helpers.'}
    p=dest/'manifest.json';p.write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'files':len(rows),'bytes':manifest['bytes'],'parts':manifest['total_content_parts'],'manifest_sha256':sha(p.read_bytes())}),flush=True)
if __name__=='__main__':main()
