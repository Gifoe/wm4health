"""Seal completed C55 controls and an explicit compact publication scope."""
from common55 import *
import math

def main():
    state=read(OUT/'state55_v2.json');assert state['status']=='COMPLETE_SCALAR_POPULATION_NUMERICS_PENDING_INTERPRETATION'
    analysis=read(OUT/'ANALYSIS55_V1.json');assert analysis['status']=='SCALAR_POPULATION_CONTROL_INTERPRETED_NO_NEW_DIRECTION_ACCEPTED'
    q=read(OUT/'QUALIFICATION55_V2.json');s=read(OUT/'population55_v2.json')
    assert analysis['population_results_sha256']==sha(OUT/'population55_v2.json')
    assert q['protocol_sha256']==sha(ROOT/'PROTOCOL_V2.md') and q['math_script_sha256']==sha(ROOT/'scripts/population55_v2.py')
    assert s['qualification_sha256']==sha(OUT/'QUALIFICATION55_V2.json')
    for name,value in analysis['reports'].items():assert sha(ROOT/name)==value
    pins=source_check();assert pins==s['source_files_sha256']
    storage=read(PROJECT/'STORAGE_LOCATION_V1.json')
    for row in s['all_144_numerical_trials']:assert sha(pathlib.Path(storage['artifact_root'])/row['trace_storage_relative_path'])==row['trace_sha256']
    closure={'utc':utc(),'status':'C55_SCALAR_POPULATION_CONTROL_AND_INTERPRETATION_CLOSED','cases':24,'numerical_trials':144,'iteration_trace_hashes_verified':144,'source_files_verified':21,'native_neural_training_updates':0,'native_finite_loss_max_error':analysis['statistics']['native_finite_loss_max_error'],'max_ideal_true_point_tightness_error':analysis['statistics']['max_ideal_true_point_tightness_error'],'reports_verified':len(analysis['reports']),'population_result_sha256':sha(OUT/'population55_v2.json'),'analysis_sha256':sha(OUT/'ANALYSIS55_V1.json'),'qualification_sha256':sha(OUT/'QUALIFICATION55_V2.json'),'protocol_sha256':sha(ROOT/'PROTOCOL_V2.md'),'derivation_sha256':sha(ROOT/'DERIVATION.md'),'first_failed_qualification_retained':(OUT/'qualify55_v1.py.stderr.log').exists(),'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'local_result_files_written':0}
    dump(OUT/'CLOSURE55_V1.json',closure)
    dest=OUT/'publication_snapshot_v1';assert not dest.exists();dest.mkdir();candidates=[]
    for folder in [ROOT,ROOT/'scripts',OUT]:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ['.py','.ps1','.md','.json','.jsonl','.csv','.log']:candidates.append(p)
    rows=[]
    for p in sorted(candidates):
        public=p.relative_to(ROOT).as_posix();original=p.read_bytes();text=original.decode('utf-8-sig');data=text.encode('utf-8')
        assert len(data)<1000000,(public,len(data))
        target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        rows.append({'path':public,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'original_sha256':hashlib.sha256(original).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':'Remove UTF8 BOM' if original.startswith(b'\xef\xbb\xbf') else None})
    manifest={'utc':utc(),'status':'C55_CLOSED_SCALAR_POPULATION_CONTROL_PUBLICATION','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'closure_sha256':sha(OUT/'CLOSURE55_V1.json'),'scope':'Authored source, all144 compact numerical outcomes, qualification/failure logs, derivation and eight reports. Full numerical iteration traces stay on the configured large disk. No formal empirical model training or new method claim.','overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'local_result_files_written':0}
    dump(dest/'manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'files':len(rows),'bytes':manifest['bytes'],'parts':manifest['total_content_parts'],'manifest_sha256':sha(dest/'manifest.json'),'closure_sha256':sha(OUT/'CLOSURE55_V1.json')}),flush=True)

if __name__=='__main__':main()
