"""Seal complete evidence only after actual figure review; preserve prior Git blobs."""
from common57 import *
import math

def main():
    assert read(OUT/'training_state57_v2.json')['status']=='ALL_32_FIXED_TERMINAL_FITS_COMPLETE_HELDOUT_UNOPENED'
    assert read(OUT/'post_state57_v2.json')['status']=='ALL_32_MODELS_AND_MAPS_SEALED_AND_SCORED_PENDING_INTERPRETATION'
    assert read(OUT/'close_state57_v1.json')['status']=='INTERPRETATION_AND_FIGURES_COMPLETE_ACTUAL_VISUAL_QA_PENDING'
    summary=read(OUT/'SUMMARY57_V1.json');closure=read(OUT/'CLOSURE57_V1.json');models=read(OUT/'MODEL_SEAL57_V1.json');maps=read(OUT/'MAP_SEAL57_V1.json')
    figs=read(OUT/'FIGURES_NUMERIC_QA57_V1.json');visual=read(OUT/'RESULT_FIGURES_ACTUAL_QA57_V1.json')
    assert visual['status']=='PASS_ACTUAL_RENDER_AND_NUMERICS'
    assert visual['numeric_qa_sha256']==sha(OUT/'FIGURES_NUMERIC_QA57_V1.json')
    assert figs['point_count']==64 and figs['all_actual_dot_offsets_verified']
    assert summary['valid_terminal_models']==32 and summary['neural_updates_valid']==300000
    assert summary['retained_failed_updates']==10000 and summary['accepted_final_directions']==0
    assert closure['summary_sha256']==sha(OUT/'SUMMARY57_V1.json')
    assert closure['evaluation_sha256']==sha(OUT/'EVALUATION57_V1.json')
    for p,h in closure['reports_sha256'].items():assert sha(ROOT/p)==h
    for model in models['models']:
        assert sha(model['checkpoint'])==model['checkpoint_sha256']
        assert sha(OUT/'runs'/model['tag']/'result.json')==model['result_sha256']
    for row in maps['maps']:assert sha(row['map'])==row['map_sha256']
    for row in read(OUT/'EVALUATION57_V1.json')['records']:assert sha(row['arrays'])==row['arrays_sha256']
    for row in figs['exports']:
        for extension in ['svg','png','pdf']:assert sha(row[extension])==row[extension+'_sha256']
        reviewed=next(r for r in visual['inspected_figures'] if r['name']==row['name'])
        assert reviewed['png_sha256']==row['png_sha256']
    pins=source_check()
    old_path=OUT/'publication_progress_v1/manifest.json';old=read(old_path)
    prior_receipt=read(OUT/'PROGRESS_PUBLICATION_RECEIPT57_V1.json')
    assert prior_receipt['commit']=='6df0d48e00ab91baace60dcbc52f20296662e54e'
    assert prior_receipt['manifest_sha256']==sha(old_path)
    seal={'utc':utc(),'status':'C57_ALL_TRAINING_SCORING_INTERPRETATION_AND_FIGURE_QA_CLOSED','valid_terminal_models':32,'train_only_mean_maps':32,'new_neural_updates_valid':300000,'retained_failed_update_attempts':10000,'deterministic_proper_fits':2,'native_source_files_verified':len(pins),'summary_sha256':sha(OUT/'SUMMARY57_V1.json'),'scientific_closure_sha256':sha(OUT/'CLOSURE57_V1.json'),'visual_qa_sha256':sha(OUT/'RESULT_FIGURES_ACTUAL_QA57_V1.json'),'all_model_map_and_array_hashes_verified':True,'all_prior_published_file_hashes_preserved':True,'previous_progress_commit':prior_receipt['commit'],'local_result_files_written':0,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    # Check every immutable file from the progress publication before final sealing.
    candidates={r['path']:ROOT/r['path'] for r in old['files']}
    for row in old['files']:
        raw=candidates[row['path']].read_bytes();data=raw.decode('utf-8-sig').encode('utf-8')
        assert hashlib.sha256(data).hexdigest()==row['sha256'],('Previously published file changed',row['path'])
    dump(OUT/'FINAL_CLOSURE57_V1.json',seal)
    candidates['PROGRESS_MANIFEST_V1.json']=old_path
    for pattern in ['*.md','*.ps1']:
        for p in ROOT.glob(pattern):candidates[p.relative_to(ROOT).as_posix()]=p
    for p in (ROOT/'scripts').glob('*.py'):candidates[p.relative_to(ROOT).as_posix()]=p
    candidates['visual-composer/visual-contract.md']=ROOT/'visual-composer/visual-contract.md'
    output_names=['MODEL_SEAL57_V1.json','MAP_SEAL57_V1.json','HELDOUT_OPEN_RECEIPT57_V1.json','EVALUATION57_V1.json','SUMMARY57_V1.json','CLOSURE57_V1.json','FINAL_CLOSURE57_V1.json','results57_v1.csv','FIGURE_VALUES57_V1.csv','FIGURES_NUMERIC_QA57_V1.json','RESULT_FIGURES_ACTUAL_QA57_V1.json','INDEPENDENT_SUPPORT57_V1.json','PROGRESS_PUBLICATION_RECEIPT57_V1.json','training_state57_v2.json','post_state57_v2.json','close_state57_v1.json','fit_maps57_v1.py.stdout.log','fit_maps57_v1.py.stderr.log','evaluate57_v1.py.stdout.log','evaluate57_v1.py.stderr.log','analyze57_v1.py.stdout.log','analyze57_v1.py.stderr.log','figures57_v1.py.stdout.log','figures57_v1.py.stderr.log']
    for name in output_names:candidates['outputs/'+name]=OUT/name
    for item in models['models']:
        for name in ['result.json','curve.jsonl']:
            p=OUT/'runs'/item['tag']/name;candidates[p.relative_to(ROOT).as_posix()]=p
        p=OUT/(item['tag']+'.stderr.log');candidates[p.relative_to(ROOT).as_posix()]=p
    for p in (OUT/'figures').glob('*.svg'):candidates[p.relative_to(ROOT).as_posix()]=p
    dest=OUT/'publication_closed_v1';dest.mkdir(exist_ok=False);rows=[]
    for public,p in sorted(candidates.items()):
        original=p.read_bytes();text=original.decode('utf-8-sig');data=text.encode('utf-8')
        assert len(data)<1000000,(public,len(data))
        target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(data)
        rows.append({'path':public,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'original_sha256':hashlib.sha256(original).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':'Remove UTF8 BOM' if original.startswith(b'\xef\xbb\xbf') else None})
    manifest={'utc':utc(),'status':'C57_CLOSED_EVIDENCE_PUBLICATION','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'final_closure_sha256':sha(OUT/'FINAL_CLOSURE57_V1.json'),'prior_progress_manifest_sha256':sha(old_path),'prior_publication_commit':prior_receipt['commit'],'scope':'All 32 terminal-fit receipts and training curves, sealed maps/scoring and complete metric tables, independent-support counterexample, retained failed unshifted adaptation, code/protocol/amendments, eight reports and source-bound reviewed SVG figures. Large checkpoint/data/prediction arrays and PNG/PDF stay on authorized server storage with verified hashes. No final research direction accepted.','overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'local_result_files_written':0}
    dump(dest/'manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'files':len(rows),'bytes':manifest['bytes'],'parts':manifest['total_content_parts'],'manifest_sha256':sha(dest/'manifest.json'),'final_closure_sha256':sha(OUT/'FINAL_CLOSURE57_V1.json')}),flush=True)

if __name__=='__main__':main()
