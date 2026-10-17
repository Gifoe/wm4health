"""Snapshot compact evidence and licensed source; raw signals/coefficients stay on G:."""
from common59 import *
import base64,csv,io,math
def main():
    closure=read(OUT/'CLOSURE59_V1.json');source=read(OUT/'DOWNLOAD59_V2.json');result=read(OUT/'BASELINE_RESULTS59_V1.json');seal=read(OUT/'MODEL_SEAL59_V1.json')
    assert closure['status']=='C59_DATA_CORE_QUALIFICATION_AND_TEN_BASELINE_SCREEN_CLOSED'
    for name,h in closure['reports_sha256'].items():assert sha(ROOT/name)==h
    assert closure['baseline_results_sha256']==sha(OUT/'BASELINE_RESULTS59_V1.json')
    for row in result['models']:
        assert sha(row['model_file'])==row['model_sha256'] and sha(row['prediction_file'])==row['prediction_sha256']
    for f in source['files']:assert sha(f['server_path'])==f['sha256']
    trace=io.StringIO(newline='');writer=csv.DictWriter(trace,fieldnames=['prefix','method','ridge','a','validation_mse']);writer.writeheader();writer.writerows(seal['validation_candidates'])
    with (OUT/'validation_candidate_trace59_v1.csv').open('x',encoding='utf-8',newline='') as f:f.write(trace.getvalue())
    selected=[]
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.md','.json','.ps1']:selected.append(p)
    selected.extend((ROOT/'scripts').glob('*.py'))
    for f in source['files']:
        if f['kind']=='source':selected.append(pathlib.Path(f['server_path']))
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in ['.json','.log','.csv']:selected.append(p)
    dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);rows=[]
    for p in sorted(set(selected)):
        public=p.relative_to(ROOT).as_posix();raw=p.read_bytes();original_sha=hashlib.sha256(raw).hexdigest()
        try:
            text=raw.decode('utf-8-sig');transformation='Remove UTF8 BOM' if raw.startswith(b'\xef\xbb\xbf') else None
        except UnicodeDecodeError:
            public+='.original_bytes.json'
            text=json.dumps({'original_relative_path':p.relative_to(ROOT).as_posix(),'original_bytes':len(raw),'original_sha256':original_sha,'encoding':'base64','reason':'Preserve original log bytes without guessing the terminal encoding','content':base64.b64encode(raw).decode()},indent=2)
            transformation='Lossless base64 envelope for non-UTF8 log'
        data=text.encode('utf-8');target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
        rows.append({'path':public,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'original_sha256':original_sha,'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':transformation})
    obj={'utc':utc(),'status':'C59_CLOSED_QUALIFICATION_AND_ORDINARY_BASELINE_SCREEN_PUBLICATION','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'closure_sha256':sha(OUT/'CLOSURE59_V1.json'),'source_commit':source['source_commit'],'source_objects_verified':len(source['files']),'candidate_fits':80,'selected_deterministic_fits':10,'neural_updates':0,'scope':'Raw provenance, 32-case equation-core qualification, fixed train/validation selection, all ten heldout outcomes, independent re-scoring, negative evidence and licensed original tutorial source. Raw signals, coefficients and prediction arrays remain on authorized server storage with hashes. No cross-system or new-method claim.','no_local_result_files_written':True,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(dest/'manifest.json',obj)
    print(json.dumps({'status':obj['status'],'files':len(rows),'bytes':obj['bytes'],'parts':obj['total_content_parts'],'manifest_sha256':sha(dest/'manifest.json')}),flush=True)
if __name__=='__main__':main()
