"""Snapshot terminal compact evidence and original licensed source on the server."""
from common60 import *
import base64, math

def main():
    closure=read(OUT/'CLOSURE60_V1.json');source=source_check()
    evaluation=read(OUT/'EVALUATION60_V2.json');seal=read(OUT/'COMBINED_MODEL_SEAL60_V1.json')
    assert closure['status']=='C60_TWO_SYSTEM_CONTROL_COMPARISON_INTERPRETED'
    assert read(OUT/'post_state60_v2.json')['status']=='ALL_NSSM_GRU_AND_BANK_RESULTS_READY_FOR_INTERPRETATION'
    for name,h in closure['reports_sha256'].items():assert sha(ROOT/name)==h
    assert closure['evaluation_sha256']==sha(OUT/'EVALUATION60_V2.json')
    assert len(seal['models'])+len(seal['failed'])==12
    for row in evaluation['records']:
        assert sha(row['arrays'])==row['arrays_sha256']
    selected=[]
    for p in ROOT.iterdir():
        if p.is_file() and p.suffix in ['.md','.json','.ps1']:selected.append(p)
    selected.extend((ROOT/'scripts').glob('*.py'))
    for row in source['files']:
        if row['kind']=='source':selected.append(pathlib.Path(row['server_path']))
    for p in OUT.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl','.log','.csv'] and not any(x.startswith('publication_') for x in p.relative_to(OUT).parts):selected.append(p)
    dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);rows=[]
    for p in sorted(set(selected)):
        public=p.relative_to(ROOT).as_posix();raw=p.read_bytes();original_sha=hashlib.sha256(raw).hexdigest()
        try:
            text=raw.decode('utf-8-sig');transformation='Remove UTF8 BOM' if raw.startswith(b'\xef\xbb\xbf') else None
        except UnicodeDecodeError:
            public+='.original_bytes.json'
            text=json.dumps({'original_relative_path':p.relative_to(ROOT).as_posix(),'original_bytes':len(raw),'original_sha256':original_sha,'encoding':'base64','reason':'Preserve original log bytes without guessing terminal encoding','content':base64.b64encode(raw).decode()},indent=2)
            transformation='Lossless base64 envelope for non-UTF8 log'
        data=text.encode('utf-8');target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
        rows.append({'path':public,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'original_sha256':original_sha,'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':transformation})
    obj={'utc':utc(),'status':'C60_CLOSED_TWO_SYSTEM_CONTROL_PUBLICATION','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'closure_sha256':sha(OUT/'CLOSURE60_V1.json'),'source_commit':source['source_commit'],'source_objects_verified':len(source['files']),'predeclared_neural_attempts':12,'ordinary_pick_place_candidates':40,'scope':'Qualified native-component NSSM and ordinary parameter-comparable GRU, all seeds and failed/superseded preflight artifacts, chronological measured mechanical split and exploratory historical N87 comparison. Raw signals, checkpoints and prediction arrays stay on authorized G storage with hashes. No new framework or shared-checkpoint claim.','no_local_result_files_written':True,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(dest/'manifest.json',obj)
    print(json.dumps({'status':obj['status'],'files':len(rows),'bytes':obj['bytes'],'parts':obj['total_content_parts'],'manifest_sha256':sha(dest/'manifest.json')}),flush=True)

if __name__=='__main__':main()
