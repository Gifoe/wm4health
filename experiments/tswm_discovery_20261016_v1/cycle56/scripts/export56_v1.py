"""Seal an explicit compact evidence scope; large raw arrays stay on server."""
from common56 import *
import math

def main():
    state=read(OUT/'state56_v1.json')
    assert state['status']=='COMPLETE_FINITE_REFERENCE_NUMERICS_PENDING_INTERPRETATION'
    closure=read(OUT/'CLOSURE56_V1.json');summary=read(OUT/'SUMMARY56_V1.json')
    result=read(OUT/'GRADIENT56_V1.json');qual=read(OUT/'QUALIFICATION56_V1.json')
    assert closure['status']=='C56_NUMERICS_AND_INTERPRETATION_CLOSED'
    assert closure['summary_sha256']==sha(OUT/'SUMMARY56_V1.json')
    assert summary['gradient_results_sha256']==sha(OUT/'GRADIENT56_V1.json')
    assert result['qualification_sha256']==sha(OUT/'QUALIFICATION56_V1.json')
    assert result['protocol_sha256']==qual['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    assert result['script_sha256']==sha(ROOT/'scripts/run56_v1.py')
    assert result['math_sha256']==sha(ROOT/'scripts/math56.py')
    assert source_check()==result['native_source_sha256']
    for p,h in closure['report_sha256'].items():assert sha(ROOT/p)==h
    for row in summary['raw_gradient_receipts']+summary['raw_bank_receipts']:assert sha(row['path'])==row['sha256']
    candidates=[ROOT/'PROTOCOL.md',ROOT/'DERIVATION.md',ROOT/'start56_v1.ps1']
    candidates += [ROOT/p for p in closure['report_sha256']]
    candidates += sorted((ROOT/'scripts').glob('*.py'))
    candidates += [OUT/p for p in ['state56_v1.json','QUALIFICATION56_V1.json','GRADIENT56_V1.json','SUMMARY56_V1.json','CLOSURE56_V1.json','gradients56_v1.csv','qualify56_v1.py.stdout.log','qualify56_v1.py.stderr.log','run56_v1.py.stdout.log','run56_v1.py.stderr.log']]
    assert len(candidates)==len(set(candidates))
    dest=OUT/'publication_snapshot_v1';dest.mkdir(exist_ok=False)
    rows=[]
    for p in sorted(candidates):
        public=p.relative_to(ROOT).as_posix();original=p.read_bytes();text=original.decode('utf-8-sig');data=text.encode('utf-8')
        assert len(data)<1000000,(public,len(data))
        target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
        rows.append({'path':public,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'original_sha256':hashlib.sha256(original).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),'parts':math.ceil(len(text)/3500),'server_snapshot_file':str(target),'publication_transform':'Remove UTF8 BOM' if original.startswith(b'\xef\xbb\xbf') else None})
    manifest={'utc':utc(),'status':'C56_CLOSED_FINITE_REFERENCE_GRADIENT_PUBLICATION','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'closure_sha256':sha(OUT/'CLOSURE56_V1.json'),'scope':'Pinned-source derivative qualification, fixed protocol, derivation, all 144 compact diagnostic rows, raw/CV gradient uncertainty, logs, code and eight reports. Per-replicate arrays and bank arrays stay on configured large-disk storage with verified hashes. No neural training or accepted method.','overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'local_result_files_written':0}
    dump(dest/'manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'files':len(rows),'bytes':manifest['bytes'],'parts':manifest['total_content_parts'],'manifest_sha256':sha(dest/'manifest.json'),'closure_sha256':sha(OUT/'CLOSURE56_V1.json')}),flush=True)

if __name__=='__main__':main()
