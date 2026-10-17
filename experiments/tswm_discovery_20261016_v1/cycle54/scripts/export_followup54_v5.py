"""Immutable public follow-up delta; server outputs stay off the local filesystem."""
import datetime, hashlib, json, math, pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
def digest(b): return hashlib.sha256(b).hexdigest()
def main():
    assert not (OUT/'evaluation54.json').exists() and not (OUT/'READOUT_SEAL54_V1.json').exists()
    dest=OUT/'followup_publication_snapshot_v5'; assert not dest.exists(); dest.mkdir()
    old={}
    for sub in ['protocol_publication_snapshot_v1','progress_publication_snapshot_v3','training_audit_publication_snapshot_v4']:
        m=json.loads((OUT/sub/'manifest.json').read_text()); old.update({r['path']:r['sha256'] for r in m['files']})
    candidates={}
    def add(p,public=None):
        assert p.is_file(); candidates[public or p.relative_to(ROOT).as_posix()]=p
    for p in ROOT.rglob('*'):
        r=p.relative_to(ROOT)
        if p.is_file() and 'outputs' not in r.parts and '__pycache__' not in r.parts and p.suffix in ['.py','.ps1','.md','.json'] and old.get(r.as_posix())!=digest(p.read_bytes()): add(p)
    for p in (OUT/'encoder_reference_audit_v1').glob('*.json'): add(p)
    for name in ['dcl_catalog_probe54_v1.json','dcl_catalog_summary54_v1.json']: add(OUT/name)
    counts=[]
    for base in [OUT/'runs',OUT/'em_option_learning_v1/runs']:
        complete=[]
        for folder in sorted(base.glob('*')):
            receipt=folder/'result.json'
            if not receipt.exists(): continue
            r=json.loads(receipt.read_text()); assert digest(pathlib.Path(r['checkpoint']).read_bytes())==r['checkpoint_sha256']
            complete.append({k:r[k] for k in ['run','method','task','seed','iterations','checkpoint_sha256','runtime_seconds']})
            for p in folder.iterdir():
                public=p.relative_to(ROOT).as_posix()
                if p.is_file() and p.suffix in ['.json','.jsonl','.log'] and old.get(public)!=digest(p.read_bytes()): add(p)
        counts.append(complete)
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    progress={'utc':stamp,'status':'ENCODER_ATTRIBUTION_COMPLETE_AND_DCL_CONTRACT_ACCESS_REVIEWED_TRAINING_CONTINUES','primary_completed':len(counts[0]),'supplemental_completed':len(counts[1]),'planned_total':16,'completed_fits':counts,'heldout_opened':False,'all_model_seals_required_before_scoring':True,'accepted_directions':0,'overall_goal':'ACTIVE_UNACHIEVED','local_result_file_sync':False}
    p=dest/'PROGRESS54_V5.json'; p.write_text(json.dumps(progress,indent=2),encoding='utf-8'); add(p,p.name)
    header='# C54 follow-up: parameter-block attribution and next-baseline input contract\n\n'
    header+=f"Snapshot UTC {stamp}: {len(counts[0])}/12 primary and {len(counts[1])}/4 existing-EM-option fits complete. No heldout scoring yet.\n\n"
    header+='The completed [encoder-reference audit](ENCODER_REFERENCE_FINDING54_V1.md) narrows the existing EM remedy to the A block at the frozen points; recognition gradients retain a predominantly magnitude difference. This is not a trained-model benefit or a new-method claim.\n\n'
    header+='The [DCL input audit](DCL_INFERENCE_CONTRACT54_V1.md) distinguishes retrospective switching identification from causal simulation and retains the paper\'s own heldout, delay and multi-step evidence. The [public data catalog is accessible from the server](DCL_DATA_ACCESS54_V1.md); only metadata were fetched. No DCL training is claimed.\n\n'
    earlier=(OUT/'training_audit_publication_snapshot_v4/files/README.md').read_text()
    p=dest/'README.md'; p.write_text(header+'## Earlier immutable V4 progress snapshot\n\n'+earlier,encoding='utf-8'); add(p,p.name)
    records=[]
    for public,p in sorted(candidates.items()):
        original=p.read_bytes(); text=original.decode('utf-8-sig'); b=text.encode('utf-8')
        if old.get(public)==digest(b): continue
        assert len(b)<350000,(public,len(b))
        target=dest/'files'/public; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(b)
        records.append({'path':public,'bytes':len(b),'characters':len(text),'parts':math.ceil(len(text)/3500),'sha256':digest(b),'original_sha256':digest(original),'publication_transform':'Remove UTF-8 BOM' if original.startswith(b'\xef\xbb\xbf') else None,'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'server_snapshot_file':str(target)})
    manifest={'utc':stamp,'status':'IMMUTABLE_FROZEN_ENCODER_AND_SOURCE_CONTRACT_FOLLOWUP_DELTA','file_count':len(records),'bytes':sum(r['bytes'] for r in records),'total_content_parts':sum(r['parts'] for r in records),'files':records,'excluded':'Signals, checkpoints, private connection helpers, changing logs and full provider metadata. Public source/API links and advertised file checksums are metadata only.'}
    p=dest/'manifest.json'; p.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(records),'bytes':manifest['bytes'],'parts':manifest['total_content_parts'],'manifest_sha256':digest(p.read_bytes()),'primary_completed':len(counts[0]),'supplemental_completed':len(counts[1])}),flush=True)
if __name__=='__main__': main()
