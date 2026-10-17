"""Explicit-scope publication excludes the new venv and all upstream packages."""
import datetime,hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    assert not (OUT/'evaluation54.json').exists() and not (OUT/'READOUT_SEAL54_V1.json').exists()
    dest=OUT/'dcl_preflight_publication_snapshot_v7';assert not dest.exists();dest.mkdir();old={};candidates={}
    for sub in ['protocol_publication_snapshot_v1','progress_publication_snapshot_v3','training_audit_publication_snapshot_v4','followup_publication_snapshot_v6']:
        for r in json.loads((OUT/sub/'manifest.json').read_text())['files']:old[r['path']]=r['sha256']
    def add(p,name=None):
        assert p.is_file();candidates[name or p.relative_to(ROOT).as_posix()]=p
    # Deliberately do not recursively scan ROOT: .venv_dcl is outside publication scope.
    for folder in [ROOT,ROOT/'scripts']:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ['.py','.ps1','.md','.json'] and old.get(p.relative_to(ROOT).as_posix())!=sha(p.read_bytes()):add(p)
    for name in ['DCL_RUNTIME_PREPARATION54_V1.json','dcl_dependency_install54_v1.stdout.log','dcl_dependency_install54_v1.stderr.log']:add(OUT/name)
    add(OUT/'dcl_preflight_v2/summary.json')
    counts=[]
    for base in [OUT/'runs',OUT/'em_option_learning_v1/runs']:
        complete=[]
        for folder in sorted(base.glob('*')):
            p=folder/'result.json'
            if not p.exists():continue
            r=json.loads(p.read_text());assert sha(pathlib.Path(r['checkpoint']).read_bytes())==r['checkpoint_sha256'];complete.append({'run':folder.name,**{k:r[k] for k in ['method','task','seed','iterations','checkpoint_sha256','runtime_seconds']}})
            for p in folder.iterdir():
                if p.is_file() and p.suffix in ['.json','.jsonl','.log'] and old.get(p.relative_to(ROOT).as_posix())!=sha(p.read_bytes()):add(p)
        counts.append(complete)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();progress={'utc':now,'status':'DCL_NATIVE_CORE_QUALIFIED_C54_LEARNING_STILL_PENDING','primary_completed':len(counts[0]),'supplemental_completed':len(counts[1]),'planned_C54_fits':16,'DCL_formal_fits':0,'DCL_discarded_updates':32,'C54_heldout_opened':False,'accepted_directions':0,'overall_goal':'ACTIVE_UNACHIEVED','local_result_files_written':0,'completed_C54_fits':counts}
    p=dest/'PROGRESS54_V7.json';p.write_text(json.dumps(progress,indent=2),encoding='utf-8');add(p,p.name)
    text='# C54: matched learning comparison continues; native DCL core qualified\n\n'+f"Snapshot UTC {now}: {len(counts[0])}/12 primary fits and {len(counts[1])}/4 existing EM-option controls complete. No heldout score has opened.\n\n"
    text+='The [DCL native core qualification](DCL_PREFLIGHT_REPORT54_V1.md) executed independent loss/gradient/matrix checks and32 discarded toy GPU updates with unchanged source. This is0 formal DCL fits and no method-performance claim. The53-file source pin, actual architecture and separate runtime are recorded. The [encoder-reference finding](ENCODER_REFERENCE_FINDING54_V1.md) and [DCL input contract](DCL_INFERENCE_CONTRACT54_V1.md) remain the scientific boundaries.\n\n'
    text+='The next decision awaits all16 fixed C54 models and their training-only diagnostic maps before heldout scoring. No final new research direction has been accepted.\n'
    p=dest/'README.md';p.write_text(text,encoding='utf-8');add(p,p.name)
    rows=[]
    for public,p in sorted(candidates.items()):
        original=p.read_bytes();content=original.decode('utf-8-sig');b=content.encode('utf-8')
        if old.get(public)==sha(b):continue
        assert len(b)<350000 and not any(t in public.split('/') for t in ['.venv_dcl','Lib','site-packages','__pycache__'])
        target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        rows.append({'path':public,'bytes':len(b),'sha256':sha(b),'original_sha256':sha(original),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'parts':math.ceil(len(content)/3500),'publication_transform':'Remove UTF8 BOM' if original.startswith(b'\xef\xbb\xbf') else None,'server_snapshot_file':str(target)})
    manifest={'utc':now,'status':'IMMUTABLE_NATIVE_DCL_PREFLIGHT_AND_C54_PROGRESS_DELTA','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'excluded':'Virtual environments, installed dependencies, upstream package bodies, model weights, raw signals, changing logs and private transport helpers.'}
    p=dest/'manifest.json';p.write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'files':len(rows),'bytes':manifest['bytes'],'parts':manifest['total_content_parts'],'manifest_sha256':sha(p.read_bytes()),'primary_completed':len(counts[0]),'supplemental_completed':len(counts[1])}),flush=True)
if __name__=='__main__':main()
