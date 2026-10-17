"""Explicit-scope publication excludes the new venv and all upstream packages."""
import datetime,hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    assert not (OUT/'evaluation54.json').exists() and not (OUT/'READOUT_SEAL54_V1.json').exists()
    dest=OUT/'literature_publication_snapshot_v10';assert not dest.exists();dest.mkdir();old={};candidates={}
    for sub in ['protocol_publication_snapshot_v1','progress_publication_snapshot_v3','training_audit_publication_snapshot_v4','followup_publication_snapshot_v6','dcl_preflight_publication_snapshot_v7','dcl_preflight_publication_snapshot_v8','risk_protocol_publication_snapshot_v9']:
        for r in json.loads((OUT/sub/'manifest.json').read_text())['files']:old[r['path']]=r['sha256']
    def add(p,name=None):
        assert p.is_file();candidates[name or p.relative_to(ROOT).as_posix()]=p
    # Deliberately do not recursively scan ROOT: .venv_dcl is outside publication scope.
    for folder in [ROOT,ROOT/'scripts']:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ['.py','.ps1','.md','.json'] and old.get(p.relative_to(ROOT).as_posix())!=sha(p.read_bytes()):add(p)
    for name in ['DCL_RUNTIME_PREPARATION54_V1.json','dcl_dependency_install54_v1.stdout.log','dcl_dependency_install54_v1.stderr.log','DCL_NUMPY_RUNTIME_AMENDMENT54_V1.json','dcl_numpy_install54_v1.stdout.log','dcl_numpy_install54_v1.stderr.log']:add(OUT/name)
    add(OUT/'dcl_preflight_v2/summary.json');add(OUT/'dcl_preflight_v3/summary.json');add(OUT/'SPECTRAL_CRITIC_SEMANTICS54_V1.json')
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
    supplemental_seal=OUT/'em_option_learning_v1/SELECTION_SEAL54_EM_TRUE_V1.json'
    if supplemental_seal.exists():add(supplemental_seal)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat();progress={'utc':now,'status':'PRIMARY_SOURCE_BOUNDARIES_AND_ANALYTIC_SPECTRAL_SEMANTICS_RECORDED','primary_completed':len(counts[0]),'supplemental_completed':len(counts[1]),'planned_C54_fits':16,'DCL_formal_fits':0,'DCL_discarded_updates':64,'C54_heldout_opened':False,'accepted_directions':0,'overall_goal':'ACTIVE_UNACHIEVED','local_result_files_written':0,'completed_C54_fits':counts}
    p=dest/'PROGRESS54_V10.json';p.write_text(json.dumps(progress,indent=2),encoding='utf-8');add(p,p.name)
    text='# C54: matched learning comparison pending; novelty boundaries tightened\n\n'+f"Snapshot UTC {now}: {len(counts[0])}/12 primary and {len(counts[1])}/4 existing-EM-option fits complete. No heldout score has opened.\n\n"
    text+='The [primary-source and implementation review](INFERENCE_AND_SPECTRAL_BOUNDARY54_V1.md) rules out generic approximation-bias and spectral-head claims as stand-alone innovations. A deterministic fixed-feature fixture shows why a spectral critic cannot automatically be rolled out as a physical conditional-mean transition. This is known objective semantics, not a trained SelfTICA failure or a new method. SelfTICA has0 executed fits here.\n\n'
    text+='The [risk interpretation protocol](CAUSAL_RISK_INTERPRETATION54_V1.md) waits for all16 frozen models/maps and completed scoring. DCL remains [64 discarded preflight updates](DCL_PREFLIGHT_REPORT54_V2.md),0 formal fits. The [source EM option comparison](EM_OPTION_LEARNING54_V1.md) remains unchanged, and all prior evidence is preserved.\n\nNo new research direction or method benefit is claimed. The overarching discovery goal remains active.\n'
    p=dest/'README.md';p.write_text(text,encoding='utf-8');add(p,p.name)
    rows=[]
    for public,p in sorted(candidates.items()):
        original=p.read_bytes();content=original.decode('utf-8-sig');b=content.encode('utf-8')
        if old.get(public)==sha(b):continue
        assert len(b)<350000 and not any(t in public.split('/') for t in ['.venv_dcl','Lib','site-packages','__pycache__'])
        target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        rows.append({'path':public,'bytes':len(b),'sha256':sha(b),'original_sha256':sha(original),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'parts':math.ceil(len(content)/3500),'publication_transform':'Remove UTF8 BOM' if original.startswith(b'\xef\xbb\xbf') else None,'server_snapshot_file':str(target)})
    manifest={'utc':now,'status':'IMMUTABLE_PRIMARY_SOURCE_BOUNDARY_AND_C54_CLOSED_FIT_PROGRESS','files':rows,'file_count':len(rows),'bytes':sum(r['bytes'] for r in rows),'total_content_parts':sum(r['parts'] for r in rows),'excluded':'Virtual environments, installed dependencies, upstream package bodies, model weights, raw signals, changing logs and private transport helpers.'}
    p=dest/'manifest.json';p.write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'files':len(rows),'bytes':manifest['bytes'],'parts':manifest['total_content_parts'],'manifest_sha256':sha(p.read_bytes()),'primary_completed':len(counts[0]),'supplemental_completed':len(counts[1])}),flush=True)
if __name__=='__main__':main()
