"""Create the user-authorized large-disk project directory and verify data copies."""
import datetime
import hashlib
import json
import pathlib
import shutil

R=pathlib.Path(__file__).resolve().parents[1]
PROJECT=R.parent
PARENT=pathlib.Path('G:/ICLR Omni-eeg cache')
assert PARENT.is_dir(), 'Observed user-designated Omni-eeg directory must exist'
DEST=PARENT/'tswm_discovery_20261016_v1'
assert DEST.resolve().parent == PARENT.resolve()
RECEIPT=R/'outputs/storage_v1.json'
assert not RECEIPT.exists(), 'Version prior evidence'
for folder in ['data','checkpoints','caches','artifacts','manifests']:
    (DEST/folder).mkdir(parents=True,exist_ok=True)
acq=json.loads((R/'outputs/acquisition_v2/receipt.json').read_text())
assert acq['status']=='COMPLETE_ORIGINAL_TIMESERIES_ONLY'
records=[]
for archive in acq['archives']:
    for item in archive['files']:
        source=R/item['local_path']
        data=source.read_bytes()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        relative=source.relative_to(R/'data')
        target=DEST/'data'/relative
        assert target.resolve().is_relative_to((DEST/'data').resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256'], 'Never replace conflicting data'
        else:
            shutil.copyfile(source,target)
        assert target.stat().st_size==item['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']
        records.append({'source':str(source),'destination':str(target),'bytes':item['bytes'],'sha256':item['sha256']})
config={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'LARGE_DISK_STORAGE_CONFIGURED_AND_DATA_COPY_VERIFIED','project_code_root':str(PROJECT),'storage_root':str(DEST),'data_root':str(DEST/'data'),'checkpoint_root':str(DEST/'checkpoints'),'cache_root':str(DEST/'caches'),'artifact_root':str(DEST/'artifacts'),'selection_note':'User requested 15TB disk under existing omnieeg folder; observed matching G: root is named ICLR Omni-eeg cache. G: 14.553 TiB, 7.060 TiB free at discovery.','policy':'New downloaded datasets, large caches and future checkpoints use this storage root. Existing completed-run paths remain intact. Original D: copy is retained for source audit provenance; no deletion or bulk relocation performed.','verified_files':len(records),'verified_bytes':sum(x['bytes'] for x in records),'files':records}
text=json.dumps(config,indent=2)
for path in [RECEIPT,PROJECT/'STORAGE_LOCATION_V1.json',DEST/'manifests/STORAGE_LOCATION_V1.json']:
    assert not path.exists(),str(path)
    path.write_text(text,encoding='utf-8')
for source,filename in [(R/'outputs/acquisition_v2/receipt.json','greenhouse3_acquisition_v1.json'),(R/'PROTOCOL.md','C52_PROTOCOL.md')]:
    path=DEST/'manifests'/filename
    assert not path.exists()
    shutil.copyfile(source,path)
    assert source.read_bytes()==path.read_bytes()
usage=shutil.disk_usage(DEST)
print(json.dumps({'status':config['status'],'storage_root':str(DEST),'verified_files':len(records),'verified_bytes':config['verified_bytes'],'disk_total_TiB':usage.total/2**40,'disk_free_TiB':usage.free/2**40}),flush=True)
