"""Extract only author's validation-selected final-budget weights for all24 runs."""
import hashlib,json,pathlib,tarfile
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/external/explore_context/parallel_attempt2';OUT=ROOT/'outputs/native_context_weights'
assert not OUT.exists();OUT.mkdir(parents=True)
inventory=json.loads((ROOT/'outputs/native_context_metadata/inventory.json').read_text())
selection={}
for row in inventory['metadata']:
    if not row['member'].endswith('/config.json'):continue
    cfg=row['content'];run=pathlib.PurePosixPath(row['member']).parent;member=str(run/'models'/f"checkpoint_step_{cfg['n_training_steps']}_best.pkl")
    assert any(x['name']==member for x in inventory['weights']);selection[member]=dict(config=cfg,run=run.as_posix(),metadata_sha256=row['sha256'])
assert len(selection)==24
receipt=json.loads((BASE/'download_receipt.json').read_text());rows=[]
with tarfile.open(BASE/receipt['archive'],'r:gz') as tf:
    for m in tf:
        if m.name not in selection:continue
        assert m.isfile() and m.size<40*1024*1024
        relative=pathlib.PurePosixPath(m.name);assert not relative.is_absolute() and '..' not in relative.parts
        target=(OUT/'files'/pathlib.Path(*relative.parts)).resolve();assert target.is_relative_to(OUT.resolve())
        b=tf.extractfile(m).read();target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        rows.append(dict(member=m.name,path=target.relative_to(OUT).as_posix(),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),**selection[m.name]))
assert len(rows)==24
(OUT/'manifest.json').write_text(json.dumps(dict(archive_sha256=receipt['sha256'],selection='Each author corrected validation-best checkpoint at published final training budget; no new evaluation used.',models=rows),indent=2),encoding='utf-8')
print(json.dumps(dict(models=len(rows),weight_bytes=sum(r['bytes'] for r in rows))))
