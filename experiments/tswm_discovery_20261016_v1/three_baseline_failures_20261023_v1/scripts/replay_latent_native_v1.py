"""Reuse the qualified native audit driver; only seed and new output roots change."""
import sys,pathlib,json,hashlib,difflib,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=pathlib.Path('D:/baseline_reproduction_seed2021_v1')
sys.path.insert(0,str(BASE/'scripts'))
seed=int(sys.argv[1]);horizon=int(sys.argv[2]);assert seed in (42,43,44) and horizon in (336,720)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
REPLAY_ROOT=pathlib.Path(cfg['artifact_root'])/ROOT.name/'native_replay_v1'
assert REPLAY_ROOT.drive.upper()=='G:'
REPLAY_ROOT.mkdir(exist_ok=True,parents=True)
source=BASE/'scripts/run_latenttsf.py';raw=source.read_bytes();old=raw.decode('utf-8-sig')
out=REPLAY_ROOT/f'Electricity_DLinear_H{horizon}_seed{seed}'
assert not out.exists(),('Existing run; inspect rather than overwrite/restart',out.name)
new=old
changes=[("out=ROOT/'results/latenttsf'/f'{dataset}_{backbone}_H{horizon}_seed42'", "out=REPLAY_ROOT/f'{dataset}_{backbone}_H{horizon}_seed{REPLAY_SEED}'",1),
         ("'seed':42","'seed':REPLAY_SEED",2),
         ("(ROOT/'patches')","(REPLAY_ROOT/'patches')",2)]
for before,after,n in changes:
    assert new.count(before)==n,(before,new.count(before),n)
    new=new.replace(before,after)
patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='qualified/run_latenttsf.py',tofile='seed_and_storage_adapter/run_latenttsf.py'))
meta=REPLAY_ROOT/'adapter_provenance';meta.mkdir(exist_ok=True)
patchfile=meta/'driver_diff_v1.patch'
if patchfile.exists():assert patchfile.read_text(encoding='utf-8')==patch
else:patchfile.write_text(patch,encoding='utf-8')
paths=[BASE/'upstream/LatentTSF'/x for x in ['my_train.py','my_utils.py','models/DLinear.py','data_provider/data_loader.py','run_train_all.sh']]
before={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
record=dict(seed=seed,horizon=horizon,qualified_driver_sha256=hashlib.sha256(raw).hexdigest(),driver_diff_sha256=hashlib.sha256(patch.encode()).hexdigest(),upstream_source_sha256=before,
            changes=['Forecast initialization seed','Fresh output directory','Fresh audit-patch directory'],source_objective_or_model_changed=False,
            recipe=dict(seq_len=720,step=60,batch_size=32,learning_rate=.001,train_epochs=100,patience=5,latent_mse_weight=10,latent_cosine_weight=15),
            scientific_identity='native source-recipe replication, with retained existing nonarchitectural audit hooks')
out.mkdir()
(out/'REPLAY_PROVENANCE_V1.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
ns={'__name__':'qualified_replay_driver','__file__':str(source),'REPLAY_ROOT':REPLAY_ROOT,'REPLAY_SEED':seed}
exec(compile(new,str(source)+'::seed_storage_adapter','exec'),ns)
ns['execute']('Electricity','DLinear',horizon)
after={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
assert after==before
metric=json.loads((out/'metrics.json').read_text())
assert metric['seed']==seed and metric['horizon']==horizon
(out/'SOURCE_UNCHANGED_AFTER_V1.json').write_text(json.dumps(dict(source_sha256=after,upstream_unchanged=True),indent=2),encoding='utf-8')
print(json.dumps({'status':'NATIVE_REPLAY_COMPLETE','seed':seed,'horizon':horizon,'mse':metric['reproduced_mse'],'mae':metric['reproduced_mae']}),flush=True)
