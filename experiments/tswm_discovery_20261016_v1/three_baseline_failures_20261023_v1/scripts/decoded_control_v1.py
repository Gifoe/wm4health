"""Six registered controls using the author's existing decoded MSE, no new model."""
import sys,pathlib,json,hashlib,difflib
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=pathlib.Path('D:/baseline_reproduction_seed2021_v1')
sys.path.insert(0,str(BASE/'scripts'))
seed=int(sys.argv[1]);horizon=int(sys.argv[2]);assert seed in (42,43,44) and horizon in (336,720)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
REPLAY_ROOT=pathlib.Path(cfg['artifact_root'])/ROOT.name/'decoded_control_v1';REPLAY_ROOT.mkdir(exist_ok=True,parents=True)
assert REPLAY_ROOT.drive.upper()=='G:'
out=REPLAY_ROOT/f'Electricity_DLinear_H{horizon}_seed{seed}';assert not out.exists()
source=BASE/'scripts/run_latenttsf.py';old=source.read_text(encoding='utf-8-sig');new=old
changes=[("out=ROOT/'results/latenttsf'/f'{dataset}_{backbone}_H{horizon}_seed42'","out=REPLAY_ROOT/f'{dataset}_{backbone}_H{horizon}_seed{REPLAY_SEED}'",1),
 ("'seed':42","'seed':REPLAY_SEED",2),("(ROOT/'patches')","(REPLAY_ROOT/'patches')",1),("(ROOT/'patches/latenttsf_audit_resume.diff')","(REPLAY_ROOT/'patches/latenttsf_audit_resume.diff')",1),
 ("'mse_weight':10.,'cosine_weight':15.,'perceptual_weight':0.","'mse_weight':0.,'cosine_weight':0.,'perceptual_weight':1.",1),
 ("  state['initial_ae_sha']=tensor_sha(enc)","  state['initial_ae_sha']=tensor_sha(enc)\n  write_json(out/'INITIALIZATION_V1.json',dict(model_tensor_sha256=tensor_sha(g['model']),ae_tensor_sha256=tensor_sha(enc),steps_per_epoch=len(g['train_loader']),torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),old_initial_tensors_not_recorded=True))",1)]
for a,b,n in changes:assert new.count(a)==n,(a,new.count(a));new=new.replace(a,b)
stop='                print("Early stopping")\n                _audit_snapshot(locals())\n                break'
assert new.count(" patch=''.join")==1
new=new.replace(" patch=''.join"," assert patched.count("+repr(stop)+")==1\n patched=patched.replace("+repr(stop)+", "+repr(stop.rsplit('break',1)[0]+'pass')+")\n patch=''.join")
out.mkdir();diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='qualified/run_latenttsf.py',tofile='decoded_control/run_latenttsf.py'))
(out/'driver_diff_v1.patch').write_text(diff,encoding='utf-8')
paths=[BASE/'upstream/LatentTSF'/x for x in ['my_train.py','my_utils.py','my_AE.py','models/DLinear.py','data_provider/data_loader.py']]
before={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(out/'PROVENANCE_V1.json').write_text(json.dumps(dict(seed=seed,horizon=horizon,protocol_sha256=hashlib.sha256((ROOT/'OFFICIAL_DECODED_CONTROL_PROTOCOL_V1.md').read_bytes()).hexdigest(),qualified_driver_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),driver_diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),upstream_source_sha256=before,objective='official decoded observation MSE only',fixed_epochs=100,new_method=False,independent_method_confirmation=False),indent=2),encoding='utf-8')
ns=dict(__name__='qualified_decoded_driver',__file__=str(source),REPLAY_ROOT=REPLAY_ROOT,REPLAY_SEED=seed)
exec(compile(new,str(source)+'::decoded_control_adapter','exec'),ns);ns['execute']('Electricity','DLinear',horizon)
after={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};assert after==before
m=json.loads((out/'metrics.json').read_text(encoding='utf-8-sig'));assert m['epochs']==100
m.update(run_type='official_decoded_loss_fixed100_attribution',status='CONTROL_NOT_ORIGINAL_RECIPE',early_stop_break_ignored=True,new_method=False,loss_weights=dict(mse=0,cosine=0,decoded=1,reconstruction=0))
(out/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
(out/'SOURCE_UNCHANGED_AFTER_V1.json').write_text(json.dumps(dict(upstream_unchanged=True,source_sha256=after),indent=2),encoding='utf-8')
print(json.dumps(dict(status='OFFICIAL_DECODED_CONTROL_COMPLETE',seed=seed,horizon=horizon,mse=m['reproduced_mse'])),flush=True)
