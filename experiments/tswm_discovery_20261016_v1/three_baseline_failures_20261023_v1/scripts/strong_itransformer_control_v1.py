"""Predeclared compatibility follow-up using official iTransformer, no new architecture."""
import pathlib,sys,json,hashlib,difflib
ROOT=pathlib.Path(__file__).resolve().parents[1];mode=sys.argv[1];seed=int(sys.argv[2]);H=int(sys.argv[3])
assert mode in ('latent','decoded','raw') and seed in (42,43,44) and H in (336,720)
j=json.loads((ROOT/'outputs/decoded_validation_v1/RESULTS_V1.json').read_text())
assert j['decision']=='RUN_REGISTERED_ITRANSFORMER_COMPATIBILITY_CONTROLS'
assert j['protocol_sha256']==hashlib.sha256((ROOT/'OFFICIAL_DECODED_CONTROL_PROTOCOL_V1.md').read_bytes()).hexdigest()
names={'latent':'matched_cap_latent_v1.py','decoded':'decoded_control_v1.py','raw':'matched_cap_raw_v1.py'}
source=ROOT/'scripts'/names[mode];old=source.read_text(encoding='utf-8-sig');new=old
rootname={'latent':'matched_cap_latent_v1','decoded':'decoded_control_v1','raw':'matched_cap_raw_v1'}[mode]
assert new.count("'"+rootname+"'")==1
new=new.replace("'"+rootname+"'","'strong_it_"+mode+"_v1'")
assert 'DLinear' in new;new=new.replace('DLinear','iTransformer')
if mode=='raw':
 a='(484512 if H==336 else 1038240)';assert new.count(a)==1;new=new.replace(a,'(4748112 if H==336 else 4945104)')
if mode=='latent':
 a="  state['initial_ae_sha']=tensor_sha(enc)"
 b=a+"\n  write_json(out/'INITIALIZATION_V1.json',dict(model_tensor_sha256=tensor_sha(g['model']),ae_tensor_sha256=tensor_sha(enc),steps_per_epoch=len(g['train_loader'])))"
 # Inject the recording-only change into the qualified driver before its other adapters.
 anchor='new=old\n';assert new.count(anchor)==1
 new=new.replace(anchor,anchor+'assert new.count('+repr(a)+')==1\nnew=new.replace('+repr(a)+','+repr(b)+')\n')
adapter=ROOT/'outputs/strong_itransformer_v1'/f'{mode}_H{H}_seed{seed}_adapter.json'
with adapter.open('x',encoding='utf-8') as f:json.dump(dict(source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),registered_validation_decision_sha256=hashlib.sha256((ROOT/'outputs/decoded_validation_v1/RESULTS_V1.json').read_bytes()).hexdigest(),changes='Official iTransformer constructor/config, new output roots, initialization recording; same registered mode objective and100epoch cap',diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))),f,indent=2)
sys.argv=[str(source),str(seed),str(H)]
exec(compile(new,str(source)+'::registered_itransformer_adapter','exec'),dict(__name__='__main__',__file__=str(__file__)))
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));folder=pathlib.Path(cfg['artifact_root'])/ROOT.name/('strong_it_'+mode+'_v1')/f'Electricity_iTransformer_H{H}_seed{seed}'
m=json.loads((folder/'metrics.json').read_text());assert m['epochs']==100 and m['num_parameters']=={336:4748112,720:4945104}[H]
m.update(run_type='official_itransformer_fixed100_compatibility_control',mode=mode,status='CONTROL_NOT_ORIGINAL_RECIPE',new_method=False,protocol_sha256=j['protocol_sha256'])
(folder/'metrics.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
print(json.dumps(dict(status='OFFICIAL_ITRANSFORMER_CONTROL_COMPLETE',mode=mode,seed=seed,horizon=H,mse=m['reproduced_mse'])),flush=True)
