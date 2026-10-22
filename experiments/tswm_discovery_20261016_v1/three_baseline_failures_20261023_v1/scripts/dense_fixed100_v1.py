"""Use the qualified dense forward unchanged, for the fixed100 registry only."""
import pathlib,sys,hashlib,difflib,json
ROOT=pathlib.Path(__file__).resolve().parents[1]
mode=sys.argv[1];assert mode in ('latent','raw')
source=ROOT/'scripts'/('dense_hourly_audit_v2.py' if mode=='latent' else 'dense_raw_mode_v1.py')
old=source.read_text(encoding='utf-8-sig');new=old
changes=[("ROOT/'outputs/"+('dense_hourly_v2' if mode=='latent' else 'raw_dense_v1')+"'", "ROOT/'outputs/fixed100_dense_"+mode+"_v1'"),
 ("ART/'"+('dense_hourly_v2' if mode=='latent' else 'raw_dense_v1')+"'", "ART/'fixed100_dense_"+mode+"_v1'"),
 ("ROOT/'outputs/"+('native_replays_v4/RESULTS_V3.json' if mode=='latent' else 'raw_control_v1/RESULTS_V1.json')+"'", "ROOT/'outputs/matched_cap_v1/RESULTS_V1.json'"),
 ("assert registry['status']=='"+('ALL_SIX_NATIVE_REPLAY_ARTIFACTS_VERIFIED' if mode=='latent' else 'ALL_SIX_RAW_MODE_CONTROLS_CLOSED')+"'", "assert registry['status']=='TWELVE_FIXED100_CONTROLS_CLOSED'\nregistry['rows']=[row for row in registry['rows'] if row['mode']=="+repr(mode)+"]\nassert len(registry['rows'])==6"),
 ("sha(ROOT/'PROTOCOL.md')", "sha(ROOT/'MATCHED_CAP_PROTOCOL_V1.md')")]
for before,after in changes:
 assert new.count(before)==1,(before,new.count(before));new=new.replace(before,after)
assert json.loads((ROOT/'outputs/matched_cap_v1_exit.json').read_text())['exit_code']==0
adapter=ROOT/'outputs'/('fixed100_dense_'+mode+'_adapter_v1.json')
with adapter.open('x',encoding='utf-8') as f:json.dump(dict(source=source.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),changes='Registry, output roots and protocol reference only; same numerical forward',diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True)))),f,indent=2)
exec(compile(new,str(source)+'::fixed100_registry_adapter','exec'),dict(__name__='__main__',__file__=str(__file__)))
