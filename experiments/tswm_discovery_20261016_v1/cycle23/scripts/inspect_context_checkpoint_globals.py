import runtime_profile
import json,pathlib,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];m=json.loads((ROOT/'outputs/native_context_weights/manifest.json').read_text())
rows=[]
for r in m['models']:
    f=ROOT/'outputs/native_context_weights'/r['path'];rows.append(dict(path=r['path'],unsafe_globals=torch.serialization.get_unsafe_globals_in_checkpoint(f)))
(ROOT/'outputs/native_context_checkpoint_globals.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(models=len(rows),distinct=sorted(set(x for r in rows for x in r['unsafe_globals'])))))
