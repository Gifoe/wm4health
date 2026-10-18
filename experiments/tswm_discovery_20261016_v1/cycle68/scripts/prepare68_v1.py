import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common68_v1 import *
rows=[];base=storage('data')
for si,subset in enumerate(['VAL','EVAL']):
 for fi,family in enumerate(['linear','cubic']):
  for i in range(6):
   seed=680100+1000*si+100*fi+i;raw,params=generate(seed,family);case=f'{subset}_{family}_{seed}';p=base/(case+'.npz');assert not p.exists();np.savez_compressed(p,raw=raw,history=raw[:,:L],known=raw[1,L:],target=raw[0,L:]);rows.append(dict(id=case,subset=subset,family=family,seed=seed,params=params,path=str(p),sha256=sha(p),shape=list(raw.shape),history_shape=[2,L],horizon=H))
dump(OUT/'CASE_MANIFEST68_V1.json',dict(utc=utc(),status='FROZEN_GENERATED_CASES',cases=rows,protocol_sha256=sha(ROOT/'PROTOCOL68_V1.md'),source_sha256=sha(__file__),L=L,H=H,neural_training=False));print(json.dumps(dict(status='FROZEN_GENERATED_CASES',cases=len(rows))),flush=True)
