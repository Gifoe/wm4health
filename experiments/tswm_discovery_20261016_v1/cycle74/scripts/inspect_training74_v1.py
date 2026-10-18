import sys,pathlib,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common74_v1 import *
from scipy.io import whosmat
for name in ['EMPS','F16']:
 p=OUT/('DATA_'+name+'74_V2.json')
 if not p.exists():p=OUT/('DATA_'+name+'74_V1.json')
 if not p.exists():print(json.dumps(dict(dataset=name,status='SOURCE_RECEIPT_PENDING')),flush=True);continue
 r=read(p)
 for row in r['training_members']:
  assert sha(row['path'])==row['sha256']
  meta=whosmat(row['path']);print(json.dumps(dict(dataset=name,member=row['member'],variables=[dict(name=k,shape=list(shape),dtype=typ) for k,shape,typ in meta])),flush=True)
