"""Numeric NPZ/JSON checkpoint backend; bypass Tensor pickle reducers."""
import pathlib,json,numpy as np,torch
def save_checkpoint(value,path):
 path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp');assert not path.exists() and not tmp.exists();arrays={}
 def encode(v):
  if isinstance(v,torch.Tensor):
   key=f'tensor_{len(arrays)}';arrays[key]=v.detach().cpu().numpy().copy();assert arrays[key].dtype.kind!='O';return ['tensor',key]
  if isinstance(v,np.ndarray):
   key=f'array_{len(arrays)}';arrays[key]=v.copy();assert arrays[key].dtype.kind!='O';return ['array',key]
  if isinstance(v,dict):return ['dict',[[encode(k),encode(x)] for k,x in v.items()]]
  if isinstance(v,list):return ['list',[encode(x) for x in v]]
  if isinstance(v,tuple):return ['tuple',[encode(x) for x in v]]
  if isinstance(v,np.generic):v=v.item()
  assert v is None or isinstance(v,(str,int,float,bool)),type(v);return ['scalar',v]
 tree=encode(value);arrays['metadata_json_utf8']=np.frombuffer(json.dumps(tree,allow_nan=False).encode(),dtype='uint8')
 with tmp.open('xb') as f:np.savez_compressed(f,**arrays)
 assert tmp.resolve().parent==path.resolve().parent;tmp.replace(path)
def load_checkpoint(path,map_location='cpu',weights_only=False):
 path=pathlib.Path(path)
 if path.suffix!='.npz':return torch.load(path,map_location=map_location,weights_only=weights_only)
 with np.load(path,allow_pickle=False) as z:
  tree=json.loads(z['metadata_json_utf8'].tobytes())
  def decode(v):
   tag,payload=v
   if tag=='tensor':return torch.from_numpy(z[payload].copy()).to(map_location)
   if tag=='array':return z[payload].copy()
   if tag=='dict':return {decode(k):decode(x) for k,x in payload}
   if tag=='list':return [decode(x) for x in payload]
   if tag=='tuple':return tuple(decode(x) for x in payload)
   assert tag=='scalar';return payload
  return decode(tree)
