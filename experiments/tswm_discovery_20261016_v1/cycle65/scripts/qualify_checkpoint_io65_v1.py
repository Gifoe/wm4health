from models65 import *
from checkpoint_io65_v1 import save_checkpoint,load_checkpoint
pp=paths();files=sorted((pp['checkpoint_root']/'recovery_v1'/'nonlinear_s2023'/'CSDI_native_coverage').glob('attempt*/step*.pt'),key=lambda p:int(p.stem[4:]));assert files;src=files[-1];z=load_checkpoint(src,map_location='cpu',weights_only=False);p=pp['artifact_root']/'CHECKPOINT_IO_FIXTURE65_V1.npz';save_checkpoint(z,p);back=load_checkpoint(p,map_location='cpu');count=0
def compare(a,b):
 global count
 if isinstance(a,torch.Tensor):assert torch.equal(a,b);count+=1
 elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
 elif isinstance(a,dict):assert list(a)==list(b);[compare(a[k],b[k]) for k in a]
 elif isinstance(a,(list,tuple)):assert type(a)==type(b) and len(a)==len(b);[compare(x,y) for x,y in zip(a,b)]
 else:assert a==b
compare(z,back);dump(OUT/'CHECKPOINT_IO_QUALIFICATION65_V1.json',dict(utc=utc(),status='PASS',source_checkpoint=str(src),source_sha256=sha(src),source_step=z['step'],fixture=str(p),fixture_sha256=sha(p),tensor_fields_checked=count,model_parameters_and_optimizer_and_RNG_bitwise_preserved=True,format='Numeric NPZ with JSON schema; allow_pickle=False; no torch.save or Tensor pickle reducers'));print(json.dumps(dict(status='PASS',step=z['step'],tensor_fields=count)))
