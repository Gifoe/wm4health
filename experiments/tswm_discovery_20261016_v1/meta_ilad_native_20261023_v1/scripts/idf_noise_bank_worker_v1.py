"""Actual native evaluation with recorded/replayed Gaussian draws; no source edits."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',TQDM_DISABLE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import pathlib,sys,json,argparse,runpy,hashlib
import torch,numpy as np
torch.set_num_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
ROOT=pathlib.Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['record','replay'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--bank');args=p.parse_args();assert ('cuda' if torch.cuda.is_available() else 'cpu')==args.device
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'));source=read(ROOT/'acquisition_recovery_v1/RESULTS_V1.json');src=pathlib.Path(source['source_root']);data=read(ROOT/'native_recipe_v1/DATA_QUALIFICATION_V1.json')['data_root'];storage=read(ROOT.parent/'STORAGE_LOCATION_V1.json');ART=pathlib.Path(storage['artifact_root'])/ROOT.name/'inference_parity_v1';ART.mkdir(parents=True,exist_ok=True);prefix=ART/args.tag;assert not pathlib.Path(str(prefix)+'.npz').exists()
bank=[]
if args.mode=='replay':
 with np.load(args.bank,allow_pickle=False) as z:bank=[z[k].copy() for k in sorted(z.files)]
captured=[];cursor=0;original=torch.randn_like
def randn_like(template,*a,**kw):
 global cursor
 assert not a and not kw
 generated=original(template)
 if args.mode=='record':value=generated;captured.append(generated.detach().cpu().numpy().copy())
 else:
  assert cursor<len(bank) and tuple(bank[cursor].shape)==tuple(template.shape);value=torch.as_tensor(bank[cursor],device=template.device,dtype=template.dtype)
 cursor+=1;return value
torch.randn_like=randn_like
script=src/'mocap/baselines/idf/evaluate.py';sys.path.insert(0,str(script.parent));sys.argv=[str(script),'--ckpt',str(src/'mocap/checkpoints'/f'idf_id_s{args.seed}.pt'),'--data-dir',data,'--seed',str(args.seed),'--out',str(prefix)]
try:runpy.run_path(str(script),run_name='__main__')
finally:torch.randn_like=original
assert cursor==251,cursor
if args.mode=='record':
 path=ART/(args.tag+'_draw_bank.npz')
 with path.open('xb') as f:np.savez_compressed(f,**{f'eps_{i:04d}':a for i,a in enumerate(captured)})
else:path=pathlib.Path(args.bank);assert cursor==len(bank)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=read(str(prefix)+'.json');out=dict(tag=args.tag,mode=args.mode,seed=args.seed,device=args.device,python=sys.version,torch=torch.__version__,numpy=np.__version__,tf32=False,gaussian_arrays=cursor,bank_path=str(path),bank_sha256=sha(path),payload_path=str(prefix)+'.npz',payload_sha256=sha(str(prefix)+'.npz'),native_metrics=m,source_revision=source['source_revision'],optimizer_updates=0)
dest=ROOT/'inference_parity_v1'/f'{args.tag}_V1.json'
with dest.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out),flush=True)
