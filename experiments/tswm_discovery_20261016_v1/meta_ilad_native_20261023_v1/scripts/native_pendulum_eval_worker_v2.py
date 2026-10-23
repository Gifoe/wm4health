"""Execute unchanged source evaluators; capture their actual returned images on G."""
import pathlib,sys,argparse,json,hashlib,datetime,importlib.util,types,traceback
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['paper','refined'],required=True);p.add_argument('--split',choices=['test','ood','trainenv'],required=True);args=p.parse_args()
name=args.variant+'_'+args.split;OUT=ROOT/'pendulum_native_remaining_v1'/name;OUT.mkdir(parents=True,exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\pendulum_native_remaining_v1')/name;ART.mkdir(parents=True,exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
def sha_file(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data_receipt=json.loads((ROOT/'pendulum_data_generation_v1/RESULTS_V1.json').read_text());cell=next(x for x in data_receipt['sets'] if x['split']==args.split);data=pathlib.Path(cell['path']);assert sha_file(data)==cell['file_sha256']
 acquisition=json.loads((ROOT/'pendulum_acquisition_recovery_v2/RESULTS_V1.json').read_text())
 for row in acquisition['files']:assert sha_file(SRC/row['path'])==row['sha256']
 code=SRC/'pendulum'/('paper_version/meta_ilad' if args.variant=='paper' else 'meta_ilad');sys.path.insert(0,str(code))
 spec=importlib.util.spec_from_file_location('_original_pendulum_evaluator',code/'evaluate.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e);base=e.iLaD
 n=cell['sequences'];shape=(n,40,32,32)
 rec=np.lib.format.open_memmap(ART/'reconstructions.npy',mode='w+',dtype=np.float32,shape=shape);pred=np.lib.format.open_memmap(ART/'predictions.npy',mode='w+',dtype=np.float32,shape=shape)
 counts={'rec':0,'pred':0};instances=[]
 class Capture(base):
  def load_state_dict(self,state,*a,**kw):
   result=super().load_state_dict(state,*a,**kw)
   assert all(torch.isfinite(v).all() for v in self.state_dict().values());assert set(state)==set(self.state_dict())
   for k,v in state.items():assert torch.equal(v,self.state_dict()[k])
   self.loaded_digest=digest(self);instances.append(self);return result
  def evaluate_R(self,xt,ctx):
   out=super().evaluate_R(xt,ctx);i=counts['rec'];rec[i:i+len(xt)]=out[0].detach().cpu().numpy();counts['rec']+=len(xt);return out
  def evaluate_TG(self,xt,ctx):
   out=super().evaluate_TG(xt,ctx);i=counts['pred'];pred[i:i+len(xt)]=out[0].detach().cpu().numpy();counts['pred']+=len(xt);return out
 e.iLaD=Capture
 ckpt=SRC/'pendulum/checkpoints/meta_ilad_s1000.pt';native_out=ART/'native'
 e.eval_model(types.SimpleNamespace(ckpt=str(ckpt),data=str(data),seed=1000,out=str(native_out)))
 assert counts=={'rec':n,'pred':n} and len(instances)==1;model=instances[0];assert digest(model)==model.loaded_digest
 rec.flush();pred.flush();native_metrics=json.loads(native_out.with_suffix('.json').read_text());dump('NATIVE_METRICS_V1.json',native_metrics)
 result=dict(status='COMPLETE_UNCHANGED_NATIVE_EVALUATOR_EXECUTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),payload_group='pendulum_native_remaining_v1',variant=args.variant,split=args.split,seed=1000,sequences=n,native_metrics=native_metrics,checkpoint_sha256=sha_file(ckpt),loaded_state_digest=model.loaded_digest,final_state_digest=digest(model),source_revision=acquisition['source_revision'],data_sha256=cell['file_sha256'],payloads={p.name:dict(bytes=p.stat().st_size,sha256=sha_file(p)) for p in ART.iterdir() if p.is_file()},new_fits=0,local_payload_sync=False)
 assert digest(model)==model.loaded_digest;dump('RESULTS_V1.json',result);print(json.dumps(dict(status=result['status'],variant=args.variant,split=args.split,native_metrics=native_metrics)),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
