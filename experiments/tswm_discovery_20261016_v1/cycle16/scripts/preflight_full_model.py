"""Verify source/data and native full-model forward/backward before long fitting."""
import runtime_profile
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'outputs';out.mkdir(parents=True,exist_ok=True)
 source=PROJECT/'cycle15/upstream/MTS3';manifest=json.loads((PROJECT/'cycle15/outputs/source/manifest.json').read_text())
 for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
 import torch,pandas as pd,numpy as np
 from config_tools import load
 sys.path.insert(0,str(source))
 from agent.worldModels.MTS3 import MTS3
 cfg=load(source/'experiments/mobileRobot/conf/model/default_mts3.yaml')
 torch.set_num_threads(1);torch.manual_seed(2021);torch.cuda.manual_seed_all(2021)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 start=time.perf_counter();model=MTS3(input_shape=(1,),action_dim=6,config=cfg).cuda()
 obs=torch.randn(2,225,1,device='cuda');obs[:,151:]=0
 act=torch.randn(2,225,6,device='cuda')
 mask=torch.zeros(2,225,1,dtype=torch.bool,device='cuda');mask[:,:151]=True
 mean,var=model(obs,act,mask)[:2]
 assert mean.shape==var.shape==(2,225,1) and torch.isfinite(mean).all() and torch.isfinite(var).all() and (var>0).all()
 target=torch.randn(2,75,1,device='cuda')
 loss=.5*(torch.log(2*torch.pi*var[:,150:225])+(mean[:,150:225]-target)**2/var[:,150:225]);loss.mean().backward()
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
 model.eval()
 with torch.inference_mode():
  base=model(obs,act,mask)[:2]
  altered=obs.clone();altered[:,151:]=37*torch.randn_like(altered[:,151:])
  changed=model(altered,act,mask)[:2]
  errors=[float((a[:,150:225]-b[:,150:225]).abs().max()) for a,b in zip(base,changed)]
  assert max(errors)==0
 data=pathlib.Path('D:/baseline_reproduction_seed2021_v1/data/NP.csv')
 previous=json.loads((PROJECT/'outputs/existing_future_support.json').read_text())['provenance']
 assert sha(data)==next(row for row in previous if row['dataset']=='NP')['data_sha256']
 frame=pd.read_csv(data);names=frame['cols'].drop_duplicates().tolist();assert names[-1]=='OT'
 frame['date']=pd.to_datetime(frame['date']);wide=frame.pivot(index='date',columns='cols',values='data')[['OT']+names[:-1]]
 assert wide.index.is_monotonic_increasing and np.isfinite(wide.to_numpy()).all()
 dt=wide.index.to_series().diff().dropna().value_counts()
 payload={'source_revision':manifest['revision'],'source_files_verified':len(manifest['files']),'native_parameters':sum(p.numel() for p in model.parameters()),
          'native_forward_backward_finite':True,'future_placeholder_errors':errors,'history':151,'horizon':75,'source_steps':225,'scored_source_outputs':[150,225],
          'required_future_covariates':75,'extra_future_covariates':0,
          'data_sha256':sha(data),'N':len(wide),'channel_order':wide.columns.tolist(),'cadence_counts':{str(k):int(v) for k,v in dt.items()},
          'elapsed_seconds':time.perf_counter()-start,'environment':{'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name()},
          'plan_sha256':sha(ROOT/'FULL_MODEL_PLAN.md'),'objective_achieved':False}
 (out/'preflight.json').write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
