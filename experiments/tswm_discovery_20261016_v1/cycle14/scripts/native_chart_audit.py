"""Trained native representation-chart and encoder-cost discriminator."""
import runtime_profile
import datetime,hashlib,json,pathlib,sys,time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs/native_chart'
sys.path.insert(0,str(PROJECT/'upstream/WorldTS/DAG-bench'))
from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def fingerprint(m):
 h=hashlib.sha256()
 for key,v in sorted(m.state_dict().items()):h.update(key.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def encode(m,hist,stats=None):
 s=m._compute_revin_stats(hist) if stats is None else stats
 x=(hist-s['mean'])/s['stdev'] if m.use_revin else hist
 a=m.target_encoder(x[...,:m.target_dim])
 b=m.condition_encoder(x[...,m.target_dim:]) if m.condition_encoder is not None else None
 return a,b,s

def difference(a,b):
 d=(a-b).flatten(1);af=a.flatten(1);bf=b.flatten(1)
 return {'relative_L2_mean':float((d.norm(dim=1)/bf.norm(dim=1).clamp_min(1e-12)).mean()),
         'absolute_max':float(d.abs().max()),'cosine_mean':float(torch.nn.functional.cosine_similarity(af,bf,dim=1).mean())}

def timing(fn):
 for _ in range(8):fn()
 torch.cuda.synchronize();rows=[]
 for _ in range(3):
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  start.record();t=time.perf_counter()
  for _ in range(40):fn()
  end.record();torch.cuda.synchronize()
  rows.append({'synchronized_wall_ms_per_call':(time.perf_counter()-t)*1000/40,'CUDA_event_ms_per_call':start.elapsed_time(end)/40})
 return rows

def main():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'results.json').exists()
 torch.set_num_threads(1);base=pathlib.Path('D:/baseline_reproduction_seed2021_v1')
 provenance=read(PROJECT/'outputs/existing_future_support.json')['provenance'];rows=[];timings=[];sources=[]
 for dataset in ['NP','PJM']:
  for horizon in [24,360]:
   record=next(r for r in provenance if r['dataset']==dataset and r['horizon']==horizon)
   folder=base/'results/worldts'/f'{dataset}_H{horizon}_seed2021';cp=folder/'stages/stage2_decoder_mix.pt';conf=folder/'resolved_config.json';data=base/'data'/f'{dataset}.csv'
   assert sha(cp)==record['checkpoint_sha256'] and sha(conf)==record['resolved_config_sha256'] and sha(data)==record['data_sha256']
   unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(cp);assert not unsafe,unsafe
   state=torch.load(cp,map_location='cpu',weights_only=True)['Model'];cfg=read(conf)
   m=WorldTSModel(SimpleNamespace(**cfg)).cuda().eval();m.load_state_dict(state,strict=True)
   for p in m.parameters():p.requires_grad_(False)
   before=fingerprint(m);frame=pd.read_csv(data);names=frame['cols'].drop_duplicates().tolist();assert names[-1]=='OT'
   frame['date']=pd.to_datetime(frame['date']);wide=frame.pivot(index='date',columns='cols',values='data')[names]
   raw=wide[['OT']+names[:-1]].to_numpy();train_end=int(int(len(raw)*.8)*.875)
   mean=raw[:train_end].mean(0);std=raw[:train_end].std(0).clip(1e-6);values=((raw-mean)/std).astype(np.float32)
   length=cfg['seq_len'];origins=length+np.arange(32)*8;assert int(origins[-1]+horizon+24)<train_end
   hist=torch.tensor(np.stack([values[t-length:t] for t in origins]),device='cuda');future=torch.tensor(np.stack([values[t:t+horizon,1:] for t in origins]),device='cuda')
   arrays={'origins':origins,'historical_inputs':hist.cpu().numpy(),'future_covariates':future.cpu().numpy(),'train_mean':mean,'train_std':std}
   with torch.inference_mode():
    a,c,old=encode(m,hist);arrays.update(old_target_states=a.cpu().numpy(),old_mean=old['mean'].cpu().numpy(),old_stdev=old['stdev'].cpu().numpy())
    if c is not None:arrays['old_condition_states']=c.cpu().numpy()
    for shift in [1,8,24]:
     if shift+m.state_patch_len>=length:continue
     moved=torch.tensor(np.stack([values[t+shift-length:t+shift] for t in origins]),device='cuda')
     na,nc,new=encode(m,moved);fa,fc,_=encode(m,moved,old);cut=m.state_patch_len-1
     old_slice=slice(shift+cut,length);new_slice=slice(cut,length-shift)
     # Exact raw-patch overlap, excluding new-window replicate padding.
     assert torch.equal(hist[:,shift+cut:],moved[:,cut:length-shift])
     item={'dataset':dataset,'horizon':horizon,'shift':shift,'overlap_tokens_per_origin':length-shift-cut,
       'target_current_chart_difference':difference(a[:,old_slice],na[:,new_slice]),
       'target_fixed_chart_control':difference(a[:,old_slice],fa[:,new_slice])}
     assert item['target_fixed_chart_control']['absolute_max']<2e-5
     if c is not None:
      item['condition_current_chart_difference']=difference(c[:,old_slice],nc[:,new_slice]);item['condition_fixed_chart_control']=difference(c[:,old_slice],fc[:,new_slice])
      assert item['condition_fixed_chart_control']['absolute_max']<2e-5
     arrays.update({f'inputs_shift{shift}':moved.cpu().numpy(),f'target_shift{shift}':na.cpu().numpy(),f'target_fixed_shift{shift}':fa.cpu().numpy(),f'mean_shift{shift}':new['mean'].cpu().numpy(),f'std_shift{shift}':new['stdev'].cpu().numpy()})
     if nc is not None:arrays.update({f'condition_shift{shift}':nc.cpu().numpy(),f'condition_fixed_shift{shift}':fc.cpu().numpy()})
     rows.append(item)
    for batch in [1,8]:
     x=hist[:batch];u=future[:batch];full=timing(lambda:m.forecast(x,x_future=u));enc=timing(lambda:encode(m,x))
     fullwall=np.median([r['synchronized_wall_ms_per_call'] for r in full]);encwall=np.median([r['synchronized_wall_ms_per_call'] for r in enc])
     timings.append({'dataset':dataset,'horizon':horizon,'batch':batch,'full_forecast':full,'history_encoders_and_stats':enc,
       'encoder_wall_fraction':float(encwall/fullwall),'encoder_elimination_ideal_speedup_upper_bound':float(1/(1-encwall/fullwall)) if encwall<fullwall else None})
    out=m.forecast(hist[:8],x_future=future[:8]);assert torch.isfinite(out).all();arrays['reference_forecast_first8']=out.cpu().numpy()
   after=fingerprint(m);assert before==after==record['model_state_sha256_before']
   np.savez_compressed(OUT/f'{dataset}_H{horizon}_arrays.npz',**arrays)
   sources.append({'dataset':dataset,'horizon':horizon,'config':cfg,'checkpoint_sha256':sha(cp),'config_sha256':sha(conf),'data_sha256':sha(data),'unsafe_globals':unsafe,
     'train_boundary':train_end,'channel_order':['OT']+names[:-1],'weights_unchanged':True,'weights_sha256':before,'training_origins':origins.tolist()})
   print(json.dumps({'completed_native_model':dataset,'horizon':horizon,'use_revin':m.use_revin,'encoder_wall_fractions':[r['encoder_wall_fraction'] for r in timings[-2:]]}),flush=True)
   del m;torch.cuda.empty_cache()
 files=[]
 for module in tuple(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if name:
   p=pathlib.Path(name)
   if p.is_file() and p.suffix=='.py' and p.resolve().is_relative_to((PROJECT/'upstream/WorldTS').resolve()):files.append({'path':p.relative_to(PROJECT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
 result={'rows':rows,'timings':timings,'native_models':sources,'executed_source_files':sorted(files,key=lambda r:r['path']),
   'native_source_revision':'269426ea6217d7152a244da0e209d922acd46d4f','plan_sha256':sha(ROOT/'NATIVE_CHART_AUDIT_PLAN.md'),'script_sha256':sha(pathlib.Path(__file__)),
   'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'cudnn_tf32':torch.backends.cudnn.allow_tf32},
   'scope':'Frozen trained native components on training contexts;no causal/forecast-accuracy failure or new-method claim',
   'scientific_complete':True,'objective_achieved':False,'novel_direction_accepted':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'native_models':len(sources),'chart_rows':len(rows),'timing_rows':len(timings),'objective_achieved':False}))

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 class Tee:
  def __init__(self,original,log):self.original=original;self.log=log
  def write(self,value):self.original.write(value);self.log.write(value)
  def flush(self):self.original.flush();self.log.flush()
 with (OUT/'stdout.log').open('x',encoding='utf-8') as log:
  sys.stdout=Tee(sys.__stdout__,log)
  try:main()
  finally:sys.stdout.flush();sys.stdout=sys.__stdout__
