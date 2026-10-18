"""All fixed C53 cells: native float32 plan-prefix and impulse-support tests."""
import sys,pathlib,time,gc,importlib.util
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common79_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71/scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
sys.path.insert(0,str(PROJECT/'cycle53/scripts'))
import native53 as old
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
from replay53 import load_compatible_cell
from common.gpu_dataset import prepare_gpu
from dataset_utils.greenhouse.dataset import ACTION_CHANNELS
sys.path.insert(0,str(old.UP/'RQ6_intervention/scripts'))
import probe_intervention as probe
probe.DEVICE='cpu';probe.AE_DIR=old.CODEC
CUTS=[4,8,12]
EDITS=[dict(kind='suffix',channel=j,cut=c,quantile=q) for j in range(7) for c in CUTS for q in [.1,.9]]+[dict(kind='impulse',channel=j,cut=0,quantile=.9) for j in range(7)]
def edit(plan,e,bounds):
 x=plan.clone();j=e['channel'];v=float(bounds[0 if e['quantile']==.1 else 1,j])
 if e['kind']=='suffix':
  x[:,e['cut']:,j]=v;assert torch.equal(x[:,:e['cut']],plan[:,:e['cut']])
 else:x[:,0,j]=v;assert torch.equal(x[:,1:],plan[:,1:])
 for k in range(7):
  if k!=j:assert torch.equal(x[:,:,k],plan[:,:,k])
 return x
def summarize(delta,factual,e,changed):
 a=np.abs(delta);tol=2e-5*np.maximum(1.,np.max(np.abs(factual),(1,2)))
 if e['kind']=='suffix':early=a[:,:e['cut']];late=a[:,e['cut']:]
 else:early=a[:,1:];late=a[:,:1]
 em=early.max((1,2));lm=late.max((1,2))
 return dict(**e,n=len(delta),prefix_max_abs=float(em.max()),prefix_mean_abs=float(early.mean()),prefix_violating_windows=int(np.sum(em>tol)),edited_max_abs=float(lm.max()),edited_mean_abs=float(late.mean()),edited_responding_windows=int(np.sum(lm>tol)),unchanged_plan_windows=int(np.sum(~changed)),tolerance_min=float(tol.min()),tolerance_max=float(tol.max()),prefix_label='times_before_cut' if e['kind']=='suffix' else 'lags_1_to_15',edited_label='times_at_or_after_cut' if e['kind']=='suffix' else 'lag_0')
def fixtures():
 x=np.linspace(.03,.97,2*16*7).reshape(2,16,7);v=x.copy();v[:,8:,0]=.99
 mean=lambda z:np.repeat(z.mean(1,keepdims=True),16,axis=1)
 causal=lambda z:np.cumsum(z,axis=1)
 noncausal=float(abs(mean(v)[:,:8]-mean(x)[:,:8]).max());prefix=float(abs(causal(v)[:,:8]-causal(x)[:,:8]).max())
 w=x.copy();w[:,0,0]=.99;imp=float(abs(causal(w)[:,1:]-causal(x)[:,1:]).max())
 assert noncausal>0 and prefix==0 and imp>0
 return dict(noncausal_positive_control_earlier_difference=noncausal,causal_suffix_prefix_difference=prefix,causal_impulse_lagged_difference=imp,status='PASS')
def main():
 started=time.perf_counter();art=storage('artifact')/'plan_contract_v1';art.mkdir(exist_ok=False)
 sealpath=old.R/'outputs/SELECTION_SEAL_V6.json';seal=read(sealpath);assert len(seal['selected_checkpoints'])==40
 evalroot=old.ART/'evaluation_v1';im=read(evalroot/'input_manifest.json');rm=read(evalroot/'raw_manifest.json')
 assert old.check_source()==125
 for s in seal['selected_checkpoints']:assert sha(s['checkpoint'])==s['sha256']
 banks={}
 for name,entry in im.items():
  path=evalroot/entry['path'];assert sha(path)==entry['sha256'];banks[name]=torch.load(path,map_location='cpu',weights_only=False)
 assert {k:v['n'] for k,v in banks.items()}=={'source_val':163,'new_crop_all':803}
 data=prepare_gpu('greenhouse',context_length=256,horizon=16,stride=80,val_ratio=.2,seed=0,root=old.DATA,device='cpu',cache_dir=old.CACHE)
 assert len(data.train)==815 and len(data.val)==163
 for bank in banks.values():assert bank['schema']==vars(data.schema)
 bounds=np.quantile(data.train.tensors['continuous_future'].numpy().reshape(-1,7),[.1,.9],axis=0)
 names=[x[0] for x in ACTION_CHANNELS];assert len(names)==7
 pin=dict(utc=utc(),source_files_verified=125,source_commit='2e5e48540ff0da48e8192163969a6c58df1d5a3e',protocol_sha256=sha(ROOT/'PROBE_PROTOCOL79_V1.md'),brief_sha256=sha(ROOT/'RESEARCH_BRIEF79_V1.md'),selection_seal_sha256=sha(sealpath),selected_checkpoints=seal['selected_checkpoints'],input_manifest=im,quantiles=dict(source='C53 source TRAIN continuous_future,815 windows x16 steps',levels=[.1,.9],channels=names,values=bounds.tolist()),process_affinity=AFFINITY,torch_version=torch.__version__,numpy_version=np.__version__,new_training=False,future_labels_used_to_choose_edits=False,independent_confirmation=False)
 dump(OUT/'SOURCE_PIN79_V1.json',pin);dump(OUT/'FIXTURES79_V1.json',fixtures())
 records=[];cells=[]
 for selected in seal['selected_checkpoints']:
  tag=selected['tag'];model,armname,seedtext=tag.split('__');seed=int(seedtext[4:]);arm={'obs':'obs','ae_concat':'concat','ae_gate':'gate','ae_gate_soft':'gate_soft'}[armname]
  cell,restore=load_compatible_cell(probe,model,arm,seed,selected['checkpoint'],banks['source_val'],data.schema)
  for bankname,bank in banks.items():
   state('PROGRESS79_V1.json',dict(status='NATIVE_PROBE_RUNNING',tag=tag,bank=bankname,closed_cells=len(cells)))
   oldentry=next(x for x in rm if x['tag']==tag and x['bank']==bankname);oldpath=evalroot/oldentry['path'];assert sha(oldpath)==oldentry['sha256']
   with np.load(oldpath,allow_pickle=False) as a:saved=a['prediction'].copy()
   factual=[];deltas=[[] for _ in EDITS];changes=[[] for _ in EDITS];replaymax=0.;corruptmax=0.
   with torch.no_grad():
    for start in range(0,bank['n'],128):
     batch={k:v[start:start+128] for k,v in bank['tensors'].items()};fw=probe.Forward(cell,batch,arm);plan=batch['continuous_future'];cat=batch['categorical_future'];exog=batch['exog_future']
     base=fw.predict(plan,cat,exog);assert torch.isfinite(base).all();base_np=base.numpy().copy();assert np.allclose(base_np,saved[start:start+128],atol=2e-5,rtol=2e-5),(tag,bankname,float(abs(base_np-saved[start:start+128]).max()))
     factual.append(base_np)
     if start==0:
      repeated=fw.predict(plan,cat,exog);replaymax=max(replaymax,float((repeated-base).abs().max()));assert torch.equal(base,repeated)
      corrupted=dict(batch);corrupted['target_future']=batch['target_future'].flip(0)+7.;other=probe.Forward(cell,corrupted,arm).predict(plan,cat,exog);corruptmax=float((other-base).abs().max());assert corruptmax==0.
     for idx,e in enumerate(EDITS):
      edited=edit(plan,e,bounds);prediction=fw.predict(edited,cat,exog);assert torch.isfinite(prediction).all()
      deltas[idx].append(prediction.numpy().astype(np.float64)-base_np.astype(np.float64));changes[idx].append((edited!=plan).flatten(1).any(1).numpy())
   factual=np.concatenate(factual);delta=np.asarray([np.concatenate(x) for x in deltas]);changed=np.asarray([np.concatenate(x) for x in changes]);path=art/(tag+'__'+bankname+'.npz');np.savez_compressed(path,factual=factual,delta=delta,changed=changed)
   rows=[dict(tag=tag,model=model,arm=armname,seed=seed,bank=bankname,action=names[e['channel']],**summarize(delta[idx],factual,e,changed[idx])) for idx,e in enumerate(EDITS)]
   records.extend(rows);cells.append(dict(tag=tag,bank=bankname,n=bank['n'],array_path=str(path),array_sha256=sha(path),factual_vs_C53_max_abs=float(abs(factual-saved).max()),repeated_factual_max_abs=replaymax,future_label_corruption_max_abs=corruptmax,checkpoint_restore=restore))
   state('PROGRESS79_V1.json',dict(status='NATIVE_PROBE_RUNNING',tag=tag,bank=bankname,closed_cells=len(cells),elapsed_seconds=time.perf_counter()-started))
   print(json.dumps(dict(tag=tag,bank=bankname,c53_factual_max_abs=cells[-1]['factual_vs_C53_max_abs'],suffix_violations=sum(x['prefix_violating_windows'] for x in rows if x['kind']=='suffix'),impulse_lagged_max=max(x['prefix_max_abs'] for x in rows if x['kind']=='impulse'))),flush=True)
  del cell;gc.collect()
 assert len(cells)==80 and len(records)==3920 and old.check_source()==125
 dump(OUT/'PLAN_RESULTS79_V1.json',dict(utc=utc(),status='ALL_40_FIXED_MODELS_BOTH_BANKS_CLOSED_UNVERIFIED',cells=cells,records=records,edits=EDITS,new_training_fits=0,independent_confirmation=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',wall_seconds=time.perf_counter()-started))
 state('PROGRESS79_V1.json',dict(status='ALL_40_FIXED_MODELS_BOTH_BANKS_CLOSED_UNVERIFIED',closed_cells=len(cells),records=len(records),elapsed_seconds=time.perf_counter()-started))
if __name__=='__main__':main()
