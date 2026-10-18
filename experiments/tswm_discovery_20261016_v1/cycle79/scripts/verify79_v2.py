"""Independent recomputation of every stored edit metric and fixed native replays."""
import sys,pathlib,gc,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common79_v1 import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
sys.path.insert(0,str(PROJECT/'cycle71/scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
sys.path.insert(0,str(PROJECT/'cycle53/scripts'))
import native53 as old
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
from replay53 import load_compatible_cell
from common.windowing import Schema
sys.path.insert(0,str(old.UP/'RQ6_intervention/scripts'))
import probe_intervention as probe
probe.DEVICE='cuda';probe.AE_DIR=old.CODEC
def main():
 started=time.perf_counter();resultpath=OUT/'PLAN_RESULTS79_V2.json';r=read(resultpath);pin=read(OUT/'SOURCE_PIN79_V2.json');assert old.check_source()==125 and len(r['cells'])==80 and len(r['records'])==3920
 banks={}
 for name,entry in pin['input_manifest'].items():
  path=old.ART/'evaluation_v1'/entry['path'];assert sha(path)==entry['sha256'];banks[name]=torch.load(path,map_location='cpu',weights_only=False)
 bounds=np.asarray(pin['quantiles']['values']);metricmax=0.;native_replaymax=0.;values=0;checks=[]
 selected={x['tag']:x for x in pin['selected_checkpoints']}
 for tag in selected:
  s=selected[tag];assert sha(s['checkpoint'])==s['sha256'];model,armname,seedtext=tag.split('__');arm={'obs':'obs','ae_concat':'concat','ae_gate':'gate','ae_gate_soft':'gate_soft'}[armname]
  cell,restore=load_compatible_cell(probe,model,arm,int(seedtext[4:]),s['checkpoint'],banks['source_val'],Schema(**banks['source_val']['schema']))
  for bankname,bank in banks.items():
   c=next(x for x in r['cells'] if x['tag']==tag and x['bank']==bankname);assert sha(c['array_path'])==c['array_sha256'];a=np.load(c['array_path'],allow_pickle=False);base=a['factual'];delta=a['delta'];changed=a['changed'];assert base.shape==(bank['n'],16,4) and delta.shape==(49,bank['n'],16,4) and np.isfinite(delta).all() and np.isfinite(base).all();values+=delta.size+base.size
   tol=2e-5*np.maximum(1.,abs(base).max(axis=(1,2)))
   for idx,e in enumerate(r['edits']):
    row=next(x for x in r['records'] if x['tag']==tag and x['bank']==bankname and all(x[k]==e[k] for k in ['kind','channel','cut','quantile']));d=np.abs(delta[idx]);part1=d[:,:e['cut']] if e['kind']=='suffix' else d[:,1:];part2=d[:,e['cut']:] if e['kind']=='suffix' else d[:,:1];pmax=part1.max(axis=(1,2));lmax=part2.max(axis=(1,2))
    scalars=dict(prefix_max_abs=pmax.max(),prefix_mean_abs=part1.mean(),edited_max_abs=lmax.max(),edited_mean_abs=part2.mean(),tolerance_min=tol.min(),tolerance_max=tol.max())
    for k,v in scalars.items():metricmax=max(metricmax,abs(float(v)-row[k]));assert abs(float(v)-row[k])<1e-12
    assert row['prefix_violating_windows']==int(np.count_nonzero(pmax>tol)) and row['edited_responding_windows']==int(np.count_nonzero(lmax>tol))
    plan=bank['tensors']['continuous_future'].numpy();j=e['channel'];v=bounds[0 if e['quantile']==.1 else 1,j];part=plan[:,e['cut']:,j] if e['kind']=='suffix' else plan[:,:1,j];actual_changed=np.any(part!=np.float32(v),axis=1);assert np.array_equal(actual_changed,changed[idx]);assert row['unchanged_plan_windows']==int(np.count_nonzero(~actual_changed))
   batch={k:v[:128].to('cuda') for k,v in bank['tensors'].items()}
   with torch.no_grad():
    fw=probe.Forward(cell,batch,arm);f=fw.predict(batch['continuous_future'],batch['categorical_future'],batch['exog_future']).cpu().numpy();assert np.allclose(f,base[:len(f)],atol=2e-5,rtol=2e-5);err=float(abs(f-base[:len(f)]).max());native_replaymax=max(native_replaymax,err)
    for idx in [0,21,40,43]:
     e=r['edits'][idx];x=batch['continuous_future'].clone();j=e['channel'];v=float(bounds[0 if e['quantile']==.1 else 1,j])
     if e['kind']=='suffix':x[:,e['cut']:,j]=v;assert torch.equal(x[:,:e['cut']],batch['continuous_future'][:,:e['cut']])
     else:x[:,0,j]=v;assert torch.equal(x[:,1:],batch['continuous_future'][:,1:])
     p=fw.predict(x,batch['categorical_future'],batch['exog_future']).cpu().numpy();actual=p.astype(np.float64)-f.astype(np.float64);err=float(abs(actual-delta[idx,:len(f)]).max());native_replaymax=max(native_replaymax,err);assert np.allclose(actual,delta[idx,:len(f)],atol=2e-5,rtol=2e-5);checks.append(dict(tag=tag,bank=bankname,edit_index=idx,max_abs_delta_replay_error=err))
   a.close()
  del cell;gc.collect()
 assert len(checks)==320
 dump(OUT/'VERIFICATION79_V2.json',dict(utc=utc(),status='PASS_ALL_METRICS_AND_320_FIXED_NATIVE_EDIT_REPLAYS',result_sha256=sha(resultpath),source_files=old.check_source(),cells=80,records=3920,array_values_checked=values,metric_max_abs_error=metricmax,native_max_abs_replay_error=native_replaymax,checks=checks,wall_seconds=time.perf_counter()-started,accepted_final_directions=0))
 print(json.dumps(dict(status='PASS',cells=80,records=3920,array_values_checked=values,native_max_abs_replay_error=native_replaymax)),flush=True)
if __name__=='__main__':main()
