"""Pinned released PFN and native CPU controls on a frozen protocol subset."""
import runtime_profile
import contextlib,csv,hashlib,json,os,pathlib,sys,time,traceback
FULL='--full' in sys.argv
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/('outputs/dotime_full' if FULL else 'outputs/dotime');OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT.parent/'upstream/dotime/src')]
os.environ['HF_ENDPOINT']='https://hf-mirror.com';os.environ['HF_HUB_DISABLE_XET']='1'
import numpy as np
import torch
from inverse_score_audit import Tee
def main():
 from dotime import baselines
 from dotime.benchmarks import load_benchmark
 from dotime.evaluation import query_obs_levels,check_shared_noise
 from dotime.reference.pfn import PFNRef,episode_to_batch_interp,run
 from dotime.reference.reference_table import target_qa
 from dotime.reference._scoring import direction_scores
 torch.set_num_threads(1);torch.manual_seed(2021)
 suite=load_benchmark('dot-Identifiability-v1',version='1.2.0',cache_dir=ROOT/'data/dotime_cache')
 episodes=[ep for _,group in suite.by_structure() for ep in (group if FULL else group[:60])]
 assert len(episodes)==(12150 if FULL else 540)
 config={'suite':suite.meta.name,'version':suite.meta.version,'selection':('All12150 released episodes, sorted native structure groups' if FULL else 'First60 episodes per structure in native suite order, sorted structure groups; fixed before metrics'),'episodes':len(episodes),'checkpoint':'s9ho_all_causal','checkpoint_revision':'bec6618c8b25f3a5d8cf608d4c597978d5a694ac','source_revision':'8f227406a5b17c2040cfdcc680b493729ef47d00','batch_size':8,'training':'none; published frozen checkpoint','comparators':['Mean','VAR-OLS','NaiveOLS'],'no_new_model_or_protocol_claim':True}
 (OUT/'eval_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
 noise=check_shared_noise(episodes);print('shared-noise',noise,flush=True)
 qa=target_qa(episodes,dir_target='effect');(OUT/'target_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
 model=PFNRef(ROOT/'data/do_over_time_pfn_best.pt',device='cuda')
 batches=[episode_to_batch_interp(ep,model.n_max,'cuda',False,model.convention) for ep in episodes]
 preds=[];quantiles=[];started=time.time();max_difference=0
 with torch.inference_mode():
  for start in range(0,len(episodes),8):
   selected=batches[start:start+8]
   # Every batch field has one episode in its leading axis, including normalization stats.
   batch={k:torch.cat([b[k] for b in selected],dim=0) for k in selected[0]}
   output=model.model(batch);head=model.model.quantile_head
   query=batch['query_target'];ii=torch.arange(len(selected),device='cuda');mean=batch['_norm_means'][ii,query];sd=batch['_norm_stds'][ii,query]
   p=head.predict_mean(output).reshape(-1)*sd+mean
   qq=output*sd[:,None]+mean[:,None]
   if start==0:
    scalar=torch.cat([model.predict(ep).reshape(-1) for ep in episodes[:8]]).to('cuda')
    max_difference=float((p-scalar).abs().max());assert torch.allclose(p,scalar,rtol=2e-5,atol=2e-5),(p,scalar)
   preds.extend(p.cpu().tolist());quantiles.extend(qq.cpu().tolist())
   if start%800==0:print('PFN',start+len(selected),len(episodes),'seconds',round(time.time()-started,2),flush=True)
 del batches
 pp=np.asarray(preds);qq=np.asarray(quantiles);tt=np.asarray([float(ep.y_true[0]) for ep in episodes]);obs=np.asarray([float(query_obs_levels(ep)[0]) for ep in episodes])
 assert np.isfinite(pp).all() and np.isfinite(qq).all()
 all_preds={'PFNRef':pp}
 for name in config['comparators']:
  control=baselines.get(name);values=np.asarray([float(control.predict(ep)[0]) for ep in episodes]);assert np.isfinite(values).all();all_preds[name]=values
 structures=sorted({ep.structure for ep in episodes});rows=[]
 for name,values in all_preds.items():
  for structure in ['ALL']+structures:
   index=np.asarray([i for i,e in enumerate(episodes) if structure=='ALL' or e.structure==structure]);p,t,o=values[index],tt[index],obs[index]
   row={'model':name,'structure':structure,'n':len(index),'rmse':float(np.sqrt(np.mean((p-t)**2))),**direction_scores(p,t,o,'effect')};rows.append(row)
   if structure=='ALL':print('RESULT',json.dumps(row),flush=True)
 by_id={ep.scm_id:i for i,ep in enumerate(episodes)};assert len(by_id)==len(episodes)
 native=run(type('CachedPrediction',(),{'predict':lambda self,ep:torch.tensor([all_preds['PFNRef'][by_id[ep.scm_id]]])})(),episodes,dir_target='effect')
 (OUT/'native_pfn_metrics.json').write_text(json.dumps(native,indent=2),encoding='utf-8')
 tau=head.tau_levels.cpu().numpy();lo=int(np.abs(tau-.05).argmin());hi=int(np.abs(tau-.95).argmin())
 diag={'native_scalar_vs_batch_max_difference':max_difference,'quantile_levels':tau.tolist(),'interval_requested':[.05,.95],'actual_interval_levels':[float(tau[lo]),float(tau[hi])],'coverage':float(np.mean((tt>=qq[:,lo])&(tt<=qq[:,hi]))),'width_mean':float(np.mean(qq[:,hi]-qq[:,lo])),'crossed_adjacent_fraction':float(np.mean(np.diff(qq,axis=1)<0)),'elapsed_seconds':time.time()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'scope':'One published frozen checkpoint; uncertainty levels are interventional outcome levels, not individual causal-effect intervals.'}
 (OUT/'diagnostics.json').write_text(json.dumps(diag,indent=2),encoding='utf-8')
 (OUT/'metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 with (OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
  names=['scm_id','structure','y_true','y_obs']+list(all_preds)+[f'q_{t:.3f}' for t in tau];writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
  for i,e in enumerate(episodes):writer.writerow({'scm_id':e.scm_id,'structure':e.structure,'y_true':tt[i],'y_obs':obs[i],**{k:v[i] for k,v in all_preds.items()},**{f'q_{t:.3f}':qq[i,j] for j,t in enumerate(tau)}})
 print('COMPLETE',json.dumps(diag),flush=True)
if __name__=='__main__':
 with (OUT/'eval_stdout.log').open('a',encoding='utf-8') as stdout,(OUT/'eval_stderr.log').open('a',encoding='utf-8') as stderr,contextlib.redirect_stdout(Tee(sys.stdout,stdout)),contextlib.redirect_stderr(Tee(sys.stderr,stderr)):
  try:main()
  except BaseException:traceback.print_exc();raise
