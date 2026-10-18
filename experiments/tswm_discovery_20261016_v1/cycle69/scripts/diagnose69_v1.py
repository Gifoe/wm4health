import sys,pathlib,time,math
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native69_v1 import *
assert read(OUT/'INDEPENDENT_REPLAY69_V1.json')['status']=='PASS';manifest=read(OUT/'DATA_MANIFEST69_V1.json');case=next(c for c in manifest['cases'] if c['benchmark']=='WH');data={k:v for k,v in np.load(case['path'],allow_pickle=False).items()};ledger=read(OUT/'FORECAST_LEDGER69_V1.json');old=next(r for r in ledger['records'] if r['benchmark']=='WH');assert sha(old['path'])==old['sha256'];oldz=np.load(old['path'],allow_pickle=False);n=50;target=data['test_y'][n:];rows=[dict(mode='primary_first75000',RMSE=float(np.sqrt(np.mean((oldz['meta_precision'][:75000]-target[:75000])**2))),points=75000,role='diagnostic subset, not replacement primary score')];model=Meta();arrays={};times={}
for mode in ['fused_feedback','privileged_measured_boundary']:
 model.set_context(data,case);tu,iu,iy=model.init_state(data,n,'WH');actual_y=(torch.tensor(data['test_y'],dtype=torch.float32,device='cuda')-model.ym)/model.ys if mode.startswith('privileged') else None;pred=[];normalized=[];start=time.perf_counter()
 for offset in range(0,len(target),100):
  t=n+offset;fu=tu[t:t+100];actual=len(fu)
  if actual<100:fu=torch.cat([fu,fu[-1:].expand(100-actual)])
  if offset>0:
   iu=tu[t-30:t][None,:,None].expand(model.contexts,-1,-1)
   if actual_y is not None:iy=actual_y[t-30:t][None,:,None].expand(model.contexts,-1,-1)
  means,stds=model.query(iu,iy,fu);point,_,_=ensemble(means[:,:,0].cpu().numpy(),stds[:,:,0].cpu().numpy(),'precision');normalized.extend(point[:actual]);pred.extend(point[:actual]*float(model.ys.cpu())+float(model.ym.cpu()));iy=torch.tensor(point[-30:][None,:,None],dtype=torch.float32,device='cuda').expand(model.contexts,-1,-1)
  if (offset//100+1)%100==0:state('diagnostic_state69_v1.json',dict(status='LIVE_DIAGNOSTIC',mode=mode,completed=offset+actual,total=len(target)))
 pred=np.asarray(pred);arrays[mode]=pred;arrays[mode+'_normalized_point']=np.asarray(normalized);d=pred-target;times[mode]=time.perf_counter()-start;rows.append(dict(mode=mode,RMSE=float(np.sqrt(np.mean(d*d))),MAE=float(abs(d).mean()),points=len(d),first100_RMSE=float(np.sqrt(np.mean(d[:100]**2))),early_half_RMSE=float(np.sqrt(np.mean(d[:len(d)//2]**2))),late_half_RMSE=float(np.sqrt(np.mean(d[len(d)//2:]**2))),information='True test outputs at each100 boundary' if actual_y is not None else 'Only original initial50 and model predictions',role='privileged explanatory control' if actual_y is not None else 'post-discovery ordinary implementation alternative, not selected replacement'))
 print(json.dumps(rows[-1]),flush=True)
path=storage('artifact')/'WH_DIAGNOSTIC69_V1.npz';assert not path.exists();np.savez_compressed(path,**arrays);checks=[]
for mode in ['fused_feedback','privileged_measured_boundary']:
 d=[float(a)-float(b) for a,b in zip(arrays[mode],target)];r=next(x for x in rows if x['mode']==mode);assert abs(math.sqrt(math.fsum(q*q for q in d)/len(d))-r['RMSE'])<1e-10;model.set_context(data,case);tu,iu,iy=model.init_state(data,n,'WH');offset=100*(len(target)//200);t=n+offset;iu=tu[t-30:t][None,:,None].expand(model.contexts,-1,-1)
 if mode=='fused_feedback':history=arrays[mode+'_normalized_point'][offset-30:offset];iy=torch.tensor(history[None,:,None],dtype=torch.float32,device='cuda').expand(model.contexts,-1,-1)
 else:history=(torch.tensor(data['test_y'][t-30:t],dtype=torch.float32,device='cuda')-model.ym)/model.ys;iy=history[None,:,None].expand(model.contexts,-1,-1)
 means,stds=model.query(iu,iy,tu[t:t+100]);point,_,_=ensemble(means[:,:,0].cpu().numpy(),stds[:,:,0].cpu().numpy(),'precision');expected=point*float(model.ys.cpu())+float(model.ym.cpu());err=float(abs(expected-arrays[mode][offset:offset+100]).max());assert err<1e-6;checks.append(dict(mode=mode,midrun_fresh_query_max_abs=err,metric_fsum_check=True))
dump(OUT/'WH_DIAGNOSTIC69_V1.json',dict(utc=utc(),status='FIXED_POST_DISCOVERY_DIAGNOSTICS_AND_REPLAY_COMPLETE',rows=rows,seconds=times,artifact=str(path),artifact_sha256=sha(path),checks=checks,protocol_sha256=sha(ROOT/'WH_DIAGNOSTIC69_V1.md'),primary_results_unchanged_sha256=sha(OUT/'RESULTS69_V1.json'),new_method_confirmation=False,accepted_final_directions=0));state('diagnostic_state69_v1.json',dict(status='DIAGNOSTIC_TERMINAL_SUCCESS'));print('DIAGNOSTIC_TERMINAL_SUCCESS',flush=True)
