"""Locked real-process mean-forecast and revealed-prefix continuation scores."""
from common48_v3 import *
from control48 import features
def metrics(prediction,truth):
 p=np.asarray(prediction,dtype='float64')[:,:,0];y=np.asarray(truth,dtype='float64')[:,:,0];error=p-y;return {'mse':(error**2).mean(1),'mae':abs(error).mean(1),'first12_mse':(error[:,:12]**2).mean(1),'last12_mse':(error[:,-12:]**2).mean(1),'tail24_mse':(error[:,-24:]**2).mean(1),'aggregate24_mse':(error.reshape(len(p),-1,24).mean(-1)**2).mean(1),'aggregate_all_mse':error.mean(1)**2}
def main():
 assert read(R/'outputs/supervisor_v1.json')['status']=='ALL_SIXTEEN_FITS_COMPLETE_EVAL_UNOPENED';selection=read(R/'outputs/control_v1/selection.json');assert selection['status']=='ALL_RIDGE_SELECTIONS_COMPLETE_EVAL_UNOPENED';fits=[]
 for case in CASES:
  for method in METHODS:
   for s in SEEDS:
    folder=R/f'outputs/fits/{case}__{method}__seed{s}';fit=read(folder/'fit_summary.json');assert sha(folder/'selected.pt')==fit['selected_sha256'];fits.append((folder,fit))
 for ent in selection['files']:assert sha(R/'outputs/control_v1'/ent['path'])==ent['sha256']
 out=R/'outputs/eval_v1';assert not out.exists();out.mkdir();dump(out/'OPENING_RECEIPT.json',{'utc':now(),'source_sha256':sha(__file__),'protocol_sha256':sha(R/'PROTOCOL.md'),'selected_weights':{p.name:f['selected_sha256'] for p,f in fits},'ridge_selection_sha256':sha(R/'outputs/control_v1/selection.json'),'EVAL_used_for_selection':False});rows=[];manifest=[];summary=[];unique=0
 for case in CASES:
  data=load(case,'eval');unique+=len(data['history']);entries=[]
  for folder,fit in fits:
   if fit['case']!=case:continue
   model=MemoryModel(fit['method']);model.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)['model']);model.eval();entries.append((folder.name,fit['method'],fit['seed'],model,fit))
  entries.extend((case+'__'+method,method,None,None,None) for method in ['ridge','persistence','seasonal24']);ridge=next(e for e in selection['files'] if e['case']==case);weights=np.load(R/'outputs/control_v1'/ridge['path'])['weights']
  for tag,method,s,model,fit in entries:
   before=checksum(model) if model is not None else None
   for reveal in [0,24]:
    history=data['history'] if not reveal else torch.cat((data['history'],data['future'][:,:reveal]),1)[:,-H:];phase=data['phase'] if not reveal else (data['phase']+reveal/24)%1;truth=data['future'][:,reveal:];horizon=FUTURE-reveal;start=time.perf_counter()
    if model is not None:
     with torch.no_grad():prediction=torch.cat([model(history[i:i+16],phase[i:i+16],horizon=horizon) for i in range(0,len(history),16)]).numpy()
    elif method=='ridge':prediction=(features({'history':history,'phase':phase})@weights)[:,:horizon,None]
    elif method=='persistence':prediction=history[:,-1:,:].expand(-1,horizon,-1).numpy()
    else:prediction=np.tile(history[:,-24:,:].numpy(),(1,horizon//24,1))
    elapsed=time.perf_counter()-start;assert prediction.shape==tuple(truth.shape) and np.isfinite(prediction).all();values=metrics(prediction,truth.numpy());path=out/(tag+f'__reveal{reveal}.npz');np.savez_compressed(path,prediction=prediction,truth=truth.numpy(),history=history.numpy(),phase=phase.numpy(),origins=data['origins'],**values);ent={'model':tag,'case':case,'method':method,'seed':s,'reveal':reveal,'horizon':horizon,'path':path.name,'sha256':sha(path),'origins':len(history)};manifest.append(ent);group={**{k:ent[k] for k in ['model','case','method','seed','reveal','horizon','origins']},**{k:float(v.mean()) for k,v in values.items()},'inference_seconds':elapsed};summary.append(group)
    for j,origin in enumerate(data['origins']):rows.append({**{k:ent[k] for k in ['model','case','method','seed','reveal','horizon']},'origin':int(origin),**{k:float(v[j]) for k,v in values.items()}})
    print(json.dumps(group),flush=True)
   if model is not None:assert checksum(model)==before and sha(R/'outputs/fits'/tag/'selected.pt')==fit['selected_sha256']
 dump(out/'raw_manifest.json',manifest);dump(out/'all_rows.json',rows);dump(out/'summary.json',{'status':'SCORED_PENDING_INDEPENDENT_VERIFICATION','utc':now(),'groups':summary,'unique_forecast_origins':unique,'independent_real_processes':2,'raw_groups':len(manifest),'model_origin_reveal_entries':len(rows),'conditional_mean_metrics_only':True,'source_sha256':sha(__file__),'raw_manifest_sha256':sha(out/'raw_manifest.json'),'all_rows_sha256':sha(out/'all_rows.json')})
if __name__=='__main__':main()
