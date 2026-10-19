"""Frozen selections, known-clock history inference and scheduled future simulation."""
from common47 import *
torch.set_num_threads(1)
def main():
 assert read(R/'outputs/supervisor_v1.json')['status']=='ALL_SIXTEEN_FITS_COMPLETE_EVAL_UNOPENED';fits=[]
 for case in CASES:
  for method in METHODS:
   for s in SEEDS:
    folder=R/f'outputs/fits/{case}__{method}__seed{s}';fit=read(folder/'fit_summary.json');assert fit['status']=='FIT_COMPLETE_EVAL_UNOPENED' and sha(folder/'selected.pt')==fit['selected_sha256'];fits.append((folder,fit))
 out=R/'outputs/eval_v1';assert not out.exists();out.mkdir(parents=True);dump(out/'OPENING_RECEIPT.json',{'utc':now(),'source_sha256':sha(__file__),'protocol_sha256':sha(R/'PROTOCOL.md'),'selected_weights':{p.name:f['selected_sha256'] for p,f in fits},'EVAL_used_for_fit_selection':False});manifest=read(R/'data/MANIFEST.json');rows=[];raw_manifest=[];couplings=[];bundles=[]
 for case in CASES:
  for name in ['sealed_eval_observations.npz','sealed_eval_auditor_truth.npz','auditor_generating_parameters.npz']:assert sha(R/'data'/case/name)==manifest['files'][case+'/'+name]['sha256']
  observations=np.load(R/'data'/case/'sealed_eval_observations.npz');truth=np.load(R/'data'/case/'sealed_eval_auditor_truth.npz');gp=np.load(R/'data'/case/'auditor_generating_parameters.npz');targets=truth['scheduled_future'];models=[]
  for folder,fit in fits:
   if fit['case']!=case:continue
   cfg=read(folder/'config.json');seed(cfg['seed']);w=World(cfg['method']=='mean_nll');w.load_state_dict(torch.load(folder/'selected.pt',map_location='cpu',weights_only=True)['model']);models.append((folder.name,w,fit,False))
  oracle=World(False)
  with torch.no_grad():oracle.mark.copy_(torch.from_numpy(gp['mark']));oracle.prior.copy_(torch.from_numpy(gp['prior']))
  oracleP=torch.as_tensor(gp['P'],dtype=torch.float32);oracle_rate=torch.as_tensor(gp['rate'],dtype=torch.float32);oracle.components=lambda:(oracleP,oracle_rate);models.append((case+'__known_process_reference',oracle,None,True))
  for tag,model,fit,is_oracle in models:
   model.eval();before=checksum(model)
   with torch.no_grad():
    P,rate=model.components();meanrates=model.mean_rates(P,T).numpy() if model.mean_clock else np.zeros((8,T));bundle=out/(tag+'__model.npz');np.savez_compressed(bundle,P=P.numpy(),rate=rate.numpy(),mark=model.mark.numpy(),prior=model.prior.numpy(),grid=model.grid.numpy(),mean_rates=meanrates,mean_clock=np.array(model.mean_clock),drift=(-.6*gp['grid'] if case=='ou' else 1.2*np.tanh(1.7*gp['grid'])-.75*gp['grid']) if is_oracle else model.drift(model.grid).numpy(),diffusion=np.array((.7 if case=='ou' else .55) if is_oracle else float(model.sigma)));bundles.append({'model':tag,'path':bundle.name,'sha256':sha(bundle),'oracle':is_oracle})
    for policy,rho in enumerate([.25,1.,2.]):
     initial=observations[f'policy{policy}_initial'];counts=observations[f'policy{policy}_counts'];rhos=observations[f'policy{policy}_rho'];assert np.all(rhos==rho);samples=[];nlls=[]
     for b in range(0,len(initial),8):
      ini=torch.as_tensor(initial[b:b+8]);ct=torch.as_tensor(counts[b:b+8],dtype=torch.float32);rr=torch.as_tensor(rhos[b:b+8]);samples.append(model.sample(ini,ct[:,:HIST],rr,64,202647800+policy*100000+b+(0 if is_oracle else fit['seed']*100)).numpy());nlls.extend((-model.joint(ini,ct,rr,P,rate)/(T+1)).numpy().tolist())
     draws=np.concatenate(samples);assert draws.shape==(256,64,FUTURE) and np.isfinite(draws).all();es=energy(torch.from_numpy(draws),targets).numpy();mse=((draws.mean(1)-targets)**2).mean(-1);low,high=np.quantile(draws,[.05,.95],axis=1);coverage=((targets>=low)&(targets<=high)).mean(-1);file=out/(tag+f'__policy{policy}.npz');np.savez_compressed(file,draws=draws,truth=targets,initial=initial,counts=counts,rho=rhos,energy=es,mse=mse,coverage=coverage,nll_per65=np.array(nlls));raw_manifest.append({'model':tag,'case':case,'rho':rho,'path':file.name,'sha256':sha(file),'origins':256,'oracle':is_oracle})
     for b in range(256):rows.append({'model':tag,'case':case,'rho':rho,'origin':b,'oracle':is_oracle,'energy':float(es[b]),'mse':float(mse[b]),'coverage90':float(coverage[b]),'nll_per65':nlls[b]})
     probability,cov=model.coupling(rho);couplings.append({'model':tag,'case':case,'rho':rho,'oracle':is_oracle,'initial_bin_probability':probability.numpy().tolist(),'conditional_count_state_covariance':cov.numpy().tolist(),'probability_weighted_covariance':float((probability*cov).sum())});print(json.dumps({'model':tag,'rho':rho,'energy':float(es.mean()),'mse':float(mse.mean()),'coverage90':float(coverage.mean()),'nll_per65':float(np.mean(nlls))}),flush=True)
   assert before==checksum(model)
   if fit is not None:assert sha(R/f'outputs/fits/{tag}/selected.pt')==fit['selected_sha256']
 dump(out/'raw_manifest.json',raw_manifest);dump(out/'model_bundles.json',bundles);dump(out/'all_rows.json',rows);dump(out/'coupling.json',couplings);summary=[]
 for record in raw_manifest:
  rs=[r for r in rows if r['model']==record['model'] and r['rho']==record['rho']];summary.append({**{k:record[k] for k in ['model','case','rho','oracle']},'n_model_origins':len(rs),**{m:float(np.mean([r[m] for r in rs])) for m in ['energy','mse','coverage90','nll_per65']}})
 dump(out/'summary.json',{'status':'SCORED_PENDING_INDEPENDENT_VERIFICATION','utc':now(),'groups':summary,'unique_physical_EVAL_trajectories':512,'paired_observation_policies':3,'model_origin_policy_entries':len(rows),'raw_groups':len(raw_manifest),'learned_models':16,'known_process_references':2,'source_sha256':sha(__file__),'raw_manifest_sha256':sha(out/'raw_manifest.json'),'model_bundles_sha256':sha(out/'model_bundles.json'),'rows_sha256':sha(out/'all_rows.json')})
if __name__=='__main__':main()
