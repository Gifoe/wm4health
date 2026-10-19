"""Eight ordinary fits, validation-only screen. No new-method or TEST claims."""
from common40_v2 import *
import datetime,csv
O=R/'outputs/screen_v1';assert not O.exists();O.mkdir()
protocol=hashlib.sha256((R/'ORDINARY_SCREEN_PROTOCOL.md').read_bytes()).hexdigest()
qualification=json.loads((R/'outputs/qualification.json').read_text());assert not qualification['validation_scored']
config={'protocol_sha256':protocol,'updates':2000,'batch':32,'seeds':[2026401,2026402],'datasets':['ett','jena'],'families':['ar8mix','flownp'],'evaluation_draws':32,'selection_steps':[0,500,1000,1500,2000],'test_scored':False}
(O/'config.json').write_text(json.dumps(config,indent=2))
def savejson(path,x):path.write_text(json.dumps(x,indent=2),encoding='utf-8')
@torch.no_grad()
def evaluate(model,family,series,ends,steps=32,seed_id=991):
 model.eval();rng=torch.random.get_rng_state();crng=torch.cuda.get_rng_state();paths=[];times=[]
 chunk=4 if family=='flownp' else 32
 for r in REVEALS:
  pieces=[];start=time.perf_counter()
  for begin in range(0,len(ends),chunk):
   es=ends[begin:begin+chunk];seq,c=windows(series,es);seed(seed_id+begin*31+r)
   if family=='flownp':x=fnp_sample(model,seq,c,r,32,steps)
   else:
    cache=model.context(seq,c,r);x=model.sample_cached(seq,c,r,cache,32)
   pieces.append(x[:,:,64-r:,0].cpu().numpy())
  times.append(time.perf_counter()-start);paths.append(np.concatenate(pieces,axis=1))
 torch.random.set_rng_state(rng);torch.cuda.set_rng_state(crng)
 truth=series[torch.as_tensor(ends,device=DEVICE)[:,None]+torch.arange(64,96,device=DEVICE)[None]].cpu().numpy()
 result=[scores(x,truth) for x in paths]
 return paths,truth,result,times
def select_origins(name):
 es=origins(name,'val')[::32];assert len(es)>8
 small=es[np.unique(np.linspace(0,len(es)-1,min(32,len(es))).astype(int))]
 return es,small
@torch.no_grad()
def cost(model,family,series,origin,steps):
 model.eval();seq,c=windows(series,[origin]);seed(12345)
 if family=='flownp':fnp_sample(model,seq,c,0,32,steps)
 else:
  ca=model.context(seq,c,0);model.sample_cached(seq,c,0,ca,32)
 rows=[];cache=None
 for r in REVEALS:
  if family=='flownp':
   _,elapsed=timed(lambda:fnp_sample(model,seq,c,r,32,steps));rows.append({'r':r,'full_query_seconds':elapsed,'conditioning_included':True})
  else:
   cache,setup=timed(lambda:model.context(seq,c,r,cache))
   _,draw=timed(lambda:model.sample_cached(seq,c,r,cache,32))
   rows.append({'r':r,'cache_build_or_real_evidence_update_seconds':setup,'32_draws_seconds':draw,'incremental_query_seconds':setup+draw,'cache_setup_is_not_free':True})
 return rows
results=[]
for name in config['datasets']:
 series=data(name);allval,selection=select_origins(name);training=origins(name,'train');assert len(training)>1000
 for sid in config['seeds']:
  # Same train-origin batches across families, pre-generated from training only.
  schedule=np.random.default_rng(sid).choice(training,size=(2000,32),replace=True)
  np.save(O/f'{name}_{sid}_schedule.npy',schedule)
  for family in config['families']:
   key=f'{name}_{sid}_{family}';D=O/key;D.mkdir();seed(sid)
   model=ARMixture().cuda() if family=='ar8mix' else make_fnp();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
   print(json.dumps({'event':'fit_start','key':key,'pid':__import__('os').getpid()}),flush=True)
   curves=[];best=float('inf');beststep=None;trainloss=[];torch.cuda.reset_peak_memory_stats();fitstart=time.perf_counter();trainsecs=0.
   for step in range(2001):
    if step in config['selection_steps']:
     _,_,sc,ts=evaluate(model,family,series,selection,seed_id=sid+80000)
     energy=float(np.mean([z['energy'].mean() for z in sc]));row={'step':step,'selection_energy':energy,'score_by_reveal':[{k:float(v.mean()) for k,v in z.items()} for z in sc],'validation_seconds':sum(ts)};curves.append(row)
     if energy<best:
      best=energy;beststep=step;torch.save({'model':model.state_dict(),'family':family,'data':name,'seed':sid,'step':step,'score':best,'protocol_sha256':protocol},D/'selected.pt')
     savejson(D/'curve.json',curves)
     print(json.dumps({'event':'validation','key':key,'step':step,'energy':energy,'best_step':beststep}),flush=True)
    if step==2000:break
    model.train();seq,c=windows(series,schedule[step]);opt.zero_grad(set_to_none=True);torch.cuda.synchronize();a=time.perf_counter()
    loss=model.loss(seq,c) if family=='ar8mix' else model(fnp_batch(seq,c,REVEALS[step%3])).loss
    assert torch.isfinite(loss),(key,step,float(loss));loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize();trainsecs+=time.perf_counter()-a
    if step%50==0:trainloss.append({'step':step+1,'loss':float(loss)})
   torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':2000,'protocol_sha256':protocol},D/'last.pt')
   savejson(D/'training_loss.json',trainloss)
   selected=torch.load(D/'selected.pt',map_location=DEVICE,weights_only=False);model.load_state_dict(selected['model'],strict=True)
   evals=[]
   for steps in ([0] if family=='ar8mix' else [1,2,8,32]):
    paths,truth,score,times=evaluate(model,family,series,allval,steps=steps,seed_id=sid+160000)
    raw={'origins':allval,'truth':truth}
    for r,x in zip(REVEALS,paths):raw[f'r{r}']=x
    rawpath=D/f'paths_steps{steps}.npz';np.savez_compressed(rawpath,**raw)
    evals.append({'steps':steps,'means_by_reveal':[{k:float(v.mean()) for k,v in z.items()} for z in score],'raw':rawpath.name,'raw_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),'batched_evaluation_seconds_by_reveal':times,'single_origin_cost':cost(model,family,series,int(allval[0]),steps)})
   rec={'key':key,'data':name,'seed':sid,'family':family,'parameters':sum(p.numel() for p in model.parameters()),'train_update_seconds':trainsecs,'fit_and_eval_wall_seconds':time.perf_counter()-fitstart,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'selected_step':beststep,'selected_at_cap':beststep==2000,'last_validation_improves_over_previous':curves[-1]['selection_energy']<curves[-2]['selection_energy'],'curves':curves,'evals':evals,'selected_checkpoint_sha256':hashlib.sha256((D/'selected.pt').read_bytes()).hexdigest(),'test_scored':False}
   savejson(D/'result.json',rec);results.append(rec);savejson(O/'results.json',results);del model,opt;torch.cuda.empty_cache()
   print(json.dumps({'event':'fit_complete','key':key,'selected_step':beststep,'total_fits':len(results)}),flush=True)
savejson(O/'COMPLETE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fits':len(results),'test_scored':False,'protocol_sha256':protocol});print('C40_SCREEN_COMPLETE',flush=True)
