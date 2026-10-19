"""Fresh selected weights only; EVAL and truth files are never loaded here."""
from common44 import *
import datetime,traceback,subprocess
Q=read(R/'outputs/qualification_v1/qualification.json');assert Q['status']=='PASS' and Q['common_sha256']==sha(R/'scripts/common44.py') and Q['protocol_sha256']==sha(R/'PROTOCOL.md')
O=R/'outputs/fits_v1';assert not O.exists();O.mkdir(parents=True);state={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fits_completed':0,'fits_total':16,'eval_opened':False,'protocol_sha256':sha(R/'PROTOCOL.md')};dump(O/'status.json',state)
try:
 for case in ['gaussian','nonlinear']:
  data=R/'data'/case;train=np.load(data/'train.npz')['observations'];val=np.load(data/'val.npz')['observations'];pca=dict(np.load(data/'pca.npz'))
  for fitseed in SEEDS:
   rng=np.random.default_rng(fitseed);schedule=rng.integers(len(train),size=(10000,24));np.save(O/f'schedule_{case}_{fitseed}.npy',schedule)
   for family in ['lgssm','ar_mixture','spf_native','spf_condition_trainable']:
    label=f'{case}_{family}_{fitseed}';folder=O/label;folder.mkdir();state.update(current_fit=label,current_step=0);dump(O/'status.json',state)
    gpu_before=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    seed(fitseed);init=time.perf_counter()
    if family.startswith('spf'):container=make_spf(family=='spf_condition_trainable');model=container.model
    elif family=='ar_mixture':model=AR(pca).cuda();container=model
    else:model=LGSSM(pca,train).cuda();container=model
    torch.cuda.synchronize();initialization_seconds=time.perf_counter()-init
    parameters=sum(p.numel() for p in model.parameters());minimum,maximum,interval=(1000,3000,500) if family=='lgssm' else (4000,10000,1000);base_lr=.005 if family=='lgssm' else .0003
    optimizer=torch.optim.Adam(model.parameters(),lr=base_lr) if family=='lgssm' else torch.optim.AdamW(model.parameters(),lr=base_lr,weight_decay=1e-4)
    if family.startswith('spf'):
     import emulator.data.loading.collate as native_collate
     native_collate._WORKER_GATES.clear()
    seed(fitseed+10);curves=[];best_score=float('inf');best=None;stale=0;validation_seconds=0.;active=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for step in range(maximum+1):
     if step%interval==0:
      model.eval();torch.cuda.synchronize();before=time.perf_counter();score=validation(family,container,val);torch.cuda.synchronize();validation_seconds+=time.perf_counter()-before;assert math.isfinite(score)
      improvement=(best_score-score)/max(abs(best_score),1e-8) if math.isfinite(best_score) else float('inf');stale=stale+1 if improvement<.002 else 0
      if score<best_score:best_score=score;best=numpy_state(model);selected=step
      row={'step':step,'val_fine_energy':score,'best_val':best_score,'best_relative_improvement':improvement if math.isfinite(improvement) else None,'stale_checkpoints':stale,'elapsed_seconds_including_validation':time.perf_counter()-start};curves.append(row);dump(folder/'curves.json',curves);print(json.dumps({'fit':label,**row}),flush=True);state.update(current_step=step);dump(O/'status.json',state)
      if step>=minimum and stale>=2:reason='validation_plateau';break
      if step==maximum:reason='cap';break
     model.train();optimizer.param_groups[0]['lr']=base_lr*(.1+.9*.5*(1+math.cos(math.pi*step/maximum)));optimizer.zero_grad(set_to_none=True);batch=torch.tensor(train[schedule[step]])
     loss=native_loss(container,batch) if family.startswith('spf') else model.loss(batch.cuda());assert torch.isfinite(loss),(label,step,float(loss))
     loss.backward();active=sum(p.numel() for p in model.parameters() if p.grad is not None);torch.nn.utils.clip_grad_norm_(model.parameters(),5. if family=='lgssm' else 1.);optimizer.step()
     if family=='lgssm':model.stabilize()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save({'selected_state':best,'last_state':numpy_state(model),'case':case,'family':family,'seed':fitseed,'selected_step':selected,'stopped_step':step,'pca_sha256':sha(data/'pca.npz'),'protocol_sha256':sha(R/'PROTOCOL.md'),'common_sha256':sha(R/'scripts/common44.py')},folder/'model.pt')
    result={'case':case,'family':family,'seed':fitseed,'selected_step':selected,'stopped_step':step,'stop_reason':reason,'selected_val_energy':best_score,'last_val_energy':curves[-1]['val_fine_energy'],'selected_cap':selected==maximum,'last_interval_improved':len(curves)>1 and curves[-1]['val_fine_energy']<curves[-2]['val_fine_energy'],'nominal_parameters':parameters,'parameters_with_last_gradient':active,'initialization_seconds':initialization_seconds,'training_and_validation_seconds':elapsed,'validation_seconds':validation_seconds,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'gpu_context_before':gpu_before,'checkpoint_sha256':sha(folder/'model.pt'),'schedule_sha256':sha(O/f'schedule_{case}_{fitseed}.npy'),'eval_opened':False};dump(folder/'summary.json',result)
    state.update(fits_completed=state['fits_completed']+1,last_completed_fit=label);dump(O/'status.json',state);del container,model,optimizer,best;torch.cuda.empty_cache()
 state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(O/'status.json',state)
 summaries=[read(p) for p in sorted(O.glob('*/summary.json'))];assert len(summaries)==16;dump(O/'training_summary.json',{'status':'PASS','fits':16,'summaries':summaries,'eval_opened':False,'formal_updates':sum(x['stopped_step'] for x in summaries),'protocol_sha256':sha(R/'PROTOCOL.md'),'training_source_sha256':sha(pathlib.Path(__file__))});print(json.dumps(state),flush=True)
except Exception:
 state.update(status='failed',traceback=traceback.format_exc(),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(O/'status.json',state);raise
