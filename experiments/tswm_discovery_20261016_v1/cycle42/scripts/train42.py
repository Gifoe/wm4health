"""Frozen 32-fit ordinary-HMM eligibility screen, no novelty inferred here."""
from common42 import *
import copy,datetime
Q=json.loads((R/'outputs/qualification_v1/qualification.json').read_text());assert Q['status']=='PASS' and Q['protocol_sha256']==sha(R/'PROTOCOL.md') and Q['common_sha256']==sha(R/'scripts/common42.py')
O=R/'outputs/fits_v1';assert not O.exists();O.mkdir(parents=True)
device=Q['chosen_device'];rows=[];all_start=time.perf_counter();state={'status':'running','fits_completed':0,'fits_total':32,'device':device,'protocol_sha256':sha(R/'PROTOCOL.md'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
dump(O/'status.json',state)
for w in range(2):
 data=data_world(w);np.savez_compressed(O/f'world{w}.npz',**data)
 oracle_nll,oracle_p=oracle(data['eval'],data['A'],data['C'],CELLS)
 np.savez_compressed(O/f'oracle{w}.npz',nll=oracle_nll,next_probability=oracle_p)
 for n in [16,128]:
  y_np,v_np,s_np=flatten(data['train'],SEEN,n);N=len(y_np)
  y=torch.tensor(y_np,device=device);v=torch.tensor(v_np,device=device);s=torch.tensor(s_np,device=device);anc=torch.tensor(data['anchor'],device=device)
  for fitseed in [2026421,2026422]:
   generator=np.random.default_rng(fitseed+100*w+10*n);schedule=generator.integers(N,size=(1500,32));np.save(O/f'schedule_w{w}_n{n}_i{fitseed}.npy',schedule)
   for om,dm in [(0,0),(1,0),(0,1),(1,1)]:
    label=f'w{w}_n{n}_i{fitseed}_o{om}d{dm}';folder=O/label;folder.mkdir()
    seed(fitseed);model=Model(om,dm).to(device);opt=torch.optim.Adam(model.parameters(),lr=.01);curves=[];best=None;best_score=float('inf');start=time.perf_counter()
    for step in range(1501):
     if step%250==0:
      model.eval();vnll,_=evaluate(model,data['val'],SEEN,device);value=float(vnll.mean());assert math.isfinite(value)
      if value<best_score:best_score=value;best=copy.deepcopy(model.state_dict());selected=step
      curves.append({'step':step,'seen_val_future_nll':value,'best':best_score});model.train()
      print(json.dumps({'fit':label,**curves[-1]}),flush=True)
     if step==1500:break
     idx=torch.tensor(schedule[step],device=device);opt.param_groups[0]['lr']=.001+.009*.5*(1+math.cos(math.pi*step/1500))
     opt.zero_grad(set_to_none=True);loss=train_batch(model,y[idx],v[idx],s[idx],anc,N);assert torch.isfinite(loss)
     loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
    elapsed=time.perf_counter()-start
    torch.save({'final':model.state_dict(),'selected':best,'obs_mod':om,'dyn_mod':dm,'selected_step':selected,'protocol_sha256':sha(R/'PROTOCOL.md')},folder/'model.pt')
    model.load_state_dict(best);model.eval();nll,nextp=evaluate(model,data['eval'],CELLS,device)
    with torch.no_grad():
     vv=torch.tensor([x[0] for x in CELLS],device=device);ss=torch.tensor([x[1] for x in CELLS],device=device);a,c=model.parameters_for(vv,ss)
    kl=np.sum(oracle_p*np.log(oracle_p/nextp),axis=-1);assert np.isfinite(nll).all() and np.isfinite(kl).all() and kl.min()>-1e-6
    np.savez_compressed(folder/'scores.npz',nll=nll,next_probability=nextp,oracle_next_kl=kl,learned_A=a.cpu().numpy(),learned_C=c.cpu().numpy())
    for group in ['seen','heldout']:
     ix=[j for j,x in enumerate(CELLS) if (x in SEEN)==(group=='seen')]
     rows.append({'world':w,'n_per_seen_cell':n,'init_seed':fitseed,'obs_mod':om,'dyn_mod':dm,'group':group,'future_nll':float(nll[ix].mean()),'oracle_future_nll':float(oracle_nll[ix].mean()),'excess_nll':float((nll[ix]-oracle_nll[ix]).mean()),'next_symbol_oracle_kl':float(kl[ix].mean()),'selected_step':selected,'nominal_parameters':sum(p.numel() for p in model.parameters()),'active_parameters':model.active_parameters(),'training_seconds_including_seen_validation':elapsed,'score_sha256':sha(folder/'scores.npz'),'checkpoint_sha256':sha(folder/'model.pt')})
    dump(folder/'curves.json',curves);dump(folder/'summary.json',rows[-2:]);state.update(fits_completed=state['fits_completed']+1,last_fit=label);dump(O/'status.json',state)
dump(O/'results.json',{'status':'PASS','rows':rows,'fits':32,'updates_per_fit':1500,'formal_updates':48000,'synthetic_evaluation_scored':True,'old_real_test_scored':False,'novel_method_fits':0,'all_models':'ordinary conditional HMM','worlds':2,'init_seeds':[2026421,2026422],'protocol_sha256':sha(R/'PROTOCOL.md'),'common_sha256':sha(R/'scripts/common42.py'),'training_source_sha256':sha(pathlib.Path(__file__)),'total_seconds':time.perf_counter()-all_start})
state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(O/'status.json',state);print(json.dumps(state),flush=True)
