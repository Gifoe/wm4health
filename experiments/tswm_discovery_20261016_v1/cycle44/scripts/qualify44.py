"""Only changed critical paths; no EVAL data or quality conclusions."""
from common44 import *
import datetime,traceback,shutil
O=R/'outputs/qualification_v1';assert not O.exists();O.mkdir(parents=True);result={'status':'running','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};dump(O/'qualification.json',result)
try:
 seed(202644);es_samples=torch.randn(2,32,FUT,P,device='cuda');es_truth=torch.randn(2,FUT,P,device='cuda');es_value=energy(es_samples,es_truth).cpu().numpy();sx=es_samples.cpu().numpy().astype('float64').reshape(2,32,-1);ty=es_truth.cpu().numpy().astype('float64').reshape(2,-1);independent_es=[]
 for j in range(2):independent_es.append((np.linalg.norm(sx[j]-ty[j],axis=-1).mean()-sum(np.linalg.norm(sx[j,a]-sx[j,b]) for a in range(32) for b in range(a+1,32))/(32*31))/math.sqrt(FUT*P))
 es_error=float(np.max(np.abs(es_value-independent_es)));assert es_error<2e-6,es_error
 train=np.load(R/'data/gaussian/train.npz')['observations'];pca=dict(np.load(R/'data/gaussian/pca.npz'));seed(2026440);ctx=make_spf(False);opt=torch.optim.AdamW(ctx.model.parameters(),lr=3e-4,weight_decay=1e-4)
 import emulator.data.loading.collate as native_collate
 native_collate._WORKER_GATES.clear();seed(20264400)
 for step in range(20):
  opt.zero_grad(set_to_none=True);loss=native_loss(ctx,torch.tensor(train[:24]));assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),1.);opt.step()
 repaired=copy.copy(ctx);repaired.model=copy.deepcopy(ctx.model);enable_condition_grad(repaired.model);repaired.parameters=repaired.model.parameters
 packed=h43.collate_fn_spf(native_examples(torch.tensor(train[24:48])),h43.collate_config(),ctx.noise_scheduler,ctx.datamodule_config.output_seq_len,True);packed=h43.to_device(packed,ctx.device);arguments={'sample':packed['noisy_latents_list'],'condition_sample':packed['cond_inputs_list'],'timestep_ratio':packed['timesteps'],'latent_size':packed['latent_size'],'latent_time':packed['latent_time'],'class_labels':None}
 ctx.model.eval();repaired.model.eval()
 with torch.no_grad():
  original_values=ctx.model(**arguments);repaired_values=repaired.model(**arguments)
 delta=max(float((a-b).abs().max()) for a,b in zip(original_values,repaired_values));assert delta==0
 ctx.model.zero_grad(set_to_none=True);repaired.model.zero_grad(set_to_none=True);ctx.training_step(packed,0)['loss'].backward();repaired.training_step(packed,0)['loss'].backward()
 original_grad=ctx.model.cond_pos_embed.proj.weight.grad;new_grad=repaired.model.cond_pos_embed.proj.weight.grad;assert original_grad is None and new_grad is not None and torch.isfinite(new_grad).all() and new_grad.abs().max()>0
 history=train[:2,:H];xs,meta=native_inputs(torch.tensor(history,device='cuda').repeat_interleave(4,0),'cuda');g=torch.Generator(device='cuda').manual_seed(9988)
 with torch.no_grad():direct,_=ctx.predict_modular(xs,metadata=meta,timescale='monthly',rng=g,latent_size=[1,2,4]);explicit=sample_spf(ctx,history,4,9988);root_delta=float((direct.reshape_as(explicit)-explicit).abs().max());assert root_delta==0
 anti=sample_spf(ctx,history,4,9988,antithetic=True);assert anti.shape==explicit.shape and torch.isfinite(anti).all()
 del repaired,opt,ctx;torch.cuda.empty_cache()
 seed(2026440);ar=AR(pca).cuda();optimizer=torch.optim.AdamW(ar.parameters(),lr=3e-4);batch=torch.tensor(train[:12],device='cuda')
 for _ in range(5):optimizer.zero_grad();value=ar.loss(batch);assert torch.isfinite(value);value.backward();optimizer.step()
 ar.eval();target=ar.project(batch[:3,1]);features,_=ar.gru(ar.project(batch[:3,:1]));logits,means,L=ar.params(features[:,0]);actual=ar.log_prob(target,logits,means,L).detach().cpu().numpy();lw=logits.detach().cpu().numpy().astype('float64');mu=means.detach().cpu().numpy().astype('float64');ell=L.detach().cpu().numpy().astype('float64');yy=target.detach().cpu().numpy().astype('float64');independent=[]
 for j in range(3):
  terms=[];logweights=lw[j]-np.log(np.exp(lw[j]-lw[j].max()).sum())-lw[j].max()
  for k in range(4):
   solved=np.linalg.solve(ell[j,k],yy[j]-mu[j,k]);terms.append(logweights[k]-.5*(ar.d*math.log(2*math.pi)+solved@solved)-np.log(np.diag(ell[j,k])).sum())
  terms=np.asarray(terms);independent.append(terms.max()+np.log(np.exp(terms-terms.max()).sum()))
 mixture_error=float(np.max(np.abs(actual-independent)));assert mixture_error<2e-5,mixture_error
 for anti_flag in [False,True]:assert torch.isfinite(ar.sample(history,4,9000,anti_flag)).all()
 del ar,optimizer;torch.cuda.empty_cache()
 seed(2026440);ssm=LGSSM(pca,train).cuda();optimizer=torch.optim.Adam(ssm.parameters(),lr=.005)
 for _ in range(5):optimizer.zero_grad();value=ssm.loss(batch);assert torch.isfinite(value);value.backward();optimizer.step();ssm.stabilize()
 ssm=ssm.double();short=batch[:3,:4].double();A,Q,P0,rr=ssm.matrices();an,qn,pn,rn=[x.detach().cpu().numpy() for x in [A,Q,P0,rr]];u=ssm.U.cpu().numpy();scale=ssm.scale.cpu().numpy();y=short.cpu().numpy();pc=y@u/scale;D=np.diag(rn/scale**2);mean=np.zeros((3,ssm.d));cov=pn.copy();expected=[]
 for t in range(4):
  if t:mean=mean@an.T;cov=an@cov@an.T+qn
  S0=cov+D;innovation=pc[:,t]-mean;sign,ld=np.linalg.slogdet(S0);assert sign>0;tail=y[:,t]-(pc[:,t]*scale)@u.T
  expected.append(.5*(ssm.d*math.log(2*math.pi)+ld+np.einsum('bi,ij,bj->b',innovation,np.linalg.inv(S0),innovation))+np.log(scale).sum()+.5*((P-ssm.d)*np.log(2*math.pi*rn)+(tail**2).sum(-1)/rn))
  gain=cov@np.linalg.inv(S0);mean+=innovation@gain.T;left=np.eye(ssm.d)-gain;cov=left@cov@left.T+gain@D@gain.T
 actual=ssm.filter(short,True)[2].detach().cpu().numpy();filter_error=float(np.max(np.abs(actual-np.stack(expected,axis=1))));assert filter_error<1e-8,filter_error
 ssm.zero_grad();value=ssm.loss(short);value.backward();gradient=float(ssm.logR.grad);old=float(ssm.logR);vals=[]
 for sign in [1,-1]:ssm.logR.data.fill_(old+sign*1e-5);vals.append(float(ssm.loss(short)))
 ssm.logR.data.fill_(old);gradient_error=abs((vals[0]-vals[1])/2e-5-gradient);assert gradient_error<1e-6,gradient_error
 ssm=ssm.float()
 for anti_flag in [False,True]:assert torch.isfinite(ssm.sample(history,4,9000,anti_flag)).all()
 result.update(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_energy_max_abs=es_error,native_autograd_only_forward_delta=delta,native_condition_gradient_is_none=original_grad is None,ordinary_condition_gradient_max=float(new_grad.abs().max()),explicit_root_vs_native_max_abs=root_delta,independent_mixture_logprob_max_abs=mixture_error,independent_Kalman_nll_max_abs=filter_error,finite_difference_logR_gradient_abs=gradient_error,discarded_updates={'spf':20,'ar':5,'lgssm':5},eval_opened=False,common_sha256=sha(R/'scripts/common44.py'),protocol_sha256=sha(R/'PROTOCOL.md'),free_disk_GiB=shutil.disk_usage(R).free/2**30,source_sha256=sha(pathlib.Path(__file__)))
 dump(O/'qualification.json',result);print(json.dumps(result),flush=True)
except Exception:
 result.update(status='FAIL',traceback=traceback.format_exc());dump(O/'qualification.json',result);raise
