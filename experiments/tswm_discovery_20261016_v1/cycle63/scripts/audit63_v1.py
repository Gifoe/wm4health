from native63 import *
import math,platform
Core,Harness,scope=load();x,y,data=fixture();pp=paths();records=[];saved={};datafile=pp['data_root']/'training_region_fixture63_v1.npz';np.savez_compressed(datafile,x=x.numpy(),y=y.numpy());dump(OUT/'FIXTURE63_V1.json',{**data,'G_copy':str(datafile),'G_copy_sha256':sha(datafile)})
for s in [2021,2022,2023]:
 seed(s);m=Core(config()).double().eval();h=Harness(m);weights=pp['checkpoint_root']/('untrained_'+str(s)+'.pt');torch.save(m.state_dict(),weights)
 # Native finite forward/backward qualification, without optimizer creation or updates.
 batch=types.SimpleNamespace(past_target_cdf=x,future_target_cdf=y)
 seed(63000);loss=h.loss(batch);assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);m.zero_grad(set_to_none=True);m.eval()
 # Native sampler components are collected without changing any returned object.
 native_decode=m.decode;trace=[]
 def traced_decode(a,b):
  rec,dist=native_decode(a,b);trace.append((dist.loc.detach().clone(),dist.scale.detach().clone()));return rec,dist
 m.decode=traced_decode
 with torch.no_grad():
  seed(63101);out20=m.sample(x,20);c20=trace.copy();trace.clear()
  seed(63101);out40=m.sample(x,40);c40=trace.copy();trace.clear()
 m.decode=native_decode
 assert len(c20)==len(c40)==20
 loc=torch.stack([a for a,b in c20]);scale=torch.stack([b for a,b in c20]);assert all(torch.equal(a,c) and torch.equal(b,d) for (a,b),(c,d) in zip(c20,c40));assert torch.equal(out20,loc)
 between=loc.var(0,unbiased=False);within=scale.square().mean(0);full=between+within;assert (within>0).all()
 saved[str(s)+'_component_location']=loc.numpy();saved[str(s)+'_component_scale']=scale.numpy();saved[str(s)+'_sample20']=out20.numpy();saved[str(s)+'_sample40']=out40.numpy()
 # Independent estimator-only interventions leave all predictive weights frozen.
 original={k:v.detach().clone() for k,v in m.estimatenoise.state_dict().items()};interventions=[];reference=None
 for alpha in [1.,.1,.01,0.]:
  m.estimatenoise.load_state_dict({k:v*alpha for k,v in original.items()});m.train();seed(63202)
  rec,prior,post,rs,rf=m(x);ss=torch.sigmoid(10*(1e-5-rs.abs())).mean();sf=torch.sigmoid(10*(1e-5-rf.abs())).mean();aux=.001*(1-ss)+.01*torch.maximum(torch.tensor(.2,dtype=x.dtype),ss-sf)
  base=F.mse_loss(rec,x)+F.mse_loss(post.loc,y)+( -post.log_prob(y).mean() if -post.log_prob(y).mean()<100 else 0)+.01*h.kld_loss(prior)
  seed(63202);actual_loss=h.loss(batch);assert torch.allclose(actual_loss,base+aux,atol=1e-10,rtol=1e-10)
  current=[rec.detach(),post.loc.detach(),post.scale.detach(),prior.loc.detach(),prior.covariance_matrix.detach()]
  if reference is None:reference=[a.clone() for a in current]
  delta=max(float((a-b).abs().max()) for a,b in zip(current,reference));assert delta==0.,(s,alpha,delta)
  m.zero_grad(set_to_none=True);aux.backward();est_norm=sum(float(p.grad.square().sum()) for p in m.estimatenoise.parameters() if p.grad is not None)**.5;encoder_norm=sum(float(p.grad.square().sum()) for p in m.encoder.parameters() if p.grad is not None)**.5
  saved[str(s)+'_residual_s_'+str(alpha)]=rs.detach().numpy();saved[str(s)+'_residual_f_'+str(alpha)]=rf.detach().numpy()
  row={'estimator_weight_multiplier':alpha,'native_total_loss':float(actual_loss.detach()),'unchanged_predictive_loss':float(base.detach()),'auxiliary_penalty':float(aux.detach()),'coarse_named_sparsity':float(ss.detach()),'other_named_sparsity':float(sf.detach()),'predictive_output_max_difference':delta,'aux_grad_estimator_norm':est_norm,'aux_grad_encoder_norm':encoder_norm};interventions.append(row)
  if alpha==0.:assert torch.count_nonzero(rs)==torch.count_nonzero(rf)==0 and est_norm==0. and encoder_norm==0.
 assert interventions[-1]['auxiliary_penalty']<=interventions[0]['auxiliary_penalty']
 m.estimatenoise.load_state_dict(original);m.eval()
 # A posterior/prior difference is measured, not classified as a defect.
 with torch.no_grad():
  _,pred,prior,_,_=m.encode(x);diag=prior.covariance_matrix.diagonal(dim1=-2,dim2=-1);kl=.5*(diag+prior.loc.square()-1-diag.log()).sum(-1)
 # Pure translation check of this internal decoder parameterization; no whole-pipeline claim.
 with torch.no_grad():
  seed(63303);m.encode(x);_,base_dist=m.decode(torch.zeros((4,4,128),dtype=x.dtype),torch.zeros((4,30,128),dtype=x.dtype));bm=base_dist.loc.clone();bs=base_dist.scale.clone()
  shift=torch.tensor([[-5.,-2.,-1.,0.,1.,2.,5.]],dtype=x.dtype).reshape(1,1,7)
  m.encode(x+shift);_,shift_dist=m.decode(torch.zeros((4,4,128),dtype=x.dtype),torch.zeros((4,30,128),dtype=x.dtype));translation_mu_error=float((shift_dist.loc-bm-shift).abs().max());translation_scale_change=float((shift_dist.scale-bs).abs().mean())
 records.append({'untrained_seed':s,'weight_path':str(weights),'weights_sha256':sha(weights),'parameters':sum(p.numel() for p in m.parameters()),'finite_native_loss_backward':True,'sample_count_law':{'component_parameters_exactly_equal':True,'native_N20_equals_component_locations':True,'mean_between_component_variance':float(between.mean()),'mean_missing_within_component_variance':float(within.mean()),'mean_full_mixture_variance':float(full.mean()),'official_config_N100_enters_special_branch':False},'estimator_interventions':interventions,'variational_posterior_to_standard_prior_mean_KL':float(kl.mean()),'prior_difference_is_not_by_itself_a_defect':True,'internal_translation_probe':{'mean_translation_error':translation_mu_error,'mean_scale_change':translation_scale_change,'scope':'fixed normalized decoder coordinates only; external-scaler and real heteroscedasticity caveats apply'}})
 print(json.dumps({'seed':s,'special_sample_branch_within_variance_omitted':float(within.mean()),'regularizer_before':interventions[0]['auxiliary_penalty'],'regularizer_zero_head':interventions[-1]['auxiliary_penalty'],'predictive_change_zero_head':interventions[-1]['predictive_output_max_difference']}),flush=True)
arrays=pp['artifact_root']/'AUDIT_TRACES63_V1.npz';np.savez_compressed(arrays,**saved)
dump(OUT/'AUDIT63_V1.json',{'utc':utc(),'status':'NATIVE_COMPONENT_AUDIT_EXECUTED','source_revision':'a53409aa6d97c65b5b36b7f6b27fc5fbdb5e3df1','protocol_sha256':sha(ROOT/'PROTOCOL63_V1.md'),'source_receipt_sha256':sha(OUT/'SOURCES63_V1.json'),'scope':scope,'config':vars(config()),'dtype':'float64','device':'CPU','torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'trained_models':0,'optimizer_updates':0,'heldout_scores':0,'records':records,'traces':str(arrays),'traces_sha256':sha(arrays),'accepted_final_directions':0})
print(json.dumps({'status':'NATIVE_COMPONENT_AUDIT_EXECUTED','optimizer_updates':0,'weight_fixtures':len(records),'traces_sha256':sha(arrays)}))
