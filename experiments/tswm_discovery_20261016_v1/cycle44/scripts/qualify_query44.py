"""Qualify only the new read-only cache and coarse marginal paths on VAL."""
from query_common44 import *
import datetime,traceback
O=R/'outputs/query_qualification_v1';assert not O.exists();O.mkdir(parents=True)
training=read(R/'outputs/fits_v1/training_summary.json');assert training['status']=='PASS' and training['fits']==16
try:
 checks=[]
 for family in ['lgssm','ar_mixture','spf_native','spf_condition_trainable']:
  summary=next(s for s in training['summaries'] if s['case']=='gaussian' and s['family']==family and s['seed']==SEEDS[0]);model,module=selected_model(summary);before=checksum(module);history=np.load(R/'data/gaussian/val.npz')['observations'][:1,:H]
  plain=sample_model(family,model,history,64,202644850)
  coarse_plain=sample_spf(model,history,64,202644850,timescale='yearly') if family.startswith('spf') else None
  with prefix_cache(family,model):
   warm_prefix(family,model,history);cached=sample_model(family,model,history,64,202644850)
   delta=float((cached-plain).abs().max());assert delta==0,(family,delta)
   if coarse_plain is not None:
    coarse_cached=sample_spf(model,history,64,202644850,timescale='yearly');coarse_delta=float((coarse_cached-coarse_plain).abs().max());assert coarse_delta==0
   else:coarse_delta=None
  row={'family':family,'warmed_cache_fine_max_abs':delta,'warmed_cache_coarse_max_abs':coarse_delta,'checkpoint_sha256':summary['checkpoint_sha256']}
  if family=='lgssm':
   # Independent block-covariance construction; compare full 64-field mean/cov.
   m0,P0,_=model.filter(torch.as_tensor(history,device=model.U.device));A,Q,_,r=model.matrices();a=A.detach().double().cpu().numpy();q=Q.detach().double().cpu().numpy();p=P0.detach().double().cpu().numpy();m=m0[0].detach().double().cpu().numpy();loading=(model.U*model.scale).detach().double().cpu().numpy();means=[];ps=[]
   for _ in range(FUT):m=a@m;p=a@p@a.T+q;means.append(m);ps.append(p.copy())
   total=np.zeros((model.d,model.d))
   for j in range(FUT):
    for i in range(FUT):
     total+=np.linalg.matrix_power(a,i-j)@ps[j] if i>=j else ps[i]@np.linalg.matrix_power(a.T,j-i)
   exact_mean=np.mean(means,0)@loading.T;exact_cov=loading@total@loading.T/FUT**2+float(r.detach())/FUT*np.eye(P)
   mean,L,noise=coarse_parameters_ssm(model,history);actual_mean=mean.detach().double().cpu().numpy()[0]@loading.T;actual_cov=loading@(L@L.T).detach().double().cpu().numpy()@loading.T+float(noise.detach())/FUT*np.eye(P)
   mean_error=float(np.max(np.abs(exact_mean-actual_mean)));cov_error=float(np.max(np.abs(exact_cov-actual_cov)));assert mean_error<5e-6 and cov_error<5e-6,(mean_error,cov_error)
   spec=read(R/'data/gaussian/functional_spec.json');moments=ssm_query_moments(model,history);analytic=ssm_analytic_readout(moments,spec);values=ssm_marginal_readout(moments,analytic,spec,64,202644851);assert values.shape==(64,5) and torch.isfinite(values).all()
   row.update(independent_coarse_field_mean_max_abs=mean_error,independent_coarse_field_covariance_max_abs=cov_error)
  assert checksum(module)==before;checks.append(row);print(json.dumps(row),flush=True);del model,module;torch.cuda.empty_cache()
 result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'eval_opened':False,'new_training_updates':0,'source_sha256':sha(pathlib.Path(__file__)),'query_common_sha256':sha(R/'scripts/query_common44.py')};dump(O/'qualification.json',result)
except Exception:
 dump(O/'qualification.json',{'status':'FAIL','traceback':traceback.format_exc(),'eval_opened':False});raise
