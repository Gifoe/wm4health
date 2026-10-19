"""Falsify implementation explanations before interpreting native model failure."""
import datetime,json,pathlib
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from native_cofe import ROOT,build
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'outputs'/'cofe_contracts';out.mkdir(parents=True,exist_ok=True)
 dump(out/'plan.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':'a53409aa6d97c65b5b36b7f6b27fc5fbdb5e3df1','scope':'Native full model/loss/sampler class bodies and stsf ETTh1 configuration; component checks, not trained forecasting evidence','seeds':[41,42,43],'checks':['forward/loss/backward finite','auxiliary head zeroing with forecast preservation','core loss gradients after auxiliary zeroing','internal location-scale mapping under location shift','sample-count dependence'],'semantic_caution':'A context-conditioned training latent versus standard-normal generation can be valid variational inference. A difference alone is not called a bug. External train-fitted scaling may cancel physical unit shifts; internal RevIN probe is not a full-pipeline unit-invariance claim.'})
 rows=[]
 for seed in [41,42,43]:
  model=build(seed=seed);torch.manual_seed(seed+1000);x=torch.randn(8,24,7,device='cuda');y=torch.randn_like(x)
  batch=SimpleNamespace(past_target_cdf=x,future_target_cdf=y)
  torch.manual_seed(seed+100);loss_before=model.loss(batch);assert torch.isfinite(loss_before)
  loss_before.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  model.eval();torch.manual_seed(seed+200)
  before=model(x)
  with torch.no_grad():
   for p in model.model.estimatenoise.parameters():p.zero_()
  torch.manual_seed(seed+200);after=model(x)
  mean_diff=(before[2].loc-after[2].loc).abs().max().item();scale_diff=(before[2].scale-after[2].scale).abs().max().item()
  assert mean_diff==0 and scale_diff==0
  torch.manual_seed(seed+100);zero_loss=model.loss(batch)
  assert zero_loss<=loss_before+1e-5
  core=[p for name,p in model.named_parameters() if 'estimatenoise' not in name]
  ga=torch.autograd.grad(zero_loss,core,allow_unused=True,retain_graph=True)
  l1,con=model.weight_l1,model.weight_con;model.weight_l1=0;model.weight_con=0
  torch.manual_seed(seed+100);without=model.loss(batch);gb=torch.autograd.grad(without,core,allow_unused=True)
  gradient_diff=max((a-b).abs().max().item() for a,b in zip(ga,gb) if a is not None and b is not None)
  assert gradient_diff<1e-7;model.weight_l1=l1;model.weight_con=con
  model.eval();mus=[];scales=[];correct=[]
  with torch.no_grad():
   for offset in [0.,5.]:
    torch.manual_seed(seed+300);rec,pred,q,rs,rf=model.model.encode(x+offset)
    sampled=pred+q.rsample();raw=model.model.decoder_sigma(sampled).reshape(8,3,8,7).reshape(8,24,7)
    expected=F.softplus(raw)*model.model.revin.stdev/(model.model.revin.affine_weight.abs()+1e-10)+1e-6
    _,d=model.model.decode(rec,sampled);mus.append(d.loc);scales.append(d.scale);correct.append(expected)
  location_error=(mus[1]-mus[0]-5).abs().max().item();native_scale_change=(scales[1]-scales[0]).abs().mean().item();proper_scale_change=(correct[1]-correct[0]).abs().max().item()
  assert location_error<1e-4 and proper_scale_change<1e-4 and native_scale_change>.1
  torch.manual_seed(seed+400);s10=model.forecast(batch,num_samples=10)
  torch.manual_seed(seed+400);s100=model.forecast(batch,num_samples=100)
  row={'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'loss_before_zero_aux':loss_before.item(),'loss_after_zero_aux':zero_loss.item(),'forecast_mean_max_change_zero_aux':mean_diff,'forecast_scale_max_change_zero_aux':scale_diff,'core_gradient_max_change_after_aux_zero_vs_noaux':gradient_diff,'internal_location_shift_mean_error':location_error,'native_scale_mean_absolute_change_under_plus5':native_scale_change,'proper_scale_max_change_under_plus5':proper_scale_change,'sample10_average_variance':s10.var(dim=1,unbiased=False).mean().item(),'sample100_average_variance':s100.var(dim=1,unbiased=False).mean().item(),'estimated_noise_residual_max_after_zero':max(after[3].abs().max().item(),after[4].abs().max().item())}
  rows.append(row);dump(out/'results.json',rows);print(json.dumps(row),flush=True)
 dump(out/'status.json',{'state':'complete','seeds':3,'trained_models':0,'new_research_gap_accepted':False,'findings':'Free auxiliary noise estimator can attain its regularizer minimum without changing forecasts or teaching the core. Internal dispersion de-normalization adds the context mean before softplus. Special sample-count branch returns component locations instead of observation draws. These are implementation/objective-path explanations to resolve before scientific attribution.'})
if __name__=='__main__':main()
