"""Check executable prior registration, evaluation dependence and density validity."""
import runtime_profile
import datetime,importlib.util,json,pathlib,sys
from types import SimpleNamespace
import numpy as np,torch
from native_child import ROOT,classes
from piecewise_density import audit
def main():
 torch.set_num_threads(1);out=ROOT/'outputs/native_contracts';out.mkdir(parents=True,exist_ok=True)
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seeds':[81,82,83],'scope':'Native source component audit; no trained representation or paper reproduction claim','checks':['real-world prior registration and one-step parameter update','evaluation path sensitivity to prior removal','synthetic inverse-MLP full-real-line conditional mass'],'novelty':'Jacobian change-of-variables requires the appropriate invertibility/preimage accounting; this principle is established, not new.'}
 (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
 sys.path.insert(0,str(ROOT/'sources/child/realworld'));spec=importlib.util.spec_from_file_location('native_real_child',ROOT/'sources/child/realworld/model/CHiLD.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 rows=[]
 for seed in plan['seeds']:
  torch.manual_seed(seed);cfg=SimpleNamespace(seq_len=24,c_type='type1',layer=[3,7],enc_in=7,is_norm=True,patch_size=5,emb_dim=128,n_concat=1,is_ln=True,activation='gelu',dropout=0.,layer_nums=3,lags=1,gpu=0,kld_weight=1e-7)
  model=module.Model(cfg).cuda();registered={id(p) for p in model.parameters()};prior=[p for group in model.transition_prior_fix.gs_lst for net in group for p in net.parameters()]
  intersection=sum(id(p) in registered for p in prior);before=[p.detach().clone() for p in prior];opt=torch.optim.Adam(model.parameters(),lr=.001)
  x=torch.randn(8,24,7,device='cuda');rec,extra=model(x,True);loss=(rec-x).abs().mean()+extra;assert torch.isfinite(loss);loss.backward();gradmax=max(0. if p.grad is None else p.grad.abs().max().item() for p in prior);opt.step();change=max((p-v).abs().max().item() for p,v in zip(prior,before))
  model.eval();prediction=model(x,False)
  with torch.no_grad():
   for p in prior:p.zero_()
  changed=model(x,False);eval_delta=(changed-prediction).abs().max().item()
  assert intersection==0 and change==0 and gradmax>0 and eval_delta==0
  env=classes();torch.manual_seed(seed);native_net=env['NLayerLeakyMLP'](in_features=3,out_features=1,num_layers=3,hidden_dim=128)
  masses=[]
  for parents in [[0.,0.],[1.,-1.],[-2.,.5]]:masses.append({'parents':parents,**audit(native_net,parents)})
  row={'seed':seed,'real_registered_parameter_tensors':len(registered),'real_prior_parameter_tensors':len(prior),'real_prior_tensors_registered':intersection,'real_prior_nonzero_gradient_max':gradmax,'real_prior_parameter_max_update':change,'real_eval_max_change_after_prior_zero':eval_delta,'synthetic_random_inverse_MLP':masses};rows.append(row)
  (out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
 (out/'status.json').write_text(json.dumps({'state':'complete','seeds':3,'full_training_runs':0,'new_gap_accepted':False}),encoding='utf-8')
if __name__=='__main__':main()
