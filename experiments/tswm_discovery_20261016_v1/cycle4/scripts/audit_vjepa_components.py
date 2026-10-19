"""Source-path checks, NOT native training or planning benchmark reproduction.

Only selected unchanged author function/class ASTs are executed. Notebook shell
magic is replaced by a comment before parsing; no Colab, apt, rendering, or main
training side effects run. Native CNN, predictor, loss, and CEM bodies are used.
The original linear experiment is checked separately with its own native classes.
"""
import ast,copy,datetime,hashlib,json,pathlib
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources'/'vjepa'
def definitions(path,names,extra=None,notebook=False):
 text=path.read_text(encoding='utf-8')
 if notebook:text='\n'.join('# notebook shell magic omitted' if line.startswith('!') else line for line in text.splitlines())
 tree=ast.parse(text);nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 ns={'torch':torch,'nn':nn,'deepcopy':copy.deepcopy}
 ns.update(extra or {});exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns
def gradients(loss,params):
 return [torch.zeros_like(p) if g is None else g for p,g in zip(params,torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True))]
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 device=torch.device('cuda')
 out=ROOT/'outputs'/'vjepa_component_audit';out.mkdir(parents=True,exist_ok=True)
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Unchanged published function/class bodies, controlled component checks. No end-to-end native training, rollout accuracy or paper table replication.','revision':'88a90d2c0af970edd46d44d53d4864d10aa6d117','checks':['native 64x64 CNN shape','exp10 variance-term encoder gradients versus VICReg alone','CEM dependence on variance-head parameters','original linear beta-KL encoder gradient dependence','original linear probe dependence on supplied x_next'],'seeds':[41,42,43],'no_new_method_claim':True}
 (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
 constants={'Z_DIM':256,'PRED_HIDDEN':512,'CNN_FLAT':4096,'EMA_TAU':.995,'CEM_HORIZON':10,'CEM_SAMPLES':1024,'CEM_ELITES':100,'CEM_ITERS':3,'CEM_MIN_STD':.05,'DISCOUNT':.99,'USE_AMP':False,'device':device}
 p=SOURCE/'exp10'/'exp10_vjepa_sigma1p0.py'
 ns=definitions(p,['make_enc','VJEPAAgent','JEPAAgent','vicreg','vjepa_world_model_loss','cem_plan'],constants)
 original=SOURCE/'vjepa_Yongchao.py'
 old=definitions(original,['LinearProbabilisticVJEPA','LinearBJEPA','vjepa_prob_loss'],notebook=True)
 rows=[]
 for seed in plan['seeds']:
  torch.manual_seed(seed);agent=ns['VJEPAAgent'](2).to(device)
  shapecheck={'input_shape':[4,3,64,64]}
  try:
   result=agent(torch.randn(4,3,64,64,device=device),torch.randn(4,2,device=device),torch.randn(4,3,64,64,device=device));shapecheck.update(success=True,output_shape=list(result[0].shape))
  except RuntimeError as e:shapecheck.update(success=False,error=str(e))
  assert shapecheck['success'],shapecheck
  x=torch.randn(64,3,64,64,device=device);future=torch.randn_like(x);a=torch.randn(64,2,device=device)
  params=list(agent.enc.parameters())+list(agent.pred_mu.parameters())
  full=ns['vjepa_world_model_loss'](agent,x,a,future)
  mean,lv,z,target=agent(x,a,future);base=ns['vicreg'](mean,target)
  fullgrad=gradients(full,params);basegrad=gradients(base,params)
  error=max((g-h).abs().max().item() for g,h in zip(fullgrad,basegrad))
  lvgrad=gradients(full,list(agent.pred_lv.parameters()));lvnorm=sum(g.square().sum().item() for g in lvgrad)**.5
  scale=max(1.,max(g.abs().max().item() for g in basegrad))
  assert error<=1e-6*scale and lvnorm>0,{'maximum_absolute_difference':error,'gradient_scale':scale,'variance_head_gradient_norm':lvnorm}
  reward=nn.Linear(256,1).to(device);initial=agent.enc(x[:1])[0].detach()
  torch.manual_seed(seed+900);first=ns['cem_plan'](initial,agent,reward,2,horizon=4,n_samples=64,n_elites=8,n_iters=2)
  with torch.no_grad():
   for parameter in agent.pred_lv.parameters():parameter.mul_(100).add_(10)
  torch.manual_seed(seed+900);second=ns['cem_plan'](initial,agent,reward,2,horizon=4,n_samples=64,n_elites=8,n_iters=2)
  planner_error=(first-second).abs().max().item();assert planner_error==0
  model=old['LinearProbabilisticVJEPA'](20,4).to(device)
  linear_x=torch.randn(64,20,device=device);linear_future=torch.randn_like(linear_x)
  sample,pred,tgt,z=model(linear_x,linear_future);oldparams=list(model.encoder.parameters())
  g0=gradients(old['vjepa_prob_loss'](sample,pred,tgt,beta=0),oldparams)
  g1=gradients(old['vjepa_prob_loss'](sample,pred,tgt,beta=1),oldparams)
  beta_error=max((g-h).abs().max().item() for g,h in zip(g0,g1));assert beta_error==0
  probe1=model.get_latent_for_probe(linear_x,linear_future);probe2=model.get_latent_for_probe(linear_x,linear_future*100)
  probe_error=(probe1-probe2).abs().max().item();assert probe_error==0
  row={'seed':seed,'native_cnn':shapecheck,'native_exp10_uncertainty_encoder_gradient_maxdiff':error,'uncertainty_head_gradient_norm':lvnorm,'variance_head_perturbed_cem_action_maxdiff':planner_error,'linear_beta0_vs_beta1_encoder_gradient_maxdiff':beta_error,'linear_probe_future_argument_perturbed_maxdiff':probe_error}
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
 report={'state':'complete','seeds':len(rows),'new_training_runs':0,'native_forward_passes_succeeded':sum(r['native_cnn']['success'] for r in rows),'conclusion':'Released exp10 native CNN accepts its declared input shape. Its native uncertainty term is deliberately detached from encoder and mean predictor, and its mean-only CEM ignores variance. Original linear target-KL supplies no encoder gradient; future probe argument is unused, so no future-input leakage found in that probe. These component facts are not an established new scientific gap or an end-to-end planning performance result.','source_files_sha256':{str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest(),str(original.relative_to(SOURCE)):hashlib.sha256(original.read_bytes()).hexdigest()}}
 (out/'status.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report))
if __name__=='__main__':main()
