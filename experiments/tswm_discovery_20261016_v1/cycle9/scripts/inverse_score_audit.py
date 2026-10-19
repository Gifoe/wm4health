"""Native loss contract plus known quadratic-score control; not a model replication."""
import runtime_profile
import ast,contextlib,datetime,hashlib,json,math,pathlib,platform,sys,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'inverse_score';OUT.mkdir(parents=True,exist_ok=True)
class Tee:
 def __init__(self,*files):self.files=files
 def write(self,s):
  for f in self.files:f.write(s);f.flush()
 def flush(self):
  for f in self.files:f.flush()
def native_function():
 source=ROOT/'sources/ac_mtm/train.py';tree=ast.parse(source.read_text(encoding='utf-8'))
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='action_discrimination_loss')
 scope={'torch':torch,'F':F};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
 return scope['action_discrimination_loss'],hashlib.sha256(source.read_bytes()).hexdigest(),ast.get_source_segment(source.read_text(encoding='utf-8'),node)
def cohort(angles,device):
 angles=torch.as_tensor(angles,device=device,dtype=torch.float32)
 v=torch.stack([angles.cos(),angles.sin()],dim=-1)
 signs=torch.tensor([-1.,1.],device=device).repeat_interleave(len(angles))
 v=v.repeat(2,1);w=signs[:,None]*torch.stack([-v[:,1],v[:,0]],dim=-1)
 act=torch.stack([v,w,-v,-w],dim=1);state=torch.zeros((len(act),3),device=device)
 trace=[state.clone()]
 for k in range(4):
  a=act[:,k];dz=(state[:,0]*a[:,1]-state[:,1]*a[:,0])/2
  state=torch.stack([state[:,0]+a[:,0],state[:,1]+a[:,1],state[:,2]+dz],dim=-1);trace.append(state.clone())
 assert torch.max(torch.abs(state[:,:2]))<1e-6
 assert torch.max(torch.abs(state[:,2]-signs))<2e-6
 return act.flatten(1),state,torch.stack(trace,dim=1)
def features(a):
 # Generic degree-two action feature map, no plant-specific cross-product supplied.
 i,j=torch.triu_indices(a.shape[-1],a.shape[-1],device=a.device)
 return a[:,i]*a[:,j]
def main():
 torch.set_num_threads(1);assert torch.cuda.is_available();device='cuda'
 native,source_sha,body=native_function();(OUT/'native_function.py.txt').write_text(body,encoding='utf-8')
 cfg={'seeds':[2021,2022,2023],'updates':600,'lr':.03,'temperature':.1,'train_angles':32,'val_angles':16,'test_angles':20,'plant':'four-step deterministic nonholonomic integrator','scope':'native function-level controlled transfer; not native full-model reproduction'}
 (OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
 env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'source_sha256':source_sha,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
 rows=[];curves=[];started=time.time()
 for seed in cfg['seeds']:
  torch.manual_seed(seed);rng=np.random.default_rng(seed)
  # Each grid includes opposite directions exactly; independent rotations by split.
  splits={name:cohort(np.arange(n)*2*math.pi/n+rng.uniform(0,2*math.pi),device) for name,n in [('train',32),('val',16),('test',20)]}
  a,s,trace=splits['train'];ids=(s[:,2]>0).long()
  q=torch.zeros((2,8),device=device,requires_grad=True)
  loss,_=native(q[ids,None,:],a[:,None,:],temperature=.1);grad=torch.autograd.grad(loss,q)[0]
  # Convexity follows logsumexp of affine q scores after row-constant cancellation.
  probes=[]
  for magnitude in [.01,.1,1.]:
   trial=torch.randn_like(q)*magnitude;probes.append(float(native(trial[ids,None,:],a[:,None,:],temperature=.1)[0]-loss))
  zero={'seed':seed,'loss':float(loss),'chance_logN':math.log(len(a)),'gradient_max':float(grad.abs().max()),'random_query_loss_deltas':probes,'max_class_action_mean':float(torch.stack([a[ids==i].mean(0) for i in range(2)]).abs().max())}
  print('zero-query',json.dumps(zero),flush=True)
  for arm in ['native_distance','generic_quadratic']:
   dim=8 if arm=='native_distance' else features(a).shape[-1]
   # Free per-endpoint-class queries are more expressive than any endpoint MLP here.
   param=torch.nn.Parameter(torch.randn(2,dim,device=device)*.03)
   opt=torch.optim.Adam([param],lr=cfg['lr']);best=None;best_val=math.inf
   def evaluate(split):
    aa,ss,_=splits[split];ii=(ss[:,2]>0).long()
    if arm=='native_distance':ll,acc=native(param[ii,None,:],aa[:,None,:],temperature=.1);logits=-(param[ii,None,:]-aa[None,:,:]).pow(2).mean(-1)/.1
    else:
     ff=features(aa);logits=param[ii]@ff.T/math.sqrt(dim);ll=F.cross_entropy(logits,torch.arange(len(aa),device=device));acc=(logits.argmax(-1)==torch.arange(len(aa),device=device)).float().mean()
    same=(ii[:,None]==ii[None,:]);classmass=(logits.softmax(-1)*same).sum(-1).mean()
    return ll,acc,classmass
   for step in range(cfg['updates']):
    opt.zero_grad();ll,_,_=evaluate('train');ll.backward();opt.step()
    if step%20==0 or step==cfg['updates']-1:
     with torch.no_grad():val=float(evaluate('val')[0])
     curves.append({'seed':seed,'arm':arm,'step':step,'train_loss':float(ll),'val_loss':val})
     if val<best_val:best_val=val;best=param.detach().clone()
   with torch.no_grad():param.copy_(best);test_loss,acc,classmass=evaluate('test')
   row={'seed':seed,'arm':arm,'trainable_query_parameters':param.numel(),'selected_val_loss':best_val,'test_loss':float(test_loss),'test_index_accuracy':float(acc),'test_correct_orientation_mass':float(classmass),'test_chance_logN':math.log(40),'test_equivalence_class_optimum':math.log(20),'test_excess_over_class_optimum':float(test_loss)-math.log(20),'zero_query':zero}
   rows.append(row);print('result',json.dumps(row),flush=True)
   torch.save({'query':param.detach().cpu(),'config':cfg,'source_sha256':source_sha,'row':row},OUT/f'{arm}_seed{seed}.pt')
  np.savez_compressed(OUT/f'cohorts_seed{seed}.npz',**{f'{name}_{key}':tensor.cpu().numpy() for name,tensors in splits.items() for key,tensor in zip(['actions','endpoint','trajectory'],tensors)})
 # Native ACRO normalization contract, not an empirical RL performance claim.
 action=torch.tensor([[1.,0.],[2.,0.],[.25,0.]],device=device)
 normalized=F.normalize(action,dim=1,p=2)
 acro={'input_actions':action.cpu().tolist(),'native_normalized_actions':normalized.cpu().tolist(),'all_positive_collinear_actions_identical':bool(torch.equal(normalized[0],normalized[1]) and torch.equal(normalized[1],normalized[2])),'source':'acro.py:update_encoder lines352-355','scope':'normalization algebra only; no native training or theorem refutation'}
 result={'rows':rows,'acro_normalization_contract':acro,'elapsed_seconds':time.time()-started,'decision':'Generic quadratic score is an existing repair; this diagnostic alone cannot establish a novel method or a native world-model failure.'}
 (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(OUT/'training_curves.json').write_text(json.dumps(curves,indent=2),encoding='utf-8');print('COMPLETE',result['elapsed_seconds'],flush=True)
if __name__=='__main__':
 with (OUT/'stdout.log').open('a',encoding='utf-8') as stdout,(OUT/'stderr.log').open('a',encoding='utf-8') as stderr,contextlib.redirect_stdout(Tee(sys.stdout,stdout)),contextlib.redirect_stderr(Tee(sys.stderr,stderr)):
  try:main()
  except BaseException:traceback.print_exc();raise
