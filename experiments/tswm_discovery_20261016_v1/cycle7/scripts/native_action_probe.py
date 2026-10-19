"""Controlled action-equivalence probe of unchanged released ActSWM objectives.

This is a new low-dimensional mechanism experiment, not Minecraft reproduction.
Full physical state is the fixed encoder: no latent recovery or pixel bottleneck.
"""
import runtime_profile
import ast,datetime,hashlib,json,math,pathlib,time,types
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources/actswm'
class Config(dict):
 def __getattr__(self,k):return self[k]
def cfg(arm):
 return Config(wm=Config(history_size=1,rollout_steps=3),loss=Config(sigreg=Config(weight=.09),action_effect=Config(weight=.5 if arm in ['hinge','full'] else 0.,margin=.3,mode='cosine'),idm=Config(weight=1. if arm=='full' else 0.,hidden_dim=512,stop_grad=True,pred_weight=1.)))
def load_native():
 env={'torch':torch,'nn':nn,'F':F,'rearrange':rearrange,'logger':types.SimpleNamespace(info=lambda *a,**k:None)}
 for file,selected in [('model/module.py',None),('train/train.py',{'_idm_det_loss','_idm_loss','lejepa_rollout_v2_forward'})]:
  tree=ast.parse((SOURCE/file).read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (selected is None or n.name in selected)]
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE/file),'exec'),env)
 return env
NATIVE=None
class World(nn.Module):
 def __init__(self):
  super().__init__();self.predictor=NATIVE['ARPredictor'](num_frames=1,depth=1,heads=2,mlp_dim=64,input_dim=4,hidden_dim=32,output_dim=4,dim_head=16,dropout=0.,emb_dropout=0.,action_mode='adaln');self.action_encoder=nn.Linear(1,4)
 def encode(self,batch):return {'emb':batch['states'],'act_emb':self.action_encoder(batch['action'])}
 def predict(self,emb,act):return self.predictor(emb,act)
class Harness(nn.Module):
 def __init__(self):
  super().__init__();self.model=World();self.idm=NATIVE['InverseDynamicsModel'](4,1,hidden_dim=512)
  for p in self.idm.parameters():p.requires_grad_(False)
  self.global_step=0;self._rollout_v2_warned=True
 def sigreg(self,emb):
  # The fixed encoder makes SIGReg independent of all trainable parameters.
  # Remove that additive constant, explicitly, rather than claim native SSL.
  return emb.new_zeros(())
 def log_dict(self,*a,**k):pass
 def forward(self,batch,arm):return NATIVE['lejepa_rollout_v2_forward'](self,dict(batch),'train',cfg(arm))
def plant(x,a,kind):
 y=.8*x+torch.tensor([0.,.04,-.03,.02],device=x.device)
 u=a[:,0]
 if kind=='null':force=torch.zeros_like(u)
 elif kind=='linear':force=.35*u
 elif kind=='deadzone':force=.35*u.sign()*F.relu(u.abs()-.5)
 else:raise ValueError(kind)
 y=y.clone();y[:,0]+=force;return y
def cohort(n,kind,seed):
 g=torch.Generator(device='cuda').manual_seed(seed);x=torch.rand(n,4,device='cuda',generator=g)*2-1
 actions=torch.rand(n,4,1,device='cuda',generator=g)*2-1
 # Exact no-ops are in support, so absence of zero-action examples is excluded.
 actions=torch.where(torch.rand(n,4,1,device='cuda',generator=g)<.3,0.,actions)
 states=[x]
 for t in range(3):states.append(plant(states[-1],actions[:,t],kind))
 return {'states':torch.stack(states,1),'action':actions}
def rollout(h,x,actions):
 preds=[];cur=x[:,None]
 for t in range(3):
  cur=h.model.predict(cur[:,-1:],h.model.action_encoder(actions[:,t:t+1]))[:,-1:];preds.append(cur[:,0])
 return torch.stack(preds,1)
def truth(x,actions,kind):
 rows=[]
 for t in range(3):x=plant(x,actions[:,t],kind);rows.append(x)
 return torch.stack(rows,1)
@torch.no_grad()
def metrics(h,batch,kind):
 h.eval();x=batch['states'][:,0];a=batch['action'];pred=rollout(h,x,a);actual=batch['states'][:,1:];zero=torch.zeros_like(a);pz=rollout(h,x,zero);yz=truth(x,zero,kind)
 result={'factual_MSE':float((pred-actual).square().mean()),'zero_plan_MSE':float((pz-yz).square().mean()),'paired_response_MSE':float(((pred-pz)-(actual-yz)).square().mean()),'predicted_cosine_gap':float((1-F.cosine_similarity(pred,pz,dim=-1)).mean()),'true_cosine_gap':float((1-F.cosine_similarity(actual,yz,dim=-1)).mean())}
 if kind in ['null','deadzone']:
  null_a=a*.4;pn=rollout(h,x,null_a);yn=truth(x,null_a,kind);assert torch.equal(yn,yz)
  result.update(null_response_RMS=float((pn-pz).square().mean().sqrt()),null_plan_MSE=float((pn-yn).square().mean()),null_cosine_gap=float((1-F.cosine_similarity(pn,pz,dim=-1)).mean()))
 # Offline candidate ranking, not closed-loop MPC. All physical truth uses the
 # same fixed initial state; none is supplied to model selection or training.
 count=min(256,len(x));x=x[:count];g=torch.Generator(device='cuda').manual_seed(82001)
 plans=torch.rand(count,64,4,1,generator=g,device='cuda')*2-1;plans[:,0]=0.
 xx=x[:,None].expand(-1,64,-1).reshape(-1,4);flat=plans.reshape(-1,4,1);yp=rollout(h,xx,flat)[:,-1];yt=truth(xx,flat,kind)[:,-1]
 goal=torch.zeros(count,4,device='cuda');goal[:,0]=torch.linspace(-.6,.6,count,device='cuda');goal=goal[:,None].expand(-1,64,-1).reshape(-1,4)
 effort=.02*flat[:,:3].square().mean((1,2));cost_pred=((yp-goal).square().mean(-1)+effort).reshape(count,64);cost_true=((yt-goal).square().mean(-1)+effort).reshape(count,64)
 selected=cost_pred.argmin(-1);realized=cost_true.gather(1,selected[:,None])[:,0];regret=realized-cost_true.min(-1).values
 result.update(offline_candidate_regret_mean=float(regret.mean()),offline_candidate_regret_p90=float(torch.quantile(regret,.9)))
 return result
def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def main():
 global NATIVE
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;NATIVE=load_native()
 out=ROOT/'outputs/action_equivalence_pilot';out.mkdir(parents=True,exist_ok=True)
 if (out/'plan.json').exists():raise RuntimeError('Existing run; inspect exact process before any restart')
 pin=json.loads((SOURCE/'SOURCE_PIN.json').read_text())
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'New controlled mechanism experiment with unchanged native rollout/loss/ARPredictor/IDM class bodies; fixed full-state encoder and smaller native architecture. Not original Minecraft reproduction.','source_commit':pin['revision'],'kinds':['null','linear','deadzone'],'arms':['prediction','hinge','full'],'seeds':[31,32,33],'cohort_seed':15001,'train':8192,'validation':1024,'test':2048,'steps':1500,'batch':128,'optimizer':'Native-default AdamW1e-4,weight_decay1e-3; gradientclip1','native_loss_weights':'hinge .5,margin .3; fixed random IDM1,pred_weight1,hidden512','architecture':'Native ARPredictor,adaln,depth1,hidden32,heads2,dim_head16,MLP64;context1,horizon3,state4/action1. Fixed identity state encoder. Native SIGReg removed as trainable-gradient-free constant.','checkpoint':'Primary final fixed1500steps; also retain minimum factual-validation-MSE checkpoint, no test selection','effect_support':'Actions are30%exactzero+70%Uniform[-1,1]; randomized independently of state; no hidden confounding','DGP':'x_next=.8*x+[0,.04,-.03,.02]+force(a)*e0;force=0 or.35a or.35sign(a)relu(abs(a)-.5)','no_new_method':'Existing prediction-only control is a stop/attribution control. This pilot alone cannot establish novelty or general deployed-model failure.'}
 dump(out/'plan.json',plan);rows=[];data_dir=ROOT/'data';data_dir.mkdir(exist_ok=True)
 for ki,kind in enumerate(plan['kinds']):
  train=cohort(8192,kind,15001+ki*100);val=cohort(1024,kind,15002+ki*100);test=cohort(2048,kind,15003+ki*100)
  path=data_dir/f'{kind}_cohorts.npz';np.savez_compressed(path,**{split+'_'+key:value.cpu().numpy() for split,b in [('train',train),('validation',val),('test',test)] for key,value in b.items()});dump(out/f'{kind}_data.json',{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
  for seed in plan['seeds']:
   for arm in plan['arms']:
    run=out/f'{kind}_{arm}_s{seed}';run.mkdir(exist_ok=True);torch.manual_seed(seed);h=Harness().cuda();optimizer=torch.optim.AdamW([p for p in h.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-3)
    gen=torch.Generator(device='cuda').manual_seed(40000+seed);best=math.inf;start=time.perf_counter()
    # Check the independently written evaluator matches the unchanged native
    # multi-step prediction calculation before using any pilot result.
    h.eval();small={k:v[:32] for k,v in val.items()}
    with torch.no_grad():native=h(small,arm)['pred_loss'];reference=(rollout(h,small['states'][:,0],small['action'])-small['states'][:,1:]).square().mean()
    assert torch.allclose(native,reference,atol=1e-7,rtol=1e-6)
    with (run/'train.jsonl').open('w',encoding='utf-8') as log:
     for step in range(1,1501):
      ids=torch.randint(8192,(128,),device='cuda',generator=gen);batch={k:v[ids] for k,v in train.items()};h.train();h.global_step=step;optimizer.zero_grad(set_to_none=True);losses=h(batch,arm);loss=losses['loss'];assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1.);optimizer.step()
      if step%250==0:
       h.eval()
       with torch.no_grad():v=h(val,arm)['pred_loss'].item()
       if v<best:best=v;torch.save(h.state_dict(),run/'best.pt')
       row={'step':step,'loss':loss.item(),'prediction':losses['pred_loss'].item(),'hinge':losses['action_effect_loss'].item(),'idm':losses['idm_loss'].item(),'validation_factual_MSE':v};log.write(json.dumps(row)+'\n');log.flush()
    torch.save(h.state_dict(),run/'final.pt');result={'kind':kind,'arm':arm,'seed':seed,'trainable_parameters':sum(p.numel() for p in h.parameters() if p.requires_grad),'frozen_IDM_parameters':sum(p.numel() for p in h.idm.parameters()),'seconds':time.perf_counter()-start,'minimum_validation_factual_MSE':best,'test_final':metrics(h,test,kind)}
    h.load_state_dict(torch.load(run/'best.pt',weights_only=True));result['test_best_validation']=metrics(h,test,kind);rows.append(result);dump(run/'result.json',result);dump(out/'partial_results.json',rows);dump(out/'status.json',{'state':'running','completed':len(rows),'total':27});print(json.dumps(result),flush=True)
 dump(out/'results.json',rows);dump(out/'status.json',{'state':'complete','completed':27,'accepted_research_gap':False,'objective_achieved':False})
if __name__=='__main__':main()
