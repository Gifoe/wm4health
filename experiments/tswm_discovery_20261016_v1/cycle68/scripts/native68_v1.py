import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common68_v1 import *
import torch,types,importlib,inspect,ast,textwrap,argparse,time
sys.path.insert(0,str(PROJECT/'cycle66'/'scripts'))
import native66_v1 as n66
class NormWear:
 def __init__(self):
  self.model,self.module,self.receipt=n66.load('cuda');self.lb=importlib.import_module('normwear66_native.latent_bayesian');self.original_sampler=self.lb.get_traj_of_state
  tr=ast.parse(textwrap.dedent(inspect.getsource(self.module.NormWear2.simulate)));n=0
  for v in ast.walk(tr):
   if isinstance(v,ast.Assign) and isinstance(v.targets[0],ast.Subscript) and isinstance(v.targets[0].value,ast.Name) and v.targets[0].value.id=='pad_context_tensor' and isinstance(v.value,ast.Subscript) and isinstance(v.value.value,ast.Name) and v.value.value.id=='dec_out':
    for s in [v.targets[0],v.value]:s.slice.elts[1]=ast.Name(id='non_visible_channel',ctx=ast.Load());n+=1
  assert n==2;ast.fix_missing_locations(tr);scope=dict(self.module.__dict__);exec(compile(tr,'<C66_hidden_only_update68>','exec'),scope);self.simulate=types.MethodType(scope['simulate'],self.model)
  tr=ast.parse(textwrap.dedent(inspect.getsource(self.original_sampler)));n=0
  for v in ast.walk(tr):
   if isinstance(v,ast.Assign) and isinstance(v.targets[0],ast.Name) and v.targets[0].id=='centroids_action_emb':v.value=ast.Call(func=ast.Name(id='_centroids68',ctx=ast.Load()),args=[ast.Name(id='centroids',ctx=ast.Load())],keywords=[]);n+=1
  assert n==1;ast.fix_missing_locations(tr);scope=dict(self.lb.__dict__);scope['_centroids68']=self.centroids;exec(compile(tr,'<C66_index_or_C67_same_view68>','exec'),scope);self.control_sampler=scope['get_traj_of_state'];self.sampler_AST_sha256=hashlib.sha256(ast.unparse(tr).encode()).hexdigest()
 def centroids(self,C):
  if self.mode=='bayes_indexed':return C[:,1::2]
  assert self.mode=='bayes_same_view';labels=np.argmin(((self.E[:,None]-C[None])**2).sum(-1),1);assert len(np.unique(labels))==len(C);return np.stack([self.action_E[labels==i].mean(0) for i in range(len(C))])
 def set_case(self,history,known):
  self.history=np.asarray(history,dtype='float64');self.known=np.asarray(known,dtype='float64');mu,sd=norm(history);self.mu=mu;self.sd=sd;raw=np.c_[history,np.vstack([np.zeros(H),known])];padding=(-raw.shape[-1])%P;raw=np.c_[raw,np.vstack([np.zeros(padding),np.repeat(known[-1],padding)])];self.x=torch.tensor(raw[None],dtype=torch.float32,device='cuda');self.z=torch.tensor(((raw-mu)/sd)[None],dtype=torch.float32,device='cuda');self.z[:,0,L:]=0
  with torch.inference_mode():
   self.enc,self.ids,self.mask=self.model.forward_encoder(self.z[:,:,:L],masking=False);self.E=self.enc.permute(1,2,0).flatten(1).cpu().numpy();a=self.model.forward_encoder(self.z[:,[1],:],masking=False)[0];self.action_E=a.permute(1,2,0).flatten(1).cpu().numpy()[:L//P]
  assert self.E.shape==(L//P,1536)
 def decode(self,future_E):
  E=np.r_[self.E,np.asarray(future_E)];x=torch.tensor(E.reshape(len(E),768,2).transpose(2,0,1),dtype=torch.float32,device='cuda')
  with torch.inference_mode():y=self.model.forward_decoder(x,self.ids,self.mask).reshape(2,-1)[0,L:L+H]
  assert y.shape==(H,);return y.cpu().numpy().astype('float64')*self.sd[0,0]+self.mu[0,0]
 def reconstruction(self,E=None):
  E=self.E if E is None else E;x=torch.tensor(E.reshape(len(E),768,2).transpose(2,0,1),dtype=torch.float32,device='cuda')
  with torch.inference_mode():y=self.model.forward_decoder(x,self.ids,self.mask).reshape(2,-1)[0,:L]
  return y.cpu().numpy().astype('float64')*self.sd[0,0]+self.mu[0,0]
 def forecast(self,mode,seed=2021):
  torch.manual_seed(seed);np.random.seed(seed);self.mode=mode
  with torch.inference_mode():
   if mode=='native_predict':y=self.model.predict(torch.tensor(self.history.T[None],dtype=torch.float32,device='cuda'),H,max_pred_length=16)[0,:H,0]
   elif mode=='clamped_simulate':y=self.simulate(self.x.permute(0,2,1).clone(),all_visible_length=L,non_visible_channel=[0],ar_step=16)[0,L:L+H,0]
   else:
    self.lb.get_traj_of_state=self.control_sampler
    try:y=self.model.latent_bayesian_transition(self.z,n_channels=2,physio_channels=[0],context_length=L,pred_length=H,num_states=10,action_channels=[1],condition_bayes=True,num_traj_sampled=4,latent_encoder=self.model)[0]*self.sd[0,0]+self.mu[0,0]
    finally:self.lb.get_traj_of_state=self.original_sampler
  assert y.shape==(H,) and torch.isfinite(y).all();return y.cpu().numpy().astype('float64')
class MetaID:
 def __init__(self):
  receipt=read(OUT/'MODEL_FILES68_V1.json');base=pathlib.Path(receipt['server_directory'])
  for f in receipt['files']:assert sha(f['path'])==f['sha256']
  import importlib.util
  spec=importlib.util.spec_from_file_location('sysid68_native',base/'transformer_sim.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
  with torch.serialization.safe_globals([argparse.Namespace]):checkpoint=torch.load(base/'ckpt_430_fine-tuned_3100000_lower_noise.pt',map_location='cpu',weights_only=True)
  self.checkpoint_keys=list(checkpoint);self.args=checkpoint['model_args'];self.cfg=checkpoint['cfg'];conf=mod.Config(**self.args);self.model=mod.TSTransformer(conf);state=checkpoint['model'];assert set(state)==set(self.model.state_dict());self.model.load_state_dict(state,strict=True);assert all(torch.equal(self.model.state_dict()[k],v) for k,v in state.items());self.model=self.model.to('cuda').eval();self.parameter_count=sum(p.numel() for p in self.model.parameters());self.state_tensors=len(state)
 def forecast(self,history,known,hidden_fill=0.):
  mu,sd=norm(history);z=(history-mu)/sd;ctx=torch.tensor(z.reshape(2,8,400).transpose(1,2,0),dtype=torch.float32,device='cuda');query_u=np.r_[z[1,-30:],(known-mu[1,0])/sd[1,0]];query_y=np.r_[z[0,-30:],np.repeat(hidden_fill,H)];u=torch.tensor(np.tile(query_u[None,:,None],(8,1,1)),dtype=torch.float32,device='cuda');y=torch.tensor(np.tile(query_y[None,:,None],(8,1,1)),dtype=torch.float32,device='cuda')
  with torch.inference_mode():mean,std,loss,rmse=self.model(ctx[:,:,0:1],ctx[:,:,1:2],u,y,30)
  assert mean.shape==std.shape==(8,H,1) and torch.isfinite(mean).all() and torch.isfinite(std).all();return mean.mean(0)[:,0].cpu().numpy().astype('float64')*sd[0,0]+mu[0,0],dict(min_predictive_sd=float(std.min().cpu()),max_predictive_sd=float(std.max().cpu()),ensemble='equal eight disjoint400-point contexts',joint_path_law=False)
