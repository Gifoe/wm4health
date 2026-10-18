"""Exact pretrained native calls with explicit ordinary distance-feature controls."""
from common67_v1 import *
import sys,torch,types,importlib,inspect,ast,textwrap,time
sys.path.insert(0,str(PROJECT/'cycle66'/'scripts'))
import native66_v1 as native66
MODES=['bayes_ignored','bayes_indexed','bayes_action_past_view','bayes_action_full_view','bayes_raw_macro']
class Engine:
 def __init__(self):
  self.model,self.mod,self.manifest=native66.load('cuda');self.lb=importlib.import_module('normwear66_native.latent_bayesian');self.original_sampler=self.lb.get_traj_of_state;self.original_encoder=self.model.forward_encoder;self.calls=dict(encoder=0,decoder=0);original_decoder=self.model.forward_decoder
  def enc(this,x,*a,**kw):
   self.calls['encoder']+=1;out=self.original_encoder(x,*a,**kw)
   if hasattr(self,'cache'):
    if x.shape[1:]==(4,L):self.cache['joint_history']=out[0].detach().permute(1,2,0).flatten(1).cpu().numpy()
    if x.shape[1:]==(3,L+H+16):self.cache['action_full']=out[0].detach().permute(1,2,0).flatten(1).cpu().numpy()
   return out
  def dec(this,*a,**kw):self.calls['decoder']+=1;return original_decoder(*a,**kw)
  self.model.forward_encoder=types.MethodType(enc,self.model);self.model.forward_decoder=types.MethodType(dec,self.model)
  tree=ast.parse(textwrap.dedent(inspect.getsource(self.mod.NormWear2.simulate)));count=0
  for node in ast.walk(tree):
   if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].value,ast.Name) and node.targets[0].value.id=='pad_context_tensor' and isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Name) and node.value.value.id=='dec_out':
    for sub in [node.targets[0],node.value]:sub.slice.elts[1]=ast.Name(id='non_visible_channel',ctx=ast.Load());count+=1
  assert count==2;ast.fix_missing_locations(tree);scope=dict(self.mod.__dict__);exec(compile(tree,'<qualified_hidden_channel_clamp67>','exec'),scope);self.simulate=types.MethodType(scope['simulate'],self.model)
  tree=ast.parse(textwrap.dedent(inspect.getsource(self.original_sampler)));count=0
  for node in ast.walk(tree):
   if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
    if node.targets[0].id=='centroids_action_emb':node.value=ast.Call(func=ast.Name(id='_centroids67',ctx=ast.Load()),args=[ast.Name(id='centroids',ctx=ast.Load())],keywords=[]);count+=1
    if node.targets[0].id=='action_emb' and isinstance(node.value,ast.Subscript):node.value=ast.Call(func=ast.Name(id='_query67',ctx=ast.Load()),args=[node.value,ast.Name(id='ss',ctx=ast.Load())],keywords=[]);count+=1
  assert count==2;ast.fix_missing_locations(tree);scope=dict(self.lb.__dict__);scope['_centroids67']=self.centroids;scope['_query67']=self.query;exec(compile(tree,'<ordinary_distance_features67>','exec'),scope);self.distance_sampler=scope['get_traj_of_state'];self.sampler_AST_sha256=hashlib.sha256(ast.unparse(tree).encode()).hexdigest()
 def set_case(self,history,known):
  self.cache={};self.history=torch.from_numpy(np.array(history,copy=True)).to('cuda');self.known=torch.from_numpy(np.array(known,copy=True)).to('cuda');self.x=torch.cat([self.history,torch.cat([torch.zeros((1,H),device='cuda'),self.known])],1)[None]
  self.mean=self.history.mean(-1,keepdim=True)[None];self.sd=self.history.std(-1,keepdim=True)[None];self.sd[self.sd==0]=1.;self.sd+=1e-8
  padded=torch.cat([self.x,torch.cat([torch.zeros((1,16),device='cuda'),self.known[:,-1:].expand(3,16)])[None]],-1);self.normalized=(padded-self.mean)/self.sd;self.normalized[:,0,L:]=0;self.mode=None;self.last_labels=None
 def centroids(self,C):
  if self.mode=='bayes_indexed':return C.reshape(len(C),768,4)[:,:,1:].reshape(len(C),-1)
  E=self.cache['joint_history'];labels=np.argmin(((E[:,None,:]-C[None,:,:])**2).sum(-1),axis=1);assert len(np.unique(labels))==len(C);self.last_labels=labels
  if self.mode=='bayes_action_past_view':F=self.cache['action_past']
  elif self.mode=='bayes_action_full_view':F=self.cache['action_full'][:L//16]
  elif self.mode=='bayes_raw_macro':F=self.normalized[0,1:,:L].reshape(3,L//16,16).mean(-1).T.cpu().numpy()
  else:raise ValueError(self.mode)
  return np.stack([F[labels==i].mean(0) for i in range(len(C))])
 def query(self,native,step):
  if self.mode=='bayes_raw_macro':return self.normalized[0,1:,L:].reshape(3,H//16+1,16).mean(-1).T[step]
  return native
 def forecast(self,mode,mc=2021):
  self.mode=mode;torch.manual_seed(mc);np.random.seed(mc);before=dict(self.calls);torch.cuda.synchronize();start=time.perf_counter()
  with torch.inference_mode():
   if mode=='native_predict':out=self.model.predict(self.history.T[None],H,max_pred_length=16)[0,:,0]
   elif mode=='clamped_simulate':out=self.simulate(self.x.permute(0,2,1).clone(),all_visible_length=L,non_visible_channel=[0],ar_step=16)[0,L:L+H,0]
   else:
    if mode=='bayes_action_past_view' and 'action_past' not in self.cache:
     E=self.original_encoder(self.normalized[:,1:,:L],masking=False)[0];self.calls['encoder']+=1;self.cache['action_past']=E.permute(1,2,0).flatten(1).cpu().numpy()
    self.lb.get_traj_of_state=self.original_sampler if mode=='bayes_ignored' else self.distance_sampler
    try:out=self.model.latent_bayesian_transition(self.normalized,n_channels=4,physio_channels=[0],context_length=L,pred_length=H,num_states=10,action_channels=[1,2,3],condition_bayes=mode!='bayes_ignored',num_traj_sampled=4,latent_encoder=self.model)[0]*self.sd[0,0,0]+self.mean[0,0,0]
    finally:self.lb.get_traj_of_state=self.original_sampler
  torch.cuda.synchronize();assert out.shape==(H,) and torch.isfinite(out).all();return out.cpu().numpy(),dict(seconds=time.perf_counter()-start,encoder_calls=self.calls['encoder']-before['encoder'],decoder_calls=self.calls['decoder']-before['decoder'],MC_seed=mc,mode=mode,cached_feature_views=sorted(self.cache),timing_scope='Serial bundle shares encoded conditioning features for ordinary geometry controls; no equal-compute standalone latency claim')
