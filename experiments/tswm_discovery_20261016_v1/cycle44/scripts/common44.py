"""Trained native SPF adaptations and strong ordinary observed-history models."""
import sys,pathlib,os,json,hashlib,math,copy,types,ast,random,time
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle43/scripts'));import harness43_v2 as h43
np=h43.np;torch=h43.torch;F=h43.F
from torch import nn
from torch.distributions import MultivariateNormal
H=8;FUT=16;P=64;LEN=H+FUT;SEEDS=[2026441,2026442]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,obj):pathlib.Path(p).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def numpy_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def checksum(model):
 digest=hashlib.sha256()
 for k,v in model.state_dict().items():digest.update(k.encode());digest.update(v.detach().cpu().numpy().tobytes())
 return digest.hexdigest()

def pca_fit(train):
 flat=train.reshape(-1,P).astype(np.float64);cov=flat.T@flat/len(flat);values,vectors=np.linalg.eigh(cov);order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
 noise=float(np.mean(values[-32:]));d=int(np.clip(np.sum(values>1.5*noise),1,8));U=vectors[:,:d];scale=np.sqrt(values[:d])
 for j in range(d):
  if U[np.argmax(np.abs(U[:,j])),j]<0:U[:,j]*=-1
 return {'U':U.astype('float32'),'scale':scale.astype('float32'),'noise':np.asarray(noise,dtype='float32'),'eigenvalues':values,'rank':np.asarray(d)}

def make_spf(condition_trainable=False,device='cuda'):
 cls,hashes=h43.methods(repair=True);ctx=cls();ctx.device=torch.device(device)
 ctx.model=h43.Backbone(height=8,width=8,cond_height=8,cond_width=8,in_channels=1,cond_in_channels=H,out_channels=1,num_attention_heads=4,attention_head_dim=32,depth=2,patch_size=2,cond_patch_size=2,max_num_frames=32,cross_attention_dim=128,pos_embed_type='sincos',temp_pos_embed_type='sincos',temp_cond_pos_embed_type='sincos',use_grad_checkpointing=False,add_temp_pos_embed=True,use_flash_attn=False).to(device)
 ctx.noise_scheduler=h43.Scheduler(ns=[[4,4],[4,4],[4,64]],num_stages=3,stage_range=[0,.33,.66,1],num_train_timesteps=1000,shift=1.)
 ctx.noise_scheduler_config=h43.attr(_target_='emulator.schedulers.temporal_flow.TemporalFlowMatchEulerDiscreteScheduler',num_stages=3,ns=[[4,4],[4,4],[4,64]])
 ctx.datamodule_config=h43.attr(output_variables=['x'],timescales=['decadal','yearly','monthly'],output_seq_len={'decadal':1,'yearly':1,'monthly':FUT})
 ctx.stages=['decadal','yearly','monthly'];ctx.climate_models=['synthetic'];ctx.num_inference_steps=8;ctx.multi_scale=True;ctx.config={'model':{'static_condition':False,'use_pathway':False,'no_time_embedding':False}}
 grid=h43.attr(lon=np.arange(8),lat=np.arange(8));ctx.trainer=h43.attr(validating=False,datamodule=h43.attr(test_ds=h43.attr(climate_model_datasets=[h43.attr(grid=grid)])))
 ctx.n_params=sum(p.numel() for p in ctx.model.parameters());ctx.parameters=ctx.model.parameters;ctx.log_dict=lambda *a,**kw:None
 if condition_trainable:enable_condition_grad(ctx.model)
 return ctx

def enable_condition_grad(model):
 module=sys.modules[model.__class__.__module__];source=pathlib.Path(module.__file__);tree=ast.parse(source.read_text(encoding='utf-8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpatiotemporalPyramidFlow');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 class Change(ast.NodeTransformer):
  count=0
  def visit_With(self,node):
   if len(node.items)==1 and ast.unparse(node.items[0].context_expr)=='torch.no_grad()':self.count+=1;return node.body
   return self.generic_visit(node)
 change=Change();fn=change.visit(fn);assert change.count==1
 namespace=dict(module.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),namespace)
 model.forward=types.MethodType(namespace['forward'],model)

def native_inputs(history,device):
 history=torch.as_tensor(history,device=device,dtype=torch.float32);n=len(history);channels=history.reshape(n,H,8,8)
 xs={name:channels[:,None].expand(n,t,H,8,8).contiguous() for name,t in [('decadal',1),('yearly',1),('monthly',FUT)]}
 metadata={name+'_Y_time':[torch.zeros(n,device=device,dtype=torch.int64),torch.zeros(n,device=device,dtype=torch.int64)] for name in xs}
 return xs,metadata

def native_examples(batch):
 history=batch[:,:H].reshape(-1,H,8,8);future=batch[:,H:].reshape(-1,FUT,1,8,8);coarse=future.mean(1,keepdim=True);examples=[]
 for j in range(len(batch)):
  item={'metadata':{'climate_model':'synthetic'}}
  for name,t,y in [('decadal',1,coarse[j]),('yearly',1,coarse[j]),('monthly',FUT,future[j])]:
   item[name]={'X':history[j][None].expand(t,H,8,8),'Y':y,'metadata':{'input_stats':[0.,1.],'stats':[0.,1.],'X_time':[0,0],'Y_time':[0,0]}}
  examples.append(item)
 return examples

def native_loss(ctx,batch):
 packed=h43.collate_fn_spf(native_examples(batch),h43.collate_config(),ctx.noise_scheduler,ctx.datamodule_config.output_seq_len,is_training=True)
 packed=h43.to_device(packed,ctx.device);return ctx.training_step(packed,0)['loss']

@torch.no_grad()
def sample_spf(ctx,history,n,draw_seed,timescale='monthly',antithetic=False):
 ctx.model.eval();hist=torch.as_tensor(history,dtype=torch.float32,device=ctx.device).repeat_interleave(n,0);xs,meta=native_inputs(hist,ctx.device);g=torch.Generator(device=ctx.device).manual_seed(draw_seed)
 root=h43.randn_tensor((len(hist),1,1,2,2),device=ctx.device,dtype=torch.float32,generator=g)
 original=ctx.sample_block_noise_seeded
 if antithetic:root=-root;ctx.sample_block_noise_seeded=lambda *a,**kw:-original(*a,**kw)
 try:value,_=ctx.predict_modular(xs,metadata=meta,timescale=timescale,init_latents=root,rng=g,latent_size=[1,2,4])
 finally:ctx.sample_block_noise_seeded=original
 return value.reshape(len(history),n,FUT if timescale=='monthly' else 1,P)

class PCAModel(nn.Module):
 def __init__(self,pca):
  super().__init__();self.d=int(pca['rank']);self.register_buffer('U',torch.tensor(pca['U']));self.register_buffer('scale',torch.tensor(pca['scale']));self.register_buffer('tail_noise',torch.tensor(float(pca['noise'])))
 def project(self,y):return y@self.U/self.scale
 def observation(self,pc,tail):return (pc*self.scale)@self.U.T+(tail-(tail@self.U)@self.U.T)*self.tail_noise.sqrt()

class AR(PCAModel):
 def __init__(self,pca):
  super().__init__(pca);d=self.d;self.K=4;self.tri=tuple(torch.tril_indices(d,d));self.gru=nn.GRU(d,256,2,batch_first=True);self.head=nn.Linear(256,self.K*(1+d+d*(d+1)//2))
 def params(self,h):
  shape=h.shape[:-1];r=self.head(h).reshape(*shape,self.K,-1);logits=r[...,0];mean=r[...,1:1+self.d];raw=r[...,1+self.d:];L=torch.zeros(*shape,self.K,self.d,self.d,device=h.device,dtype=h.dtype);L[...,self.tri[0],self.tri[1]]=raw
  diagonal=torch.arange(self.d,device=h.device);L[...,diagonal,diagonal]=F.softplus(L[...,diagonal,diagonal])+.02
  return logits,mean,L
 def log_prob(self,target,logits,mean,L):
  delta=(target.unsqueeze(-2)-mean).unsqueeze(-1);solved=torch.linalg.solve_triangular(L,delta,upper=False).squeeze(-1);component=-.5*(self.d*math.log(2*math.pi)+solved.square().sum(-1))-L.diagonal(dim1=-2,dim2=-1).log().sum(-1)
  return torch.logsumexp(logits.log_softmax(-1)+component,-1)
 def loss(self,y):
  pc=self.project(y);h,_=self.gru(pc[:,:-1]);return -self.log_prob(pc[:,1:],*self.params(h)).mean()/self.d
 @torch.no_grad()
 def sample(self,history,n,draw_seed,antithetic=False):
  self.eval();hist=torch.as_tensor(history,device=self.U.device,dtype=self.U.dtype);pc=self.project(hist);hidden_features,hidden=self.gru(pc);features=hidden_features[:,-1].repeat_interleave(n,0);hidden=hidden.repeat_interleave(n,1);b=len(features);g=torch.Generator(device=self.U.device).manual_seed(draw_seed);out=[]
  for t in range(FUT):
   logits,means,L=self.params(features);u=torch.rand(b,1,device=self.U.device,generator=g);eps=torch.randn(b,self.d,device=self.U.device,generator=g);tail=torch.randn(b,P,device=self.U.device,generator=g)
   if antithetic:u=1-u;eps=-eps;tail=-tail
   component=(u>logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(self.K-1);index=torch.arange(b,device=self.U.device);next_pc=means[index,component]+torch.bmm(L[index,component],eps[:,:,None]).squeeze(-1);out.append(self.observation(next_pc,tail));features0,hidden=self.gru(next_pc[:,None],hidden);features=features0[:,0]
  return torch.stack(out,1).reshape(len(history),n,FUT,P)

def inverse_softplus(x):return x+np.log(-np.expm1(-x))
def chol_parameter(matrix):
 L=np.linalg.cholesky(matrix);diag=np.diag_indices_from(L);L[diag]=inverse_softplus(np.maximum(L[diag]-1e-4,1e-5));return torch.tensor(L,dtype=torch.float32)
def positive_chol(raw):
 L=torch.tril(raw,-1);return L+torch.diag_embed(F.softplus(torch.diagonal(raw,dim1=-2,dim2=-1))+1e-4)
def floor_cov(cov,floor=1e-4):
 vals,vec=np.linalg.eigh((cov+cov.T)/2);return (vec*np.maximum(vals,floor))@vec.T

class LGSSM(PCAModel):
 def __init__(self,pca,train):
  super().__init__(pca);d=self.d;pc=np.asarray(train,dtype=np.float64)@np.asarray(pca['U'],dtype=np.float64)/np.asarray(pca['scale']);r=float(pca['noise']);D=np.diag(r/np.asarray(pca['scale'],dtype=np.float64)**2);left=pc[:,:-1].reshape(-1,d);right=pc[:,1:].reshape(-1,d);cov=left.T@left/len(left);A=(right.T@left/len(left))@np.linalg.inv(floor_cov(cov-D));norm=np.linalg.norm(A,2)
  if norm>1.02:A*=1.02/norm
  residual=right-left@A.T;Q=floor_cov(residual.T@residual/len(left)-D-A@D@A.T);P0=floor_cov(pc[:,0].T@pc[:,0]/len(pc)-D)
  self.A=nn.Parameter(torch.tensor(A,dtype=torch.float32));self.rawQ=nn.Parameter(chol_parameter(Q));self.rawP=nn.Parameter(chol_parameter(P0));self.logR=nn.Parameter(torch.tensor(math.log(r),dtype=torch.float32))
 def matrices(self):
  lq=positive_chol(self.rawQ);lp=positive_chol(self.rawP);return self.A,lq@lq.T,lp@lp.T,self.logR.exp()
 def filter(self,y,with_parts=False):
  pc=self.project(y);A,Q,P0,r=self.matrices();m=torch.zeros(len(y),self.d,device=y.device,dtype=y.dtype);P0=P0.to(y.dtype);A=A.to(y.dtype);Q=Q.to(y.dtype);r=r.to(y.dtype);cov=P0;parts=[];D=torch.diag(r/self.scale.to(y.dtype).square());eye=torch.eye(self.d,device=y.device,dtype=y.dtype)
  for t in range(y.shape[1]):
   if t:m=m@A.T;cov=A@cov@A.T+Q
   S0=(cov+cov.T)/2+D;L=torch.linalg.cholesky(S0);innovation=pc[:,t]-m;whiten=torch.linalg.solve_triangular(L,innovation.T,upper=False).T
   if with_parts:
    tail=y[:,t]-(pc[:,t]*self.scale)@self.U.T;nl=.5*(self.d*math.log(2*math.pi)+2*L.diagonal().log().sum()+whiten.square().sum(-1))+self.scale.log().sum();nl=nl+.5*((P-self.d)*torch.log(2*math.pi*r)+tail.square().sum(-1)/r);parts.append(nl)
   gain=torch.cholesky_solve(cov,L).T;m=m+innovation@gain.T;remaining=eye-gain;cov=remaining@cov@remaining.T+gain@D@gain.T;cov=(cov+cov.T)/2
  return m,cov,(torch.stack(parts,1) if with_parts else None)
 def loss(self,y):return self.filter(y,True)[2].mean()/P
 def stabilize(self):
  with torch.no_grad():
   norm=torch.linalg.matrix_norm(self.A,2)
   if norm>1.05:self.A.mul_(1.05/norm)
 @torch.no_grad()
 def sample(self,history,n,draw_seed,antithetic=False):
  self.eval();hist=torch.as_tensor(history,device=self.U.device,dtype=self.U.dtype);m,cov,_=self.filter(hist);A,Q,_,r=self.matrices();lp=torch.linalg.cholesky(cov);lq=torch.linalg.cholesky(Q);m=m.repeat_interleave(n,0);b=len(m);g=torch.Generator(device=m.device).manual_seed(draw_seed);sign=-1 if antithetic else 1;state=m+sign*torch.randn(b,self.d,device=m.device,generator=g)@lp.T;out=[]
  for t in range(FUT):
   e=sign*torch.randn(b,self.d,device=m.device,generator=g);noise=sign*torch.randn(b,P,device=m.device,generator=g);state=state@A.T+e@lq.T;out.append((state*self.scale)@self.U.T+r.sqrt()*noise)
  return torch.stack(out,1).reshape(len(history),n,FUT,P)

@torch.no_grad()
def sample_model(family,model,history,n,draw_seed,antithetic=False):
 return sample_spf(model,history,n,draw_seed,antithetic=antithetic) if family.startswith('spf') else model.sample(history,n,draw_seed,antithetic)

def energy(samples,truth):
 truth=torch.as_tensor(truth,device=samples.device,dtype=samples.dtype);s=samples.shape[1];flat=samples.flatten(2);target=truth.flatten(1);first=torch.linalg.vector_norm(flat-target[:,None],dim=-1).mean(1);distances=torch.cdist(flat,flat);distances.diagonal(dim1=-2,dim2=-1).zero_();pair=distances.sum((1,2))/(2*s*(s-1));return (first-pair)/math.sqrt(FUT*P)

@torch.no_grad()
def validation(family,model,val,draw_seed=20264499):
 values=[]
 for j in range(0,16,2):
  samples=sample_model(family,model,val[j:j+2,:H],32,draw_seed+j);values.extend(energy(samples,val[j:j+2,H:]).cpu().tolist())
 return float(np.mean(values))
