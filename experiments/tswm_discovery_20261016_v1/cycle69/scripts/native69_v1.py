import sys,pathlib,types,ast,copy,importlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common69_v1 import *
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
sys.path.insert(0,str(PROJECT/'cycle68'/'scripts'))
import native68_v1 as n68
class Meta:
 def __init__(self):
  original=n68.MetaID();self.uncached=original.model;self.args=original.args;mod=sys.modules['sysid68_native'];self.model=mod.TSTransformerBenchmark(mod.Config(**self.args)).to('cuda').eval();sd=self.uncached.state_dict();assert set(sd)==set(self.model.state_dict());self.model.load_state_dict(sd,strict=True);assert all(torch.equal(v,self.model.state_dict()[k]) for k,v in sd.items());self.parameters=sum(p.numel() for p in self.model.parameters());self.state_tensors=len(sd)
 def set_context(self,data,case):
  self.train_u=torch.tensor(data['train_u'],dtype=torch.float32,device='cuda');self.train_y=torch.tensor(data['train_y'],dtype=torch.float32,device='cuda');self.um=self.train_u.mean();self.us=self.train_u.std()+1e-6;self.ym=self.train_y.mean();self.ys=self.train_y.std()+1e-6;self.tu=(self.train_u-self.um)/self.us;self.ty=(self.train_y-self.ym)/self.ys;self.starts=list(range(0,case['context_length']-400+1,case['context_stride']));self.cu=torch.stack([self.tu[i:i+400] for i in self.starts])[:,:,None];self.cy=torch.stack([self.ty[i:i+400] for i in self.starts])[:,:,None];self.model.src=None;self.model.mem=None
  with torch.inference_mode():self.model.src=self.model.embed_ctx(self.cy,self.cu);self.model.mem=self.model.encoder(self.model.src)
  self.contexts=len(self.starts)
 def init_state(self,data,n,benchmark):
  tu=(torch.tensor(data['test_u'],dtype=torch.float32,device='cuda')-self.um)/self.us;ty=(torch.tensor(data['test_y'][:n],dtype=torch.float32,device='cuda')-self.ym)/self.ys
  if n>=30:iu=tu[n-30:n];iy=ty[n-30:n]
  elif benchmark=='CED':iu=torch.cat([self.tu[-(30-n):],tu[:n]]);iy=torch.cat([self.ty[-(30-n):],ty])
  else:iu=torch.cat([torch.zeros(30-n,device='cuda'),tu[:n]]);iy=torch.cat([torch.zeros(30-n,device='cuda'),ty])
  return tu,iu[None,:,None].expand(self.contexts,-1,-1),iy[None,:,None].expand(self.contexts,-1,-1)
 def query(self,initial_u,initial_y,future_u,hidden_fill=0.,cached=True):
  u=torch.cat([initial_u,future_u[None,:,None].expand(self.contexts,-1,-1)],1);y=torch.cat([initial_y,torch.full((self.contexts,100,1),hidden_fill,device='cuda')],1)
  with torch.inference_mode():m,s,_,_=self.model(self.cy,self.cu,u,y,30) if cached else self.uncached(self.cy,self.cu,u,y,30)
  assert m.shape==s.shape==(self.contexts,100,1) and torch.isfinite(m).all() and torch.isfinite(s).all() and (s>0).all();return m,s
 def all_queries(self,data,case,n,progress=None):
  tu,iu,iy=self.init_state(data,n,case['benchmark']);total=len(tu)-n;means=np.empty((self.contexts,total),dtype='float32');stds=np.empty_like(means);queries=[]
  for offset in range(0,total,100):
   t=n+offset;fu=tu[t:t+100];actual=len(fu)
   if actual<100:fu=torch.cat([fu,fu[-1:].expand(100-actual)])
   m,s=self.query(iu,iy,fu);means[:,offset:offset+actual]=m[:,:actual,0].cpu().numpy();stds[:,offset:offset+actual]=s[:,:actual,0].cpu().numpy();iy=m[:,-30:];iu=fu[-30:][None,:,None].expand(self.contexts,-1,-1);queries.append(dict(offset=offset,test_start=t,actual=actual,known_input_pad=100-actual,initial_y_source='allowed measured initial prefix' if offset==0 else 'same context expert previous predicted outputs only'))
   if progress and (len(queries)%10==0 or offset+actual==total):progress(len(queries),offset+actual,total)
  return means,stds,queries
def ensemble(means,stds,mode):
 m=np.asarray(means,dtype='float64');s=np.asarray(stds,dtype='float64');w=np.ones_like(m) if mode=='mean' else 1/s**2;w/=w.sum(0);mu=(w*m).sum(0);est_sd=np.sqrt((w*w*s*s).sum(0));mix_sd=np.sqrt((w*(s*s+(m-mu)**2)).sum(0));return mu,est_sd,mix_sd
class Dyno:
 def __init__(self):
  receipt=read(OUT/'DATA_SOURCE69_V2.json') if (OUT/'DATA_SOURCE69_V2.json').exists() else read(OUT/'ACQUISITION69_V1.json');self.sources={r['path']:pathlib.Path(r['server_path']) for r in receipt['sources']}
  for r in receipt['sources']:assert sha(r['server_path'])==r['sha256']
  lib=self.sources['dynonet_library/src/dynonet/lti.py'].parent;pkg=types.ModuleType('dynonet');pkg.__path__=[str(lib)];sys.modules['dynonet']=pkg;lti=importlib.import_module('dynonet.lti');static=importlib.import_module('dynonet.static');text=self.sources['dynonet_baseline/dynonet_baseline.py'].read_text(encoding='utf-8-sig');tree=ast.parse(text);names=['my_dynoNet','Scaler','apply_model','data_normalizer'];nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names];assert {x.name for x in nodes}==set(names);tr=ast.Module(body=nodes,type_ignores=[]);self.AST_sha256=hashlib.sha256(ast.unparse(tr).encode()).hexdigest();self.scope=dict(torch=torch,np=np,copy=copy,nn=torch.nn,MimoLinearDynamicalOperator=lti.MimoLinearDynamicalOperator,MimoStaticNonLinearity=static.MimoStaticNonLinearity,config={});exec(compile(tr,'<exact_author_dyno_model_and_apply69>','exec'),self.scope);self.models={};self.metadata={}
  for benchmark,name in [('WH','wienerhammerstein'),('Tanks','cascaded_tanks'),('CED','ced')]:
   p=self.sources['dynonet_baseline/checkpoints/'+name+'_checkpoint.pth'];ck=torch.load(p,map_location='cpu',weights_only=True);model=self.scope['my_dynoNet'](input_size=1,hidden_size=20,output_size=1,n_a=10,n_b=10).eval();sd=ck['model_state_dict'];assert set(sd)==set(model.state_dict());model.load_state_dict(sd,strict=True);assert all(torch.equal(v,model.state_dict()[k]) for k,v in sd.items());self.models[benchmark]=model;self.metadata[benchmark]=dict(parameters=sum(p.numel() for p in model.parameters()),state_tensors=len(sd),epoch=ck.get('epoch'),source_checkpoint_sha256=sha(p),training_and_selection_provenance='Released author target-trained artifact; not independently reconstructed')
 def forecast(self,data,case,all_data):
  rows=[r for r in all_data if r['case']['benchmark']==case['benchmark']];train=[types.SimpleNamespace(u=r['arrays']['train_u'][:,None],y=r['arrays']['train_y'][:,None]) for r in rows];_,us,ys=self.scope['data_normalizer'](train);self.scope['config']['simulate_train']=case['benchmark']=='CED';d=dict(model=self.models[case['benchmark']],u_scaler=us,y_scaler=ys)
  with torch.inference_mode():pred=self.scope['apply_model'](d,data['test_u'][:,None],data['test_y'][:case['official_initial'],None],u_train=data['train_u'][:,None])
  return pred[:,0].astype('float64'),dict(u_mean=float(us.mean),u_sd=float(us.std),y_mean=float(ys.mean),y_sd=float(ys.std),normalizer_train_cases=[r['case']['case'] for r in rows])
