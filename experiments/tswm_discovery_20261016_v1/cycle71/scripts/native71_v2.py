"""Exact source scientific bodies, explicit data/logging/compatibility adapters."""
import sys,pathlib,ast,types,copy,random,time,argparse,hashlib,json,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
import numpy as np
os.environ['MPLCONFIGDIR']=str(storage('artifact')/'mplcache')
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ACQ=read(OUT/'ACQUISITION71_V1.json');SOURCE=pathlib.Path(ACQ['source_root'])/'neurips2019'

class ScalarRecorder:
 def __init__(self,log_dir):
  p=pathlib.Path(log_dir);assert p.drive.upper()=='G:';p.mkdir(exist_ok=False,parents=True);self.handle=(p/'scalars.jsonl').open('x',encoding='utf-8');self.scalars={}
 def add_scalar(self,tag,value,step):
  value=float(value);assert np.isfinite(value),(tag,step,value);self.scalars[tag]=value;self.handle.write(json.dumps(dict(step=int(step),tag=tag,value=value))+'\n')
 def add_figure(self,*a,**k):raise AssertionError('Figure path not part of this protocol')
 def flush(self):self.handle.flush()
 def close(self):self.handle.close()

class SchedulerCompat(torch.optim.lr_scheduler.ReduceLROnPlateau):
 def __init__(self,*a,**kw):kw.pop('verbose',None);super().__init__(*a,**kw)

EXECUTED_BODIES={}
def load_source(name):
 p=SOURCE/(name+'.py');b=p.read_bytes();spec=next(r for r in ACQ['sources'] if r['path']=='neurips2019/'+name+'.py');assert hashlib.sha256(b).hexdigest()==spec['sha256'];tree=ast.parse(b,filename=str(p));body=[];skipped=[]
 for node in tree.body:
  omit=(isinstance(node,ast.Import) and any(a.name=='h5py' for a in node.names)) or (isinstance(node,ast.ImportFrom) and node.module in ['tensorboardX','torch.optim.lr_scheduler'])
  if omit:skipped.append(ast.unparse(node));continue
  body.append(node)
 mod=types.ModuleType(name);mod.__file__=str(p);mod.__dict__.update(SummaryWriter=ScalarRecorder,ReduceLROnPlateau=SchedulerCompat);sys.modules[name]=mod;exec(compile(ast.Module(body=body,type_ignores=[]),str(p),'exec'),mod.__dict__)
 EXECUTED_BODIES[name]=dict(source_sha256=spec['sha256'],skipped_imports=skipped,scientific_bodies={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))})
 return mod

for _name in ['layers','custom_module','composition','structure','hgnn_composer','data_loading','encoder','modular_metalearning']:load_source(_name)
from data_loading import MetaDataset,NPDataset,convert_to_torch_tensors
from hgnn_composer import HGNN_Structure
from modular_metalearning import BounceGrad

class BatchStream:
 # Same shuffled batching distribution; explicit NumPy stream avoids the source
 # CUDA-default-tensor / modern CPU DataLoader generator mismatch.
 def __init__(self,rows,batch,seed):self.rows=rows;self.batch=batch;self.rng=np.random.default_rng(seed)
 def __len__(self):return (len(self.rows)+self.batch-1)//self.batch
 def __iter__(self):
  order=self.rng.permutation(len(self.rows))
  for i in range(0,len(order),self.batch):yield [self.rows[j] for j in order[i:i+self.batch]]

def parse_native(seed,epochs,ntrain,tag):
 p=SOURCE/'modular_main.py';tree=ast.parse(p.read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');nodes=[]
 for n in fn.body:
  nodes.append(n)
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in n.targets):break
 else:raise AssertionError('Native parser not found')
 namespace=dict(argparse=argparse);old=sys.argv
 sys.argv=[str(p),'--num_modules','1,2','--composer','hgnn@'+str(SOURCE/'nri_structure.json'),'--dont_normalize','--plot_name',tag,'--data_split','49,21,30','--plot_freq','50','--meta_lr','0.01','--initial_acc','-1','--initial_temp','-9.8','--meta_batch_size','128','--data_desc','NPY_SR5@springs5','--temp_slope_opt_steps','800','--optimization_steps',str(epochs),'--max_datasets',str(ntrain),'--encoder','state','--proposal_type','node','--log_dir_comment','','--log_dir',str(storage('artifact')/tag/'logs'),'--models_path',str(storage('checkpoint')/tag),'--encoder_type','mlp','--torch_seed',str(seed),'--cpu_threads','4']
 try:exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace)
 finally:sys.argv=old
 args=namespace['args'];args.cpu_threads=int(args.cpu_threads);return args

class ExplicitMeta(MetaDataset):
 def create_datasets(self,filename):
  assert filename=='springs5';data=pathlib.Path(ACQ['data_root']);tr=np.load(data/'state_springs5_train.npy',mmap_mode='r',allow_pickle=False);te=np.load(data/'state_springs5_test.npy',mmap_mode='r',allow_pickle=False);er=np.load(data/'edges_springs5_train.npy',allow_pickle=False);ee=np.load(data/'edges_springs5_test.npy',allow_pickle=False);nv=128 if self.max_datasets==128 else 256
  assert tr.shape==(2048,49,5,4) and te.shape==(512,99,5,4) and np.isin(er[:self.max_datasets],[0,1]).all() and np.isin(ee[:nv],[0,1]).all()
  def row(x,e,i,split,all_train):return NPDataset(input_data=x,output_data=x,train=self.train,val=self.val,test=self.test,train_val_are_all=all_train,name=split+'_'+str(i),edge_data=e,structure_idx=i)
  self.MTRAIN=[row(np.array(tr[i],copy=True),er[i].copy(),i,'train',True) for i in range(self.max_datasets)];self.MVAL=[row(np.array(te[i],copy=True),ee[i].copy(),i,'val',False) for i in range(nv)];self.MTEST=[];self.ALL=self.MTRAIN+self.MVAL;self.mtrain=len(self.MTRAIN);self.mval=len(self.MVAL);self.mtest=0
  assert all(x.TrainInput.shape==(49,5,4) and x.TrainOutput.shape==(48,5,4) for x in self.MTRAIN)
  assert all(x.TrainInput.shape==(49,5,4) and x.ValInput.shape==(21,5,4) and x.ValOutput.shape==(20,5,4) for x in self.MVAL)

class InstrumentedNative(BounceGrad):
 def __init__(self,S,args,tag):
  self.tag71=tag;self.seed71=int(args.torch_seed);self.module_updates71=0;self.proposal_updates71=0;self.best71=float('inf');self.selected71=None;self.receipts71=[];self.start71=time.perf_counter();super().__init__(S,args);self.SOpt.register_step_post_hook(self._module_hook);self.encoder_opt.register_step_post_hook(self._proposal_hook);self.initial_parameters71={k:v.detach().clone() for k,v in self.L.state_dict().items()}
 def _module_hook(self,*a,**kw):self.module_updates71+=1
 def _proposal_hook(self,*a,**kw):self.proposal_updates71+=1
 def get_filename(self,comment=''):return 'seed'+str(self.seed71)
 def initialize_dataset(self,data_desc):
  self.data_source=data_desc;self.D=ExplicitMeta(train=self.data_split[0],val=self.data_split[1],test=self.data_split[2],filename='springs5',max_datasets=self.max_datasets,normalize=False);self.T=convert_to_torch_tensors(self.D,self.nn_device);self.DL_Mtrain=BatchStream(self.T.MTRAIN,self.meta_batch_size,self.seed71+11);self.DL_Mval=BatchStream(self.T.MVAL,self.meta_batch_size,self.seed71+12);self.DL_Mtest=None;self.initialize_answers()
 def update_metrics(self):
  super().update_metrics();self.writer.flush();m=dict(utc=utc(),tag=self.tag71,epoch=int(self.step),module_updates=self.module_updates71,proposal_updates=self.proposal_updates71,elapsed_seconds=time.perf_counter()-self.start71,train_loss=self.writer.scalars['mean_loss/T_train'],val_loss=self.writer.scalars['mean_loss/V_val'],native_encoder_accuracy=self.writer.scalars.get('accuracy_encoder_truth/Mval'),native_structure_accuracy=self.writer.scalars.get('accuracy_SA_truth/Mval'),gpu_peak_bytes=int(torch.cuda.max_memory_allocated()),reserved_eval_rows_accessed=0)
  dump(OUT/(self.tag71+'_progress.json'),m,exclusive=False)
  if self.step<3 or self.step%25==0:print(json.dumps(m),flush=True)
 def save_train_state(self):
  p=storage('checkpoint')/self.tag71/('epoch'+str(self.step).zfill(5)+'.pt');p.parent.mkdir(exist_ok=True,parents=True)
  state=dict(epoch=int(self.step),seed=self.seed71,module_state=self.L.state_dict(),proposal_state=self.encoder.state_dict(),module_optimizer=self.SOpt.state_dict(),proposal_optimizer=self.encoder_opt.state_dict(),scheduler=self.SOpt_scheduler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),train_shuffle=self.DL_Mtrain.rng.bit_generator.state,val_shuffle=self.DL_Mval.rng.bit_generator.state,train_structures=self.S.TrainStructures,val_structures=self.S.ValStructures,SA_running_acc_rate=self.SA_running_acc_rate,SA_running_factor=self.SA_running_factor,metrics=dict(self.writer.scalars),module_updates=self.module_updates71,proposal_updates=self.proposal_updates71)
  with p.open('xb') as f:torch.save(state,f)
  rec=dict(epoch=int(self.step),path=str(p),sha256=sha(p),val_loss=self.writer.scalars['mean_loss/V_val'],module_updates=self.module_updates71,proposal_updates=self.proposal_updates71)
  if rec['val_loss']<self.best71:self.best71=rec['val_loss'];self.selected71=rec
  self.receipts71.append(rec);dump(OUT/(self.tag71+'_checkpoints.json'),dict(checkpoints=self.receipts71,selected=self.selected71),exclusive=False)

def fit(seed,epochs,ntrain,tag):
 assert torch.cuda.is_available();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats();args=parse_native(seed,epochs,ntrain,tag);S=HGNN_Structure(args=args);bg=InstrumentedNative(S,args,tag);bg.SAConfig_SGDModules(epochs);bg.writer.close();assert bg.module_updates71==epochs*((ntrain+127)//128);change=max(float(abs(v-bg.initial_parameters71[k]).max().item()) for k,v in bg.L.state_dict().items());assert change>0 and np.isfinite(change)
 obj=dict(utc=utc(),status='NATIVE_SUBSET_FIT_COMPLETE',seed=seed,tag=tag,epochs=epochs,train_tasks=ntrain,val_tasks=len(bg.T.MVAL),reserved_eval_rows_accessed=0,module_updates=bg.module_updates71,proposal_updates=bg.proposal_updates71,module_parameters=sum(p.numel() for p in bg.L.parameters()),proposal_parameters=sum(p.numel() for p in bg.encoder.parameters()),max_parameter_change=change,selected=bg.selected71,last=bg.receipts71[-1],checkpoints=bg.receipts71,wall_seconds=time.perf_counter()-bg.start71,gpu_peak_bytes=int(torch.cuda.max_memory_allocated()),source_bodies=EXECUTED_BODIES,source_revision=ACQ['revision'],acquisition_sha256=sha(OUT/'ACQUISITION71_V1.json'),protocol_sha256=sha(ROOT/'PROTOCOL71_V1.md'),is_original_full_paper_reproduction=False,new_method=False,accepted_final_directions=0)
 dump(OUT/(tag+'_result.json'),obj);print(json.dumps({k:obj[k] for k in ['status','seed','epochs','module_updates','proposal_updates','module_parameters','proposal_parameters','wall_seconds','gpu_peak_bytes','selected']}),flush=True);return obj
