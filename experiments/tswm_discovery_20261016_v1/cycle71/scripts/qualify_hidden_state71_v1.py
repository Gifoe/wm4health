"""Existing native hidden-state capability, device-only CPU adapter, TRAIN only."""
import os,sys,pathlib,ast,types,copy,time,random,traceback
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
import numpy as np
import torch
from torch import nn
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
torch.set_num_threads(2)
assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
ACQ=read(OUT/'ACQUISITION71_V1.json');SOURCE=pathlib.Path(ACQ['source_root'])/'neurips2019'
ADAPT=[];BODIES={}

class DeviceOnly(ast.NodeTransformer):
 def __init__(self,name):self.name=name
 def visit_Constant(self,node):
  if isinstance(node.value,str) and node.value in ['cuda:0','cuda','torch.cuda.FloatTensor']:
   value='torch.FloatTensor' if node.value=='torch.cuda.FloatTensor' else 'cpu'
   ADAPT.append(dict(file=self.name,line=node.lineno,before=node.value,after=value))
   return ast.copy_location(ast.Constant(value=value),node)
  return node
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda' and not node.args and not node.keywords:
   ADAPT.append(dict(file=self.name,line=node.lineno,before=ast.unparse(node),after='same tensor on CPU'))
   return ast.copy_location(ast.Call(func=ast.Attribute(value=node.func.value,attr='to',ctx=ast.Load()),args=[],keywords=[ast.keyword(arg='device',value=ast.Constant(value='cpu'))]),node)
  return node
 def visit_Attribute(self,node):
  node=self.generic_visit(node)
  if ast.unparse(node)=='torch.cuda.LongTensor':
   ADAPT.append(dict(file=self.name,line=node.lineno,before='torch.cuda.LongTensor',after='torch.LongTensor'))
   return ast.copy_location(ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='LongTensor',ctx=ast.Load()),node)
  return node

def load(name):
 p=SOURCE/(name+'.py');spec=next(x for x in ACQ['sources'] if x['path']=='neurips2019/'+name+'.py');assert sha(p)==spec['sha256']
 tree=ast.parse(p.read_bytes());BODIES[name]=dict(source_sha256=spec['sha256'],original_class_hashes={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))})
 body=[]
 for node in tree.body:
  if isinstance(node,ast.Import) and any(x.name=='h5py' for x in node.names):continue
  if isinstance(node,ast.ImportFrom) and node.module=='tensorboardX':continue
  body.append(node)
 tree=ast.fix_missing_locations(DeviceOnly(name).visit(ast.Module(body=body,type_ignores=[])))
 mod=types.ModuleType(name);mod.__file__=str(p);sys.modules[name]=mod;exec(compile(tree,str(p),'exec'),mod.__dict__);return mod

for name in ['layers','custom_module','composition','structure','hgnn_composer','data_loading','encoder','modular_metalearning']:load(name)
from hgnn_composer import HGNN_Structure
from modular_metalearning import BounceGrad
from custom_module import torch_NN

def model_hash(L):
 h=hashlib.sha256()
 for k,v in L.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
 return h.hexdigest()

def shell(state):
 args=types.SimpleNamespace(composer='hgnn@'+str(SOURCE/'nri_structure.json'),num_modules='1,2')
 S=HGNN_Structure(args);S.find_node=False
 specs=[v.split('-') for v in S.graph['type_modules'].split(',')];nets=[]
 for count,spec in zip([1,2],specs):
  activation=spec[0];sizes=list(map(int,spec[1:]))
  for _ in range(count):nets.append(torch_NN(inp=sizes[0],out=sizes[-1],hidden=sizes[1:-1],final_act=activation))
 L=nn.ModuleList(nets);L.load_state_dict(state['module_state'],strict=True)
 assert sum(p.numel() for p in L.parameters())==467204
 for p in L.parameters():p.requires_grad_(False)
 L.eval();bg=BounceGrad.__new__(BounceGrad);bg.S=S;bg.L=L;bg.composer='gnn';bg.BG_mode='eval';bg.input_noise=0.;bg.everys=[1,10,20];return bg

def predict(bg,structures,x,H):
 mega=bg.S.compose_multiple_structures(copy.deepcopy(structures));mega['self_regress_steps']=H
 return bg.run_model(mega,x).reshape(H,len(structures),5,4).permute(1,0,2,3)

def numpy_net(module,x):
 for part in [module.features,module.fa]:
  for m in part.children():
   if isinstance(m,nn.Linear):x=x@m.weight.detach().numpy().astype(np.float64).T+m.bias.detach().numpy().astype(np.float64)
   elif isinstance(m,nn.ReLU):x=np.maximum(x,0.)
   else:raise TypeError(type(m))
 return x

def numpy_roll(bg,s,last,H):
 nodes=last.astype(np.float64).copy();out=[]
 for _ in range(H):
  incoming=np.zeros((5,256))
  for j,(a,b) in enumerate(s['graph']['edges']):incoming[b]+=numpy_net(bg.L[1+s['edge_idx_inv'][j]],np.concatenate([nodes[a],nodes[b]]))
  nodes=nodes+numpy_net(bg.L[0],np.concatenate([incoming,nodes],axis=1));out.append(nodes.copy())
 return np.asarray(out)

def qualify_branch(bg,structures,x):
 # Genuine native batched_evaluate body, synthetic targets as a contract fixture.
 initial_a=np.zeros((49,49,1,4),dtype=np.float32);initial_b=initial_a.copy();initial_b[:,0,0]=[.25,-.2,.04,-.03]
 def setup(which,targets=None):
  bg.S.find_node=True;bg.S.StructureParameters=nn.ParameterList([nn.Parameter(torch.from_numpy(v.copy())) for v in np.concatenate([initial_b,initial_a])]);ss=copy.deepcopy(structures)
  for i,s in enumerate(ss):s['parameters']=range(49+i,50+i)
  bg.find_node_dist_rels={i:[] for i in range(5)};bg.S.find_node_dist=0.;bg.find_ct=0.
  ds=[types.SimpleNamespace(TrainInput=torch.from_numpy(x[i].copy()),TrainOutput=torch.from_numpy(targets[i].copy()) if targets is not None else torch.from_numpy(x[i,1:].copy()),Edges=np.zeros(20,dtype=np.int64)) for i in range(49)]
  return ss,ds
 ss,ds=setup('a')
 with torch.no_grad():
  xa=x.copy();xb=x.copy();xa[:,:,0]=0.;xb[:,:,0]=0.;xb[:,0,0]=initial_b[:,0,0]
  # Native output truncation gives 48 predictions even at 50 rollout steps.
  mega=bg.S.compose_multiple_structures(copy.deepcopy(ss));mega['self_regress_steps']=50
  pa=bg.run_model(mega,torch.from_numpy(xa.transpose(1,0,2,3).reshape(49,-1,4).copy())).reshape(48,49,5,4).permute(1,0,2,3).numpy().copy()
  pb=bg.run_model(mega,torch.from_numpy(xb.transpose(1,0,2,3).reshape(49,-1,4).copy())).reshape(48,49,5,4).permute(1,0,2,3).numpy().copy()
 midpoint=(pa+pb)/2.;records=[]
 for label,hidden in [('current_hidden_truth',pa),('candidate_hidden_truth',pb)]:
  targets=midpoint.copy();targets[:,:,0]=hidden[:,:,0];ss,ds=setup(label,targets)
  with torch.no_grad():losses,_,preds=bg.batched_evaluate(ss,ds,'Train')
  choices=[bool(torch.equal(bg.S.StructureParameters[49+i].detach(),torch.from_numpy(initial_b[i]))) for i in range(49)]
  records.append(dict(hidden_target=label,candidate_selected=choices,losses=[float(v) for v in losses]))
 changed=sum(a!=b for a,b in zip(records[0]['candidate_selected'],records[1]['candidate_selected']))
 return dict(status='NATIVE_BRANCH_EXECUTED',tasks=49,hidden_parameter_shape=[49,1,4],candidate_comparison_slices_time_axis=True,changed_choices_under_hidden_target_only_intervention=changed,records=records,observed_targets_identical=True,synthetic_contract_fixture=True,performance_evidence=False)

def observer(bg,structures,history,sentinel):
 bg.S.find_node=False;masked=torch.from_numpy(history.copy());masked[:, :,0]=sentinel;masked.requires_grad_(True)
 z=nn.Parameter(torch.zeros(4,1,4));opt=torch.optim.Adam([z],lr=.01);trace=[];grad_trace=[];all_losses=[]
 for step in range(101):
  initial=torch.cat([z,masked[:,0,1:]],dim=1);padding=torch.zeros((24,20,4));inp=torch.cat([initial.reshape(1,20,4),padding],dim=0)
  pred=predict(bg,structures,inp,24);target=masked[:,1:];row=((pred[:,:,1:]-target[:,:,1:])**2).mean(dim=(1,2,3));loss=row.mean()
  all_losses.append(float(loss.detach()))
  if step in [0,100]:trace.append(dict(step=step,per_row=row.detach().tolist(),loss=float(loss.detach())))
  if step==100:break
  opt.zero_grad();masked.grad=None;loss.backward();assert torch.count_nonzero(masked.grad[:,:,0])==0
  grad_trace.append(z.grad.detach().numpy().copy());assert all(p.grad is None for p in bg.L.parameters());opt.step()
 return dict(losses=trace,all_losses=all_losses,z=z.detach().numpy().copy(),gradients=np.asarray(grad_trace),adam=copy.deepcopy(opt.state_dict()))

def main():
 started=time.perf_counter();random.seed(2026731);np.random.seed(2026731);torch.manual_seed(2026731)
 cp=read(OUT/'formal_seed2026711_bounded01000_v1_segment_result.json')['last'];assert cp['epoch']==1000 and sha(cp['path'])==cp['sha256'];state=torch.load(cp['path'],map_location='cpu',weights_only=False)
 data=pathlib.Path(ACQ['data_root'])/'state_springs5_train.npy';assert sha(data)=='87ffbd8fc604c9087fc1ade73bbd659d403a569872235fa4588994f754634c53';x=np.load(data,mmap_mode='r',allow_pickle=False)[:49].copy();assert x.shape==(49,49,5,4)
 bg=shell(state);before=model_hash(bg.L);ss=copy.deepcopy(state['train_structures'][:49]);refs=[]
 with torch.no_grad():
  inp=torch.from_numpy(x[:2].transpose(1,0,2,3).reshape(49,10,4).copy());p=predict(bg,ss[:2],inp,48).numpy()
  for i in range(2):
   ref=numpy_roll(bg,ss[i],x[i,0],48);err=float(abs(ref-p[i]).max());scaled=err/max(1.,float(abs(ref).max()));assert scaled<2e-4;refs.append(dict(row=i,max_abs=err,scaled=scaled))
 branch=qualify_branch(bg,ss,x);a=observer(bg,ss[:4],x[:4,:25],-123.);b=observer(bg,ss[:4],x[:4,:25],456.)
 assert a['all_losses']==b['all_losses'] and a['losses']==b['losses'] and np.array_equal(a['z'],b['z']) and np.array_equal(a['gradients'],b['gradients']);assert abs(a['gradients']).max()>0
 for k in a['adam']['state']:
  for name,v in a['adam']['state'][k].items():assert torch.equal(v,b['adam']['state'][k][name])
 assert before==model_hash(bg.L) and sha(cp['path'])==cp['sha256'];assert not torch.cuda.is_initialized()
 artifact=storage('artifact')/'HIDDEN_STATE_QUALIFICATION71_V1.npz';assert not artifact.exists();np.savez_compressed(artifact,initial_state=a['z'],gradients=a['gradients'])
 obj=dict(utc=utc(),status='PASS_ORDINARY_HIDDEN_INITIAL_STATE_OBSERVER_QUALIFICATION',checkpoint=cp,protocol_sha256=sha(ROOT/'HIDDEN_STATE_QUALIFICATION71_V1.md'),source_revision=ACQ['revision'],source_bodies=BODIES,device_only_adaptations=ADAPT,numpy_forward_reference=refs,native_branch=branch,observer=dict(train_rows=[0,1,2,3],history_steps=25,updates=100,optimizer='Adam',lr=.01,hidden_target_sentinels=[-123.,456.],complete_optimizer_trajectories_identical=True,hidden_target_gradient_max_abs=0.,initial_state_gradient_max_abs=float(abs(a['gradients']).max()),losses=a['losses']),weights_unchanged_sha256=before,checkpoint_unchanged=True,array_path=str(artifact),array_sha256=sha(artifact),val_rows_accessed=0,eval_rows_accessed=0,new_method=False,paper_reproduction=False,accepted_final_directions=0,wall_seconds=time.perf_counter()-started)
 dump(OUT/'HIDDEN_STATE_QUALIFICATION71_V1.json',obj);print(json.dumps({k:obj[k] for k in ['status','numpy_forward_reference','observer','wall_seconds']}),flush=True)

if __name__=='__main__':main()
