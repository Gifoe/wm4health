"""Source-based graph inference and true free-rollout interface; no future targets."""
import sys,pathlib,types,copy,random,hashlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native71_v3 import *

def weights_hash(bg):
 h=hashlib.sha256()
 for prefix,model in [('modules',bg.L),('proposal',bg.encoder)]:
  for k,v in model.state_dict().items():h.update((prefix+k).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def context_inference(bg,history,steps=100):
 # The source loss/search sees history only; no real future or graph labels.
 bg.S.find_node=False;bg.S.num_types=int(bg.S.graph.get('num_types',2));structures=[];datasets=[]
 for i,x in enumerate(history):
  s=bg.S.initialize_structure(find_node=False,dataset=None);s['original_input_shape']=(49,5,4);s['original_output_shape']=(48,5,4);structures.append(s);v=torch.as_tensor(x,dtype=torch.float32,device='cuda');datasets.append(types.SimpleNamespace(TrainInput=v,TrainOutput=v[1:],ValInput=v,ValOutput=v[1:],Edges=np.zeros(20,dtype=np.int64),structure_idx=i))
 bg.BG_mode='eval';bg.SA_running_acc_rate=bg.SA_running_factor=1e-9;temp=np.exp(-9.8);accepted=[]
 with torch.no_grad():
  for step in range(steps):
   target=np.exp(-1.-5.*step/800.)
   temp=temp*1.1 if bg.SA_running_acc_rate/bg.SA_running_factor<target else max(temp/1.1,np.exp(-100.))
   result=bg.batched_bounce(structures,datasets,temp,False);structures=list(result[0]);accepted.append(len(result[12]))
 return structures,dict(steps=steps,accepted_per_step=accepted,last_temperature=float(temp),dummy_labels_only=True)

def rollout(bg,structures,last,H,padding):
 mega=bg.S.compose_multiple_structures(copy.deepcopy(structures));mega['self_regress_steps']=H;x=np.full((H+1,len(last)*5,4),padding,dtype=np.float32);x[0]=last.reshape(-1,4)
 with torch.no_grad():pred=bg.run_model(mega,torch.as_tensor(x,dtype=torch.float32,device='cuda')).reshape(H,len(last),5,4).permute(1,0,2,3)
 return pred.cpu().numpy().copy()

def numpy_net(module,x):
 # Independent linear/ReLU algebra; do not invoke native module.forward.
 from torch import nn
 def layer(m,y):
  if isinstance(m,nn.Linear):return y@m.weight.detach().cpu().numpy().astype(np.float64).T+m.bias.detach().cpu().numpy().astype(np.float64)
  if isinstance(m,nn.ReLU):return np.maximum(y,0.)
  if isinstance(m,nn.Sequential):
   for child in m.children():y=layer(child,y)
   return y
  raise TypeError(type(m))
 return layer(module.fa,layer(module.features,x))

def numpy_rollout(bg,structure,last,H):
 nodes=np.array(last,dtype=np.float64,copy=True);result=[];edges=structure['graph']['edges'];kinds=structure['edge_idx_inv'];end=structure['end_nodes'];assert end==1
 for _ in range(H):
  incoming=np.zeros((5,structure['msg_sz']),dtype=np.float64)
  for kind in range(2):
   for j,(src,sink) in enumerate(edges):
    if kinds[j]==kind:incoming[sink]+=numpy_net(bg.L[end+kind],np.concatenate([nodes[src],nodes[sink]]))
  nodes=nodes+numpy_net(bg.L[0],np.concatenate([incoming,nodes],axis=1));result.append(nodes.copy())
 return np.stack(result)

def qualify():
 start=time.perf_counter();manifest=read(OUT/'formal_seed2026711_v3_checkpoints.json');ck=next(x for x in manifest['checkpoints'] if x['epoch']==50);p=pathlib.Path(ck['path']);assert sha(p)==ck['sha256'];state=torch.load(p,map_location='cpu',weights_only=False)
 for row in ACQ['sources']:assert sha(row['server_path'])==row['sha256']
 data=pathlib.Path(ACQ['data_root']);x=np.load(data/'state_springs5_train.npy',mmap_mode='r',allow_pickle=False)[:4].copy();assert x.shape==(4,49,5,4);seed=2026721;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);args=parse_native(seed,1,128,'open_loop_qualification71_v1');bg=InstrumentedNative(HGNN_Structure(args),args,'open_loop_qualification71_v1');bg.L.load_state_dict(state['module_state'],strict=True);bg.encoder.load_state_dict(state['proposal_state'],strict=True);bg.L.eval();bg.encoder.eval();assert sum(v.numel() for v in bg.L.parameters())==467204 and sum(v.numel() for v in bg.encoder.parameters())==2735106;before=weights_hash(bg);structures,search=context_inference(bg,x);arrays=dict(history=x,assignments=np.asarray([s['edge_idx_inv'] for s in structures]));rows=[]
 for H in [1,10,50]:
  a=rollout(bg,structures,x[:,-1],H,0.);b=rollout(bg,structures,x[:,-1],H,111.);assert a.shape==(4,H,5,4) and np.isfinite(a).all() and np.isfinite(b).all();error=float(abs(a-b).max());assert error==0.,(H,error);arrays['native_H'+str(H)]=a;arrays['padding111_H'+str(H)]=b;rows.append(dict(horizon=H,shape=list(a.shape),padding_max_abs=error,finite=True))
 refs=[];reference_errors=[]
 for i in range(2):
  y=numpy_rollout(bg,structures[i],x[i,-1],50);assert np.isfinite(y).all();err=float(abs(y-arrays['native_H50'][i]).max());relative=err/max(1.,float(abs(y).max()));assert relative<=2e-4,(i,relative);refs.append(y);reference_errors.append(dict(row=i,absolute=err,relative_scaled=relative))
 arrays['independent_numpy_H50']=np.asarray(refs);assert weights_hash(bg)==before and sha(p)==ck['sha256'];assert bg.module_updates71==bg.proposal_updates71==0;bg.writer.close();dest=storage('artifact')/'OPEN_LOOP_QUALIFICATION71_V1.npz';assert not dest.exists();np.savez_compressed(dest,**arrays)
 obj=dict(utc=utc(),status='PASS_COMMON_HISTORY_FREE_ROLLOUT_INTERFACE',checkpoint=ck,protocol_sha256=sha(ROOT/'OPEN_LOOP_PROTOCOL71_V1.md'),source_bodies=EXECUTED_BODIES,source_revision=ACQ['revision'],module_parameters=467204,proposal_parameters=2735106,search=search,horizons=rows,numpy_reference=reference_errors,weights_unchanged_sha256=before,array_path=str(dest),array_sha256=sha(dest),train_rows=[0,1,2,3],val_forecast_rows=0,eval_forecast_rows=0,module_updates=0,proposal_updates=0,wall_seconds=time.perf_counter()-start,performance_test=False,new_method=False,accepted_final_directions=0)
 dump(OUT/'OPEN_LOOP_QUALIFICATION71_V1.json',obj);print(json.dumps({k:obj[k] for k in ['status','checkpoint','horizons','numpy_reference','wall_seconds']}),flush=True)

if __name__=='__main__':qualify()
