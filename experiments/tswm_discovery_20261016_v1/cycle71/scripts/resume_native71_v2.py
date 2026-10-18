"""Checkpoint continuation adapter; the original epoch-loop body stays exact."""
import sys,pathlib,ast,copy,types,random,time,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native71_v3 import *

def temperature_from_log(seed,tag,epoch,initial=-9.8,minimum=-100.):
 p=storage('artifact')/tag/'logs'/('seed'+str(seed))/'scalars.jsonl';values={}
 for line in p.open(encoding='utf-8'):
  row=json.loads(line)
  if row['tag']=='SA/temp' and row['step']<=epoch:values[row['step']]=row['value']
 assert len(values)==epoch+1
 t=np.exp(initial)
 for step in range(epoch+1):
  options=[t*1.1,max(t/1.1,np.exp(minimum))];matches=[v for v in options if float(np.log10(v))==values[step]];assert len(matches)==1,(step,matches);t=matches[0]
 return float(t)

def bind_continuation(bg):
 p=SOURCE/'modular_metalearning.py';tree=ast.parse(p.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BounceGrad');fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='SAConfig_SGDModules'))
 loop=next(n for n in fn.body if isinstance(n,ast.For));loop_hash=hashlib.sha256(ast.dump(ast.Module(body=loop.body,type_ignores=[]),include_attributes=False).encode()).hexdigest();changes=[]
 for n in fn.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='temp':n.value=ast.parse('self.resume_temperature71',mode='eval').body;changes.append('restore temperature')
  if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self.S.initialize_all_structures':n.value=ast.parse('self.restore_continuation71()',mode='eval').body;changes.append('initialize structures then restore checkpoint/RNG')
 call=loop.iter;assert isinstance(call,ast.Call) and ast.unparse(call.func)=='tqdm' and len(call.args)==1
 inner=call.args[0];assert ast.unparse(inner)=='range(optimization_steps)';inner.args.insert(0,ast.parse('self.resume_start71',mode='eval').body);changes.append('start at saved epoch+1')
 assert len(changes)==3 and hashlib.sha256(ast.dump(ast.Module(body=loop.body,type_ignores=[]),include_attributes=False).encode()).hexdigest()==loop_hash
 namespace=dict(BounceGrad.SAConfig_SGDModules.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(p)+'::continuation', 'exec'),namespace)
 bg.SAConfig_SGDModules=types.MethodType(namespace['SAConfig_SGDModules'],bg)
 return dict(original_source_sha256=sha(p),unchanged_epoch_loop_body_sha256=loop_hash,entry_only_changes=changes,transformed_function_ast_sha256=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest())

class ResumedNative(InstrumentedNative):
 def configure_restore71(self,state,record,prefix,temperature):
  self.restore_payload71=state;self.resume_start71=int(state['epoch'])+1;self.resume_temperature71=temperature
  self.receipts71=copy.deepcopy(prefix);self.selected71=min(self.receipts71,key=lambda r:r['val_loss']);self.best71=self.selected71['val_loss'];self.parent_checkpoint71=record
 def restore_continuation71(self):
  state=self.restore_payload71
  # Retain all native nonrandom structure bookkeeping, then restore actual graphs.
  self.S.initialize_all_structures(T=self.T)
  self.L.load_state_dict(state['module_state'],strict=True);self.encoder.load_state_dict(state['proposal_state'],strict=True)
  self.SOpt.load_state_dict(state['module_optimizer']);self.encoder_opt.load_state_dict(state['proposal_optimizer']);self.SOpt_scheduler.load_state_dict(state['scheduler'])
  self.S.TrainStructures=copy.deepcopy(state['train_structures']);self.S.ValStructures=copy.deepcopy(state['val_structures'])
  self.SA_running_acc_rate=state['SA_running_acc_rate'];self.SA_running_factor=state['SA_running_factor'];self.mean_current_val=state['metrics']['mean_loss/T_val'];self.writer.scalars=copy.deepcopy(state['metrics'])
  self.module_updates71=state['module_updates'];self.proposal_updates71=state['proposal_updates'];self.DL_Mtrain.rng.bit_generator.state=copy.deepcopy(state['train_shuffle']);self.DL_Mval.rng.bit_generator.state=copy.deepcopy(state['val_shuffle'])
  random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
  assert not self.auto_temp and self.stop_edge_acc<=0 and not self.S.find_node and self.LocalOpt is None and self.load_structures_and_metrics==''
 def save_train_state(self):
  super().save_train_state()
  # Additive lineage receipt; do not alter previous source/checkpoint artifacts.
  dump(OUT/(self.tag71+'_lineage.json'),dict(parent_checkpoint=self.parent_checkpoint71,effective_module_updates=self.module_updates71,effective_proposal_updates=self.proposal_updates71,last_epoch=int(self.step),entry_adapter=self.resume_adapter71),exclusive=False)

def resumed_fit(seed,total_epochs,ntrain,tag,manifest,temperature,finish=None):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
 record=manifest['checkpoints'][-1];assert sha(record['path'])==record['sha256'];state=torch.load(record['path'],map_location='cpu',weights_only=False)
 args=parse_native(seed,total_epochs,ntrain,tag);bg=ResumedNative(HGNN_Structure(args),args,tag);bg.configure_restore71(state,record,manifest['checkpoints'],temperature);bg.resume_adapter71=bind_continuation(bg);bg.SAConfig_SGDModules(total_epochs if finish is None else finish);bg.writer.close()
 return bg

def nested_compare(a,b):
 result=dict(tensors=0,max_abs=0.,non_tensor_mismatches=0)
 def walk(x,y):
  if isinstance(x,torch.Tensor):
   assert isinstance(y,torch.Tensor) and x.shape==y.shape;result['tensors']+=1
   delta=float((x.detach().cpu().to(torch.float64)-y.detach().cpu().to(torch.float64)).abs().max()) if x.numel() else 0.;result['max_abs']=max(result['max_abs'],delta)
  elif isinstance(x,np.ndarray):
   if not np.array_equal(x,y):result['non_tensor_mismatches']+=1
  elif isinstance(x,dict):
   assert set(x)==set(y)
   for k in x:walk(x[k],y[k])
  elif isinstance(x,(tuple,list)):
   assert len(x)==len(y)
   for xx,yy in zip(x,y):walk(xx,yy)
  elif x!=y:result['non_tensor_mismatches']+=1
 walk(a,b);return result

def qualify():
 # One changed critical path: uninterrupted three epochs versus checkpoint0 resume.
 seed=2026731;fulltag='resume_full_control71_v1';original=read(OUT/(fulltag+'_result.json'));assert original['epochs']==3;manifest=read(OUT/(fulltag+'_checkpoints.json'));prefix=[r for r in manifest['checkpoints'] if r['epoch']==0];assert len(prefix)==1
 temperature=temperature_from_log(seed,fulltag,0);bg=resumed_fit(seed,3,128,'resume_split_control71_v2',dict(checkpoints=prefix),temperature);last=bg.receipts71[-1];a=torch.load(original['last']['path'],map_location='cpu',weights_only=False);b=torch.load(last['path'],map_location='cpu',weights_only=False)
 checked=['module_state','proposal_state','module_optimizer','proposal_optimizer','scheduler','torch_rng','cuda_rng','numpy_rng','python_rng','train_shuffle','val_shuffle','train_structures','val_structures','SA_running_acc_rate','SA_running_factor','metrics','module_updates','proposal_updates'];comparison=nested_compare({k:a[k] for k in checked},{k:b[k] for k in checked})
 assert comparison['max_abs']<=1e-7 and comparison['non_tensor_mismatches']==0,comparison
 obj=dict(utc=utc(),status='PASS_CHECKPOINT_CONTINUATION_EQUIVALENCE',comparison=comparison,keys_compared=checked,full_checkpoint=original['last'],resumed_checkpoint=last,entry_adapter=bg.resume_adapter71,qualification_only=True,performance_evidence=False,reserved_eval_rows_accessed=0,new_method=False,accepted_final_directions=0)
 dump(OUT/'RESUME_QUALIFICATION71_V2.json',obj);print(json.dumps(obj),flush=True)

def formal():
 assert read(OUT/'RESUME_QUALIFICATION71_V2.json')['status']=='PASS_CHECKPOINT_CONTINUATION_EQUIVALENCE'
 incident=read(OUT/'TRAINING_INCIDENT71_V1.json');manifest=read(OUT/'formal_seed2026711_v3_checkpoints.json');assert manifest['checkpoints'][-1]==incident['last_checkpoint'];seed=2026711;tag='formal_seed2026711_resumed_v2'
 bg=resumed_fit(seed,2000,2048,tag,manifest,float.fromhex(incident['annealing_temperature_exact_replay']['float64_hex']))
 assert bg.module_updates71==bg.proposal_updates71==32000 and bg.step==1999
 change=max(float(abs(v-bg.initial_parameters71[k]).max().item()) for k,v in bg.L.state_dict().items());assert np.isfinite(change) and change>0
 obj=dict(utc=utc(),status='NATIVE_SUBSET_FIT_COMPLETE_AFTER_CHECKPOINT_CONTINUATION',seed=seed,tag=tag,epochs=2000,train_tasks=2048,val_tasks=256,module_updates=bg.module_updates71,proposal_updates=bg.proposal_updates71,module_parameters=sum(p.numel() for p in bg.L.parameters()),proposal_parameters=sum(p.numel() for p in bg.encoder.parameters()),selected=bg.selected71,last=bg.receipts71[-1],checkpoints=bg.receipts71,wall_seconds_this_process=time.perf_counter()-bg.start71,parent_checkpoint=incident['last_checkpoint'],incident_sha256=sha(OUT/'TRAINING_INCIDENT71_V1.json'),resume_qualification_sha256=sha(OUT/'RESUME_QUALIFICATION71_V2.json'),entry_adapter=bg.resume_adapter71,discarded_suffix='Epoch651 and an incomplete652 after checkpoint650; physical work is not represented as effective lineage updates',effective_completed_lineage_updates=32000,source_bodies=EXECUTED_BODIES,source_revision=ACQ['revision'],acquisition_sha256=sha(OUT/'ACQUISITION71_V1.json'),protocol_sha256=sha(ROOT/'PROTOCOL71_V1.md'),gpu_peak_bytes=int(torch.cuda.max_memory_allocated()),reserved_eval_rows_accessed=0,is_original_full_paper_reproduction=False,new_method=False,accepted_final_directions=0)
 dump(OUT/(tag+'_result.json'),obj);print(json.dumps({k:obj[k] for k in ['status','seed','module_updates','selected','wall_seconds_this_process']}),flush=True)

if __name__=='__main__':
 assert sys.argv[1] in ('qualify','formal')
 qualify() if sys.argv[1]=='qualify' else formal()
