"""Confirm labels are diagnostics, not hidden structural supervision."""
import sys,pathlib,random,copy
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native71_v3 import *
q=read(OUT/'qualification71_v3_result.json');p=pathlib.Path(q['selected']['path']);assert sha(p)==q['selected']['sha256'];state=torch.load(p,map_location='cpu',weights_only=False)
def tensor_check(x):
 if isinstance(x,torch.Tensor):return dict(tensors=1,nonfinite=int(not torch.isfinite(x).all().item()))
 if isinstance(x,dict):vals=[tensor_check(v) for v in x.values()]
 elif isinstance(x,(list,tuple)):vals=[tensor_check(v) for v in x]
 else:return dict(tensors=0,nonfinite=0)
 return {k:sum(v[k] for v in vals) for k in ['tensors','nonfinite']}
optimizer=tensor_check(state['module_optimizer']);assert optimizer['nonfinite']==0
def run(flip):
 seed=2026714;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);tag='label_audit_'+str(int(flip))+'_v1';args=parse_native(seed,1,128,tag);S=HGNN_Structure(args);bg=InstrumentedNative(S,args,tag);bg.S.initialize_all_structures(bg.T);bg.SA_running_acc_rate=1e-9;bg.SA_running_factor=1e-9;bg.BG_mode='train';rows=next(iter(bg.DL_Mtrain));structs=[bg.S.TrainStructures[x.structure_idx] for x in rows]
 if flip:
  for x in rows:x.Edges=1-x.Edges
 result=bg.batched_bounce(structs,rows,np.exp(-9.8),True);grads={k:p.grad.detach().cpu().numpy().copy() for k,p in bg.L.named_parameters()};accept=result[12]
 if len(result[13]):bg.encoder_step(bg.encoder_loss(result[12],result[13],len(rows)),result[11])
 bg.SOpt.step();weights={k:v.detach().cpu().numpy().copy() for k,v in bg.L.state_dict().items()};proposal={k:v.detach().cpu().numpy().copy() for k,v in bg.encoder.state_dict().items()};bg.writer.close()
 return dict(gradients=grads,weights=weights,proposal=proposal,train_losses=np.array(result[1]),val_losses=np.array(result[2]),raw_edge_accuracy_count=float(result[7]),accepted=len(result[12]),assignments=np.asarray([s['edge_idx_inv'] for s in result[0]]))
a=run(False);b=run(True);errors={}
for field in ['gradients','weights','proposal']:errors[field]=max(float(abs(a[field][k]-b[field][k]).max()) for k in a[field])
for field in ['train_losses','val_losses','assignments']:errors[field]=float(abs(a[field]-b[field]).max())
assert max(errors.values())<1e-6 and a['accepted']==b['accepted'];assert sha(p)==q['selected']['sha256']
raw={}
for label,obj in [('original',a),('flipped',b)]:
 for field in ['gradients','weights','proposal']:
  for key,val in obj[field].items():raw[label+'__'+field+'__'+key]=val
 for field in ['train_losses','val_losses','assignments']:raw[label+'__'+field]=obj[field]
dest=storage('artifact')/'LABEL_AUDIT71_V1.npz';assert not dest.exists();np.savez_compressed(dest,**raw)
dump(OUT/'GRADIENT_AUDIT71_V1.json',dict(utc=utc(),status='PASS',max_errors=errors,qualification_optimizer=optimizer,original_raw_edge_accuracy_count=a['raw_edge_accuracy_count'],flipped_raw_edge_accuracy_count=b['raw_edge_accuracy_count'],accepted_structures=a['accepted'],array_path=str(dest),array_sha256=sha(dest),reserved_eval_rows_accessed=0,qualification_only_not_performance=True,upstream_files_unchanged=True))
print(json.dumps(dict(status='PASS',max_errors=errors,qualification_optimizer=optimizer,accepted_structures=a['accepted'])),flush=True)
