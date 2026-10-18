from native87_v1 import *
from physics87_v1 import field,FAMILIES
import time,gc
ARMS=['whole_filter','resampled_ic','unselected'];SEEDS=[2026872,2026873,2026874]
def dataset(split,arm):
 rows=[]
 for fam in FAMILIES:
  with np.load(RAW/'population_v1'/f'{split}_{fam}_{arm}_v1.npz') as z:d={k:z[k] for k in z.files}
  # Native training appends observations as supervised locations before corruption.
  # With externally added noise, those appended locations are noisy too: evaluate
  # the known synthetic field at those actual locations, not at the clean states.
  assert np.allclose(d['drift_obs'],field(fam,d['parameters'],d['context_clean']),atol=2e-5,rtol=2e-6)
  d['drift_obs_native']=field(fam,d['parameters'],d['context']).astype(np.float32);rows.append(d)
 return {k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
def get_batch(d,idx,device='cuda'):return batch(d['context'][idx],d['locations'][idx],d['drift'][idx],d['drift_obs_native'][idx],device)
def validation(model,val):
 model.eval();total=0.;count=0
 with torch.no_grad():
  for first in range(0,len(val['context']),16):
   ix=np.arange(first,min(first+16,len(val['context'])));out=model(get_batch(val,ix))['losses'];v=float(out['l1']);assert np.isfinite(v);total+=v*len(ix);count+=len(ix)
 return total/count
def main():
 start=time.perf_counter();check_native();assert read(O/'NATIVE_SMOKE87_V1.json')['status']=='PASS_NATIVE_FULL_PARAMETER_GRADIENT_AND_GPU';manifest=read(O/'DATA_MANIFEST87_V1.json')
 for row in manifest['files']:assert sha(row['array_path'])==row['array_sha256']
 assert not manifest['eval_opened'];data={arm:dataset('train',arm) for arm in ARMS};val=dataset('val','unselected')
 for k in ['mechanism_ids','parameters','locations','drift','noise']:assert np.array_equal(data[ARMS[0]][k],data[ARMS[1]][k]),k
 for a in ARMS[1:]:assert np.array_equal(data[a]['noise'],data[ARMS[0]]['noise'])
 folder=CKPT/'native_adaptation_v1';folder.mkdir(exist_ok=False);results=[];write(O/'TRAIN_OPENING87_V1.json',dict(utc=utc(),protocol_sha256=sha(C/'PROTOCOL87_V1.md'),script_sha256=sha(__file__),data_manifest_sha256=sha(O/'DATA_MANIFEST87_V1.json'),original_smoke_sha256=sha(O/'NATIVE_SMOKE87_V1.json'),arms=ARMS,seeds=SEEDS,max_updates=3000,appended_location_targets='exact synthetic field at externally noisy observation coordinates',eval_opened=False))
 for seed in SEEDS:
  for arm in ARMS:
   torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);m=load_native();rng=np.random.default_rng(seed);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);trace=[];fit_start=time.perf_counter();init_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};best=None
   for step in range(3001):
    if step%500==0:
     v=validation(m,val);path=folder/f'seed{seed}_{arm}_step{step:05d}_v1.pt'
     torch.save(dict(state_dict=m.state_dict(),step=step,seed=seed,arm=arm,val_l1=v,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),batch_rng=rng.bit_generator.state,native_source_manifest_sha256=sha(ROOT/'cycle80/outputs/SOURCE_MANIFEST80_V1.json')),path)
     row=dict(step=step,val_normalized_field_l1=v,checkpoint_path=str(path),checkpoint_sha256=sha(path),elapsed_seconds=time.perf_counter()-fit_start);trace.append(row)
     if best is None or v<best['val_normalized_field_l1']:best=dict(row)
     progress=dict(utc=utc(),status='NATIVE_NONLINEAR_PAIRED_TRAINING',seed=seed,arm=arm,step=step,closed_fits=len(results),val_l1=v,best_step=best['step'],eval_opened=False,gpu_peak_bytes=torch.cuda.max_memory_allocated())
     # Live status is the sole mutable record; immutable trace/checkpoints retain history.
     tmp=O/'TRAIN_PROGRESS87_V1.tmp';tmp.write_text(json.dumps(progress),encoding='utf-8');tmp.replace(O/'TRAIN_PROGRESS87_V1.json');log(dict(stage='native_checkpoint',**{k:v for k,v in progress.items() if k!='utc'}))
    if step==3000:break
    m.train();ix=rng.integers(0,len(data[arm]['context']),16);opt.zero_grad(set_to_none=True);out=m(get_batch(data[arm],ix));loss=loss_value(out);assert torch.isfinite(loss),(seed,arm,step,'nonfinite loss');loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step()
   changed=sum(not torch.equal(init_state[k],v.detach().cpu()) for k,v in m.state_dict().items());assert changed>0
   fit=dict(seed=seed,arm=arm,trace=trace,selected=best,updates=3000,changed_state_tensors=changed,initial_state_identical_to_pinned_checkpoint=True,wall_seconds=time.perf_counter()-fit_start)
   write(O/f'FIT87_seed{seed}_{arm}_V1.json',fit);results.append(fit);log(dict(stage='fit_complete',seed=seed,arm=arm,best_step=best['step'],val_l1=best['val_normalized_field_l1'],changed_state_tensors=changed))
   del m,opt,init_state;gc.collect();torch.cuda.empty_cache()
 assert len(results)==9;check_native();write(O/'SELECTION87_V1.json',dict(utc=utc(),status='NINE_NATIVE_FITS_SELECTED_BEFORE_EVALUATION',fits=results,selection='minimum common unselected VAL normalized field l1, earliest tie',eval_opened=False,parameters=12968068,wall_seconds=time.perf_counter()-start,protocol_sha256=sha(C/'PROTOCOL87_V1.md'),data_manifest_sha256=sha(O/'DATA_MANIFEST87_V1.json')));log(dict(stage='all_training_complete',closed_fits=9,eval_opened=False))
if __name__=='__main__':main()
