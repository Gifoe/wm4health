from common87_v1 import *
from physics87_v1 import *
import time
def main():
 started=time.perf_counter();protocol=sha(C/'PROTOCOL87_V1.md');manifest=[];folder=RAW/'population_v1';folder.mkdir(exist_ok=False)
 for split,seed,n in [('train',2026870,8192),('val',2026871,1024)]:
  for fi,family in enumerate(FAMILIES):
   rng=np.random.default_rng(np.random.SeedSequence([seed,fi]));p=parameters(family,n,rng);initial=rng.normal(0,.8,(n,9,2));paths=trajectories(family,p,initial)
   accepted=(np.abs(paths)<2.5).all(axis=(1,2,3));ids=np.flatnonzero(accepted);required=1024 if split=='train' else 128
   path=folder/f'{split}_{family}_v1.npz';save(path,parameters=p,initial=initial,paths=paths,accepted=accepted,ids=np.arange(n),times=np.arange(201)*.05)
   row=dict(split=split,family=family,seed=seed,pool_count=n,accepted_count=int(accepted.sum()),retained_fraction=float(accepted.mean()),all_finite=True,max_abs=float(abs(paths).max()),array_path=str(path),array_sha256=sha(path));manifest.append(row);log(dict(stage='population',**row))
   assert len(ids)>=required,('PROSPECTIVE_ABORT_INSUFFICIENT_ACCEPTED',row,required)
   count=1024 if split=='train' else 128
   for arm in ['whole_filter','resampled_ic','unselected']:
    pick=ids[:count] if arm!='unselected' else np.arange(count)
    if arm=='resampled_ic':
     rr=np.random.default_rng(np.random.SeedSequence([seed,fi,987]));ic=rr.normal(0,.8,(count,2,2));context=trajectories(family,p[pick],ic,steps=32)
    else:context=paths[pick,:2,:33].copy();ic=initial[pick,:2].copy()
    # Query/measurement budgets and noise realization are exactly common across arms.
    qrng=np.random.default_rng(np.random.SeedSequence([seed,fi,444]));locations=qrng.uniform(-2.5,2.5,(count,32,2)).astype(np.float32);noise=qrng.normal(0,.01,context.shape).astype(np.float32)
    observed=context+noise;fx=field(family,p[pick],locations).astype(np.float32);fxobs=field(family,p[pick],context).astype(np.float32)
    out=folder/f'{split}_{family}_{arm}_v1.npz';save(out,mechanism_ids=pick,parameters=p[pick],initial=ic,context_clean=context,context=observed,noise=noise,locations=locations,drift=fx,drift_obs=fxobs)
    manifest.append(dict(split=split,family=family,arm=arm,count=count,array_path=str(out),array_sha256=sha(out)))
 write(O/'DATA_MANIFEST87_V1.json',dict(utc=utc(),status='FIXED_TRAIN_VAL_ONLY_CREATED',protocol_sha256=protocol,files=manifest,eval_opened=False,elapsed_seconds=time.perf_counter()-started))
 log(dict(stage='generation_complete',eval_opened=False,seconds=time.perf_counter()-started))
if __name__=='__main__':main()
