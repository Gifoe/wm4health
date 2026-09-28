"""Exact-size, exact-block-count random removal preserving target cells."""
import numpy as np,pandas as pd
from scipy.optimize import linprog
from pathlib import Path
p=Path(__file__).resolve().parents[1];out=p/'outputs';arrays=out/'arrays'
f=pd.read_csv(out/'support_removal_manifest.csv');z=f[~f.keep_Z]
blocks=np.load(arrays/'random_matched_blocks.npy');need=len(z)
eligible=f[f.block_id.isin(blocks)&~f.target_cell].copy()
assert eligible.block_id.nunique()==z.block_id.nunique()==len(blocks)
assert eligible.subjectid.nunique()==z.subjectid.nunique()
ks=f.state_cluster.max()+1;ka=f.action_cluster.max()+1
cells=sorted(set(zip(eligible.state_cluster,eligible.action_cluster)))
A=np.zeros((ks+ka-1,len(cells)))
for j,(s,a) in enumerate(cells):
 A[s,j]=1
 if a<ka-1:A[ks+a,j]=1
goal=np.r_[np.bincount(z.state_cluster,minlength=ks),np.bincount(z.action_cluster,minlength=ka)[:-1]]
rng=np.random.default_rng(117)
baseline=None;quotas=None
group=eligible.groupby('block_id')
for attempt in range(100):
 picked=np.array([rng.choice(g.index.to_numpy()) for _,g in group],dtype=int)
 first=eligible.loc[picked]
 residual=goal-np.r_[np.bincount(first.state_cluster,minlength=ks),
                     np.bincount(first.action_cluster,minlength=ka)[:-1]]
 rem=eligible.drop(index=picked)
 cap=rem.groupby(['state_cluster','action_cluster']).size()
 lp=linprog(rng.uniform(0,1e-4,len(cells)),A_eq=A,b_eq=residual,
            bounds=[(0,cap.get(c,0)) for c in cells],method='highs')
 if lp.success:
  q=np.rint(lp.x).astype(int)
  if np.allclose(A@q,residual,atol=1e-3) and q.sum()+len(picked)==need:
   baseline=picked;quotas=q;break
if baseline is None:raise RuntimeError('No feasible one-window-per-block random control after 100 seeded attempts')
chosen=list(baseline)
for cell,n in zip(cells,quotas):
 if n==0:continue
 g=rem[(rem.state_cluster==cell[0])&(rem.action_cluster==cell[1])]
 chosen.extend(rng.choice(g.index.to_numpy(),size=n,replace=False).tolist())
chosen=np.asarray(chosen,dtype=int)
assert len(chosen)==need and len(np.unique(chosen))==need
rkeep=np.ones(len(f),bool);rkeep[chosen]=False
assert rkeep[f.target_cell.to_numpy()].all()
r=f[~rkeep]
assert r.block_id.nunique()==z.block_id.nunique()
assert r.subjectid.nunique()==z.subjectid.nunique()
assert np.array_equal(np.bincount(r.state_cluster,minlength=ks),np.bincount(z.state_cluster,minlength=ks))
assert np.array_equal(np.bincount(r.action_cluster,minlength=ka),np.bincount(z.action_cluster,minlength=ka))
f['keep_R']=rkeep;f.to_csv(out/'support_removal_manifest.csv',index=False)
np.save(arrays/'train_keep_R.npy',rkeep)
summary=pd.read_csv(out/'training_pool_summary.csv')
row=summary.condition=='R'
summary.loc[row,'windows']=rkeep.sum();summary.loc[row,'removed_windows']=need
summary.loc[row,'blocks']=f.loc[rkeep,'block_id'].nunique()
summary.loc[row,'patients']=f.loc[rkeep,'subjectid'].nunique()
summary.loc[row,'cases']=f.loc[rkeep,'caseid'].nunique()
summary.to_csv(out/'training_pool_summary.csv',index=False)
print('EXACT_RANDOM_CONTROL',{'windows':need,'touched_blocks':r.block_id.nunique(),
 'patients':r.subjectid.nunique(),'cases':r.caseid.nunique(),'baseline_attempt':attempt,
 'state_margins_exact':True,'action_margins_exact':True,'target_cell_retained':True},flush=True)
