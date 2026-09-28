import numpy as np,pandas as pd
from scipy.optimize import linprog
from pathlib import Path
p=Path(__file__).resolve().parents[1]
f=pd.read_csv(p/'outputs/support_removal_manifest.csv')
z=f[~f.keep_Z];r=f[~f.keep_R]
nblocks=z.block_id.nunique();ranks=r.block_id.value_counts()
blocks=ranks.head(nblocks).index
non=f[f.block_id.isin(blocks)&~f.target_cell]
ks=f.state_cluster.max()+1;ka=f.action_cluster.max()+1
chosen=set(tuple(x) for x in f[f.target_cell][['state_cluster','action_cluster']].drop_duplicates().to_numpy())
cells=[(s,a) for s in range(ks) for a in range(ka) if (s,a) not in chosen]
caps=non.groupby(['state_cluster','action_cluster']).size()
A=np.zeros((ks+ka-1,len(cells)))
for j,(s,a) in enumerate(cells):
 A[s,j]=1
 if a<ka-1:A[ks+a,j]=1
goal=np.r_[np.bincount(z.state_cluster,minlength=ks),np.bincount(z.action_cluster,minlength=ka)[:-1]]
lp=linprog(np.random.default_rng(0).uniform(0,1e-4,len(cells)),A_eq=A,b_eq=goal,
 bounds=[(0,caps.get(c,0)) for c in cells],method='highs')
print('blocks',nblocks,'capacity',len(non),'patients',non.subjectid.nunique(),'success',lp.success,lp.message)
print('goal action',np.bincount(z.action_cluster,minlength=ka))
print('capacity action',np.bincount(non.action_cluster,minlength=ka))
