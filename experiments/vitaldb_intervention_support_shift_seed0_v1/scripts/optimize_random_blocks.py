"""Find a same-block-count random control with exact state/action margins."""
import numpy as np,pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix,vstack
from pathlib import Path
p=Path(__file__).resolve().parents[1]
f=pd.read_csv(p/'outputs/support_removal_manifest.csv')
z=f[~f.keep_Z];r=f[~f.keep_R]
patients=set(z.subjectid.unique());non=f[f.subjectid.isin(patients)&~f.target_cell].copy()
blocks=np.sort(non.block_id.unique());bid={b:i for i,b in enumerate(blocks)}
cells=sorted(set(zip(non.state_cluster,non.action_cluster)))
cid={c:i for i,c in enumerate(cells)}
goal=r.groupby(['state_cluster','action_cluster']).size()
row=[];col=[];data=[]
for (b,s,a),n in non.groupby(['block_id','state_cluster','action_cluster']).size().items():
 row.append(cid[(s,a)]);col.append(bid[b]);data.append(n)
M=coo_matrix((data,(row,col)),shape=(len(cells),len(blocks))).tocsr()
subjects=sorted(patients);pid={q:i for i,q in enumerate(subjects)}
P=coo_matrix((np.ones(len(blocks)),([pid[q] for q in non.drop_duplicates('block_id').set_index('block_id').loc[blocks].subjectid],np.arange(len(blocks)))),shape=(len(subjects),len(blocks))).tocsr()
ks=f.state_cluster.max()+1;ka=f.action_cluster.max()+1
S=coo_matrix((np.ones(len(cells)),([c[0] for c in cells],np.arange(len(cells)))),shape=(ks,len(cells))).tocsr()@M
A=coo_matrix((np.ones(len(cells)),([c[1] for c in cells],np.arange(len(cells)))),shape=(ka,len(cells))).tocsr()@M
Q=vstack([S,A,P,coo_matrix(np.ones((1,len(blocks))))]).tocsr()
lower=np.r_[np.bincount(z.state_cluster,minlength=ks),np.bincount(z.action_cluster,minlength=ka),
            np.ones(len(subjects)),z.block_id.nunique()]
upper=np.r_[np.full(ks+ka+len(subjects),np.inf),z.block_id.nunique()]
removed=r.block_id.value_counts();w=np.array([removed.get(b,0) for b in blocks])
capacity=np.asarray(M.sum(0)).ravel()
cost=-(w+0.01*capacity)+np.random.default_rng(0).uniform(0,.001,len(blocks))
print('MILP',len(blocks),len(cells),len(subjects),'target blocks',z.block_id.nunique(),flush=True)
out=milp(cost,integrality=np.ones(len(blocks)),bounds=Bounds(np.zeros(len(blocks)),np.ones(len(blocks))),
 constraints=LinearConstraint(Q,lower,upper),options={'time_limit':300,'mip_rel_gap':.02})
print('MILP_RESULT',out.message,'feasible',out.x is not None,flush=True)
if out.x is not None:
 selected=blocks[out.x>.5]
 print('selected',len(selected),'capacity',int(np.asarray(M[:,out.x>.5].sum()).item()),flush=True)
 np.save(p/'outputs/arrays/random_matched_blocks.npy',selected)
 from scipy.optimize import linprog
 non2=non[non.block_id.isin(selected)]
 caps=non2.groupby(['state_cluster','action_cluster']).size()
 Eq=np.zeros((ks+ka-1,len(cells)))
 for j,(s,a) in enumerate(cells):
  Eq[s,j]=1
  if a<ka-1:Eq[ks+a,j]=1
 target=np.r_[np.bincount(z.state_cluster,minlength=ks),np.bincount(z.action_cluster,minlength=ka)[:-1]]
 lp=linprog(np.random.default_rng(1).uniform(0,1e-4,len(cells)),A_eq=Eq,b_eq=target,
            bounds=[(0,caps.get(c,0)) for c in cells],method='highs')
 print('MARGINAL_LP',lp.success,lp.message,flush=True)
