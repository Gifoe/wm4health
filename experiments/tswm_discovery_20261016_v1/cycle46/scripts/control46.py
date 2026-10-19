"""Ordinary anchor-aligned PCA/LGSSM; no generator truths or latent labels."""
from common46 import *

def regular(cov,floor=1e-5):
 cov=(cov+cov.T)/2;w,v=np.linalg.eigh(cov);return (v*np.maximum(w,floor))@v.T
def regress(x,y,ridge):return np.linalg.solve(x.T@x+ridge*np.eye(x.shape[-1]),x.T@y)

class AlignedSSM:
 def __init__(self,train,anchors,ridge):
  self.ridge=ridge;self.views={};latent={};ref=None
  for v in range(4):
   samples=np.concatenate([x.reshape(-1,Y) for k,x in train.items() if k.startswith(f'v{v}_')]);mean=samples.mean(0);_,_,V=np.linalg.svd(samples-mean,full_matrices=False);W=V[:D].T;anchor=(anchors[:,0,v*Y:(v+1)*Y]-mean)@W
   if v==0:ref=anchor
   M=np.eye(D) if v==0 else regress(anchor,ref,ridge);self.views[v]={'pca_mean':mean,'W':W,'alignment':M}
   for k,x in train.items():
    if k.startswith(f'v{v}_'):latent[k]=(x-mean)@W@M
   zs=np.concatenate([z.reshape(-1,D) for k,z in latent.items() if k.startswith(f'v{v}_')]);features=np.column_stack([zs,np.ones(len(zs))]);mapping=regress(features,samples,ridge);residual=samples-features@mapping;self.views[v].update(H=mapping[:D].T,offset=mapping[D],R=regular(np.cov(residual.T)))
  self.systems={}
  for s in range(4):
   x=np.concatenate([z[:,:-1].reshape(-1,D) for k,z in latent.items() if k.endswith(f's{s}')]);y=np.concatenate([z[:,1:].reshape(-1,D) for k,z in latent.items() if k.endswith(f's{s}')]);A=regress(x,y,ridge).T;radius=max(abs(np.linalg.eigvals(A)))
   if radius>.999:A*=.999/radius
   residual=y-x@A.T;self.systems[s]={'A':A,'Q':regular(np.cov(residual.T))}
  initial=np.concatenate([z[:,0] for z in latent.values()]);self.m0=initial.mean(0);self.P0=regular(np.cov(initial.T));self.fit_proxy='PCA4, static least-squares alignment, VAR1, affine full-covariance emissions'
 def sample(self,history,view,system,n,draw_seed):
  gen=np.random.default_rng(draw_seed);obs=self.views[view];dyn=self.systems[system];H=obs['H'];A=dyn['A'];Q=dyn['Q'];Rm=obs['R'];ms=[];ps=[]
  for values in history:
   m=self.m0.copy();P=self.P0.copy()
   for t,y in enumerate(values):
    if t:m=A@m;P=A@P@A.T+Q
    S=H@P@H.T+Rm;K=np.linalg.solve(S,H@P).T;m=m+K@(y-obs['offset']-H@m);left=np.eye(D)-K@H;P=regular(left@P@left.T+K@Rm@K.T)
   ms.append(m);ps.append(P)
  z=np.asarray(ms)[:,None]+np.einsum('bnj,bij->bni',gen.normal(size=(len(history),n,D)),np.linalg.cholesky(np.asarray(ps)));paths=[];LQ=np.linalg.cholesky(Q);LR=np.linalg.cholesky(Rm)
  for t in range(FUTURE):z=z@A.T+gen.normal(size=z.shape)@LQ.T;paths.append(z@H.T+obs['offset']+gen.normal(size=(*z.shape[:-1],Y))@LR.T)
  return np.stack(paths,2).astype('float32')
 def save(self,p):
  values={'m0':self.m0,'P0':self.P0,'ridge':np.asarray(self.ridge)}
  for v,params in self.views.items():values.update({f'view{v}_{k}':x for k,x in params.items()})
  for s,params in self.systems.items():values.update({f'system{s}_{k}':x for k,x in params.items()})
  np.savez_compressed(p,**values)

def main():
 out=R/'outputs/ordinary_control_v1';assert not out.exists();out.mkdir(parents=True);train=dict(np.load(R/'data/train.npz'));val=dict(np.load(R/'data/val.npz'));anchors=np.load(R/'data/anchors.npz')['X'];scores=[];best=None
 for ridge in [1e-6,1e-4,.01,.1,1.]:
  model=AlignedSSM(train,anchors,ridge);es=[]
  for j,k in enumerate(sorted(val)):
   draws=model.sample(val[k][:8,:HIST],int(k[1]),int(k[-1]),32,202646700+j);es.extend(energy(torch.from_numpy(draws),val[k][:8,HIST:]).tolist())
  score=float(np.mean(es));scores.append({'ridge':ridge,'val_energy':score})
  if best is None or score<best[0]:best=(score,model)
 best[1].save(out/'selected_parameters.npz');dump(out/'selection.json',{'status':'VAL_SELECTION_COMPLETE_EVAL_UNOPENED','scores':scores,'selected_ridge':best[1].ridge,'selected_val_energy':best[0],'source_sha256':sha(__file__),'supplement_sha256':sha(R/'CONTROL_SUPPLEMENT.md'),'parameters_sha256':sha(out/'selected_parameters.npz'),'fit_proxy':best[1].fit_proxy});print(json.dumps(scores),flush=True)
if __name__=='__main__':main()
