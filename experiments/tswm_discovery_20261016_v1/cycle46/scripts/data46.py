"""Fresh controlled data; latent trajectories never enter learner files."""
from common46 import *
from scipy.linalg import expm,solve_continuous_lyapunov
import datetime

def main():
 out=R/'data';assert not out.exists();out.mkdir(parents=True);rng=np.random.default_rng(20264600)
 def skew(pairs):
  a=np.zeros((D,D))
  for i,j,x in pairs:a[i,j]=x;a[j,i]=-x
  return a
 Fs=[-.25*np.eye(D)+skew([(0,1,.5),(2,3,.3)]),-.05*np.eye(D)+skew([(0,2,.3),(1,3,.2)]),-.08*np.eye(D)+skew([(0,3,.4),(1,2,-.25)])]
 views=[]
 for v in range(4):
  rotation=np.linalg.qr(rng.normal(size=(D,D)))[0];projection=np.linalg.qr(rng.normal(size=(Y,D)))[0]*1.5;views.append((rotation,projection,[.08,.1,.12,.15][v]))
 systems=[]
 for s in range(4):
  F=Fs[0]+(s&1)*Fs[1]+((s>>1)&1)*Fs[2];A=expm(F*.25);P=solve_continuous_lyapunov(F,-.25*np.eye(D));Q=P-A@P@A.T;systems.append((F,A,np.linalg.cholesky((Q+Q.T)/2)))
 def observe(z,v,gen):
  rot,proj,sigma=views[v];u=z@rot.T;return (u+.3*u**3)@proj.T+gen.normal(size=(*z.shape[:-1],Y))*sigma
 def trajectories(n,s,v,gen):
  z=gen.normal(size=(n,D));zs=[z]
  for t in range(1,T):z=z@systems[s][1].T+gen.normal(size=(n,D))@systems[s][2].T;zs.append(z)
  return observe(np.stack(zs,1),v,gen).astype('float32')
 train={};val={};sealed={}
 for v in range(4):
  for s in range(4):
   key=f'v{v}_s{s}'
   if v!=s:
    train[key]=trajectories(64,s,v,np.random.default_rng(20264610+v*4+s));val[key]=trajectories(32,s,v,np.random.default_rng(20264630+v*4+s))
   sealed[key]=trajectories(128 if v==s else 64,s,v,np.random.default_rng(20264650+v*4+s))
 normalizers={}
 for v in range(4):
  all_y=np.concatenate([y for k,y in train.items() if k.startswith(f'v{v}_')]);mean=all_y.mean((0,1));scale=float(np.sqrt(np.mean((all_y-mean)**2)));normalizers[str(v)]={'mean':mean.tolist(),'scalar_std':scale}
  for collection in (train,val,sealed):
   for k in collection:
    if k.startswith(f'v{v}_'):collection[k]=((collection[k]-mean)/scale).astype('float32')
 az=rng.normal(size=(64,1,D));anchors=[]
 for v in range(4):
  norm=normalizers[str(v)];anchors.append((observe(az,v,rng)-np.asarray(norm['mean']))/norm['scalar_std'])
 np.savez_compressed(out/'train.npz',**train);np.savez_compressed(out/'val.npz',**val);np.savez_compressed(out/'sealed_eval.npz',**sealed);np.savez_compressed(out/'anchors.npz',X=np.concatenate(anchors,-1).astype('float32'))
 dump(out/'normalizers.json',normalizers);dump(out/'GENERATOR_TRUTH_NOT_LEARNER_INPUT.json',{'F_primitives':[f.tolist() for f in Fs],'Q':(.25*np.eye(D)).tolist(),'views':[{'rotation':a.tolist(),'projection':b.tolist(),'noise_std':c} for a,b,c in views],'initial_distribution':'N(0,I)','dt':.25,'timepoints':T,'latent_values_saved':False})
 manifest={'status':'DATA_FROZEN','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'protocol_sha256':sha(R/'PROTOCOL.md'),'train_cells':list(train),'heldout_cells':[f'v{i}_s{i}' for i in range(4)],'normalizer_fit':'TRAIN temporal observations only','anchors':'64 simultaneous static views from common initial population, no temporal or latent labels; available identically to all models','files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
 dump(out/'MANIFEST.json',manifest);print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
