"""Execute native neural API and freeze/gradient contracts before formal data access."""
from common85_v1 import *
BC,ALIGN,HASHES=native_modules()
def make(nu,seed):
    torch.manual_seed(seed)
    return BC.BCObsAgent(dict(obs_dim=2,robot_obs_dim=2,obj_obs_dim=0,lat_obs_dim=2),dict(act_dim=nu+1),'cpu',n_layers=2,hidden_dim=64,lr=3e-4)
class Quiet:
    def add_scalar(self,*args):pass
rows=[]
for nu in [1,2]:
    a=make(nu,20268500);rng=np.random.default_rng(20268500)
    x=tensor(rng.normal(0,.3,(16,2)));xn=x+.05*tensor(rng.normal(size=(16,2)));u=tensor(rng.uniform(-1,1,(16,nu)))
    a.update_dyn_cons(x,torch.cat([u,torch.zeros((16,1))],dim=-1),xn,Quiet(),1)
    freeze(a);frozen=pack(a);b=copy.deepcopy(a);trainable(b,['obs_enc','obs_dec'])
    aligner=ALIGN.ObsAligner(a,b,'cpu',n_layers=2,hidden_dim=64,lr=3e-4)
    aligner.update_disc(x,u,xn,u,Quiet(),1);aligner.update_gen(x,u,xn,xn,u,x,Quiet(),1)
    source_changed=max(float(torch.max(torch.abs(z-frozen[n][k]))) for n,m in pack(a).items() for k,z in m.items())
    assert source_changed==0
    assert all(not p.requires_grad for m in a.modules for p in m.parameters())
    c=copy.deepcopy(a);params=trainable(c,['obs_enc','obs_dec','fwd_dyn']);opt=torch.optim.Adam(params,lr=3e-4)
    yy=tensor(rng.normal(0,.3,(4,9,2)));uu=tensor(rng.uniform(-1,1,(4,8,nu)));opt.zero_grad();loss,recon=output_loss(c,yy,uu);(loss+.1*recon).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    opt.step();freeze(c);prediction=predict(c,yy[:,0].numpy(),uu.numpy());assert np.isfinite(prediction).all()
    rows.append(dict(nu=nu,native_update_executed=True,native_aligner_updates_executed=True,source_max_parameter_change=source_changed,joint_loss=float(loss.detach()),prediction_shape=list(prediction.shape),source_trainable_parameter_count=sum(p.numel() for n in ['obs_enc','obs_dec','fwd_dyn','inv_dyn'] for p in getattr(a,n).parameters())))
result=dict(status='PASS',utc=utc(),native_hashes=HASHES,rows=rows,formal_source_or_target_data_accessed=False,numerical_source=sha(Path(__file__)),protocol_sha256=sha(C/'PROTOCOL85_V1.md'),device='cpu',numpy=np.__version__,torch=torch.__version__)
write(O/'QUALIFICATION85_V1.json',result);print(json.dumps(result,indent=2),flush=True)
