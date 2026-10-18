"""Source-only capacity repair; preserve original campaign and all original gates."""
import common85_v1 as common
import run85_v1 as old
from common85_v1 import *
import time

STAGE=C/'outputs'/'capacity_v2'
STAGE.mkdir(parents=True,exist_ok=True)
old.O=STAGE;common.O=STAGE
old.RAW=Path(CONFIG['artifact_root'])/'cycle85'/'anchored_screen_capacity_v2'
old.DATA=Path(CONFIG['data_root'])/'cycle85'/'anchored_screen_capacity_v2'
old.WEIGHTS=Path(CONFIG['checkpoint_root'])/'cycle85'/'anchored_screen_capacity_v2'

def make(seed,nu):
    torch.manual_seed(seed)
    return old.BC.BCObsAgent(dict(obs_dim=2,robot_obs_dim=2,obj_obs_dim=0,lat_obs_dim=4),dict(act_dim=nu+1),'cpu',n_layers=2,hidden_dim=128,lr=3e-4)
old.new_agent=make

class Quiet:
    def add_scalar(self,*args):pass
def qualify():
    a=make(20268500,2);r=np.random.default_rng(20268500)
    x=tensor(r.normal(0,.3,(16,2)));xn=x+.05*tensor(r.normal(size=(16,2)));u=tensor(r.uniform(-1,1,(16,2)))
    a.update_dyn_cons(x,torch.cat([u,torch.zeros((16,1))],dim=-1),xn,Quiet(),1);freeze(a);before=pack(a)
    b=copy.deepcopy(a);trainable(b,['obs_enc','obs_dec']);aligner=old.ALIGN.ObsAligner(a,b,'cpu',n_layers=2,hidden_dim=64,lr=3e-4)
    aligner.update_disc(x,u,xn,u,Quiet(),1);aligner.update_gen(x,u,xn,xn,u,x,Quiet(),1)
    assert all(torch.equal(z,pack(a)[n][k]) for n,m in before.items() for k,z in m.items())
    c=copy.deepcopy(a);params=trainable(c,['obs_enc','obs_dec','fwd_dyn']);yy=tensor(r.normal(0,.3,(4,9,2)));uu=tensor(r.uniform(-1,1,(4,8,2)))
    loss,recon=output_loss(c,yy,uu);(loss+.1*recon).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    record=dict(status='PASS',utc=utc(),native_hashes=old.NATIVE_HASHES,latent_dim=4,hidden_dim=128,n_layers=2,formal_data_accessed=False,protocol_sha256=sha(C/'PROTOCOL85_V1.md'),amendment_sha256=sha(C/'CAPACITY_AMENDMENT85_V1.md'))
    write(STAGE/'QUALIFICATION85_CAPACITY_V1.json',record)

def main():
    original=json.loads((C/'outputs'/'RESULT85_V1.json').read_text());assert original['closed_cases']==8
    assert not list((C/'outputs').glob('CASE85_brusselator*_V1.json'))
    for seed in SEEDS:
        r=json.loads((C/'outputs'/f'SOURCE_READY85_brusselator_{seed}_V1.json').read_text());assert not r['qualified']
    qualify();old.RAW.mkdir(parents=True,exist_ok=True);sources=[];cases=[]
    log(dict(kind='capacity_campaign_start',pid=os.getpid(),latent_dim=4,hidden_dim=128,amendment_sha256=sha(C/'CAPACITY_AMENDMENT85_V1.md')))
    for seed in SEEDS:
        write(STAGE/'PROGRESS85_V1.json',dict(status='SOURCE_CAPACITY_QUALIFICATION',pid=os.getpid(),family='brusselator',seed=seed,closed_cases=len(cases),utc=utc()),replace=True)
        a,data,r=old.source('brusselator',seed);sources.append(r)
        prior=np.load(Path(CONFIG['data_root'])/'cycle85'/'anchored_screen_v1'/f'brusselator_{seed}'/'source_trainval_v1.npz',allow_pickle=False)
        assert all(np.array_equal(data[k],prior[k]) for k in prior.files)
        if not r['qualified']:continue
        for change in SHIFT_CASES:
            write(STAGE/'PROGRESS85_V1.json',dict(status='TARGET_FITTING',pid=os.getpid(),family='brusselator',seed=seed,case=change,closed_cases=len(cases),utc=utc()),replace=True)
            cases.append(old.evaluate_case(a,data,r,'brusselator',seed,change))
    result=dict(utc=utc(),status='CAPACITY_SCREEN_COMPLETE_NOT_FINAL_RESEARCH',sources=sources,qualified_sources=sum(z['qualified'] for z in sources),closed_cases=len(cases),cases=[{k:v for k,v in z.items() if k!='rows'} for z in cases],qualified_research_directions=0,new_method_tested=False,amendment_sha256=sha(C/'CAPACITY_AMENDMENT85_V1.md'),source_data_equal_to_original=True,architecture=dict(hidden_dim=128,latent_dim=4,n_layers=2))
    write(STAGE/'RESULT85_V1.json',result);write(STAGE/'PROGRESS85_V1.json',dict(status=result['status'],pid=os.getpid(),closed_cases=len(cases),qualified_sources=result['qualified_sources'],utc=utc()),replace=True)
    log(dict(kind='capacity_campaign_closed',cases=len(cases),qualified_sources=result['qualified_sources']))

if __name__=='__main__':main()
