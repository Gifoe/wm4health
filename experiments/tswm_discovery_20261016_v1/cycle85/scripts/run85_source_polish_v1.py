import common85_v1 as common
import run85_v1 as old
from common85_v1 import *
import time

STAGE=C/'outputs'/'source_polish_v3';STAGE.mkdir(parents=True,exist_ok=True)
old.O=STAGE;common.O=STAGE
old.RAW=Path(CONFIG['artifact_root'])/'cycle85'/'anchored_screen_source_polish_v3'
old.DATA=Path(CONFIG['data_root'])/'cycle85'/'anchored_screen_source_polish_v3'
old.WEIGHTS=Path(CONFIG['checkpoint_root'])/'cycle85'/'anchored_screen_source_polish_v3'
def make(seed,nu):
    torch.manual_seed(seed)
    return old.BC.BCObsAgent(dict(obs_dim=2,robot_obs_dim=2,obj_obs_dim=0,lat_obs_dim=4),dict(act_dim=nu+1),'cpu',n_layers=2,hidden_dim=128,lr=3e-4)
old.new_agent=make

def extended_source(family,seed):
    assert family=='brusselator';tag=f'{family}_{seed}'
    parent_path=C/'outputs'/'capacity_v2'/f'SOURCE_READY85_{tag}_V1.json';parent=json.loads(parent_path.read_text());assert not parent['qualified']
    assert sha(parent['checkpoint'])==parent['checkpoint_sha256']
    olddata=np.load(Path(CONFIG['data_root'])/'cycle85'/'anchored_screen_capacity_v2'/tag/'source_trainval_v1.npz',allow_pickle=False)
    data={k:olddata[k] for k in olddata.files};d=old.DATA/tag;d.mkdir(parents=True,exist_ok=True);old.save_npz(d/'source_trainval_v1.npz',**data)
    a=make(seed,2);unpack(a,torch.load(parent['checkpoint'],map_location='cpu',weights_only=True));params=trainable(a,['obs_enc','obs_dec','fwd_dyn','inv_dyn']);opt=torch.optim.Adam(params,lr=1e-4)
    rng=np.random.default_rng(seed+12000);best=float('inf');state=None;selected=None;checks=[];qualified=False;start=time.perf_counter()
    for updates in range(1,8001):
        y,u=batch_windows(data['y'],data['u'],rng,32,h=32);opt.zero_grad();oe,recon=output_loss(a,y,u)
        z=a.obs_enc(y[:,0]);zn=a.obs_enc(y[:,1]);inv=F.mse_loss(a.inv_dyn(torch.cat([z,zn],dim=-1)),u[:,0]);loss=oe+.1*recon+.001*inv;loss.backward();opt.step()
        if updates not in [2000,4000,8000]:continue
        scores=old.val_scores(a,data['val_y'],data['val_u']);info=dict(additional_polish_updates=updates,ancestor_selected_polish_updates=parent['selected']['polish_updates'],loss=float(loss.detach()),**scores);checks.append(info)
        log(dict(kind='extended_source_validation',family=family,seed=seed,**info))
        if scores['mean32']<best:best=scores['mean32'];state=pack(a);selected=info
        if scores['median32']<=.15 and scores['median128']<=.35:qualified=True;state=pack(a);selected=info;break
    assert state is not None;unpack(a,state);freeze(a);path=old.WEIGHTS/tag/'source_selected_v1.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save(state,path)
    record=dict(utc=utc(),family=family,seed=seed,nu=2,qualified=qualified,checkpoint=str(path),checkpoint_sha256=sha(path),selected=selected,validation_checks=checks,source_data_sha256=sha(d/'source_trainval_v1.npz'),trained_parameters=sum(p.numel() for n in ['obs_enc','obs_dec','fwd_dyn','inv_dyn'] for p in getattr(a,n).parameters()),wall_seconds=time.perf_counter()-start,ancestor_checkpoint=parent['checkpoint'],ancestor_checkpoint_sha256=parent['checkpoint_sha256'],ancestor_record_sha256=sha(parent_path),ancestor_completed_polish_updates=max(q['polish_updates'] for q in parent['validation_checks']),additional_completed_updates=updates,source_objective_horizon=32)
    write(STAGE/f'SOURCE_READY85_{tag}_V1.json',record);log(dict(kind='extended_source_closed',family=family,seed=seed,qualified=qualified,selected=selected));return a,data,record
old.source=extended_source

def main():
    cap=json.loads((C/'outputs'/'capacity_v2'/'RESULT85_V1.json').read_text());assert cap['closed_cases']==0 and cap['qualified_sources']==0
    assert not list((C/'outputs').glob('CASE85_brusselator*_V1.json')) and not list((C/'outputs'/'capacity_v2').glob('CASE85_brusselator*_V1.json'))
    old.RAW.mkdir(parents=True,exist_ok=True);sources=[];cases=[]
    log(dict(kind='extended_source_campaign_start',pid=os.getpid(),extension_sha256=sha(C/'SOURCE_POLISH_EXTENSION85_V1.md')))
    for seed in SEEDS:
        write(STAGE/'PROGRESS85_V1.json',dict(status='EXTENDED_SOURCE_TRAINING',pid=os.getpid(),family='brusselator',seed=seed,closed_cases=len(cases),utc=utc()),replace=True)
        a,data,r=extended_source('brusselator',seed);sources.append(r)
        if not r['qualified']:continue
        for change in SHIFT_CASES:
            write(STAGE/'PROGRESS85_V1.json',dict(status='TARGET_FITTING',pid=os.getpid(),family='brusselator',seed=seed,case=change,closed_cases=len(cases),utc=utc()),replace=True)
            cases.append(old.evaluate_case(a,data,r,'brusselator',seed,change))
    result=dict(utc=utc(),status='SOURCE_POLISH_CONTROL_COMPLETE_NOT_FINAL_RESEARCH',sources=sources,qualified_sources=sum(z['qualified'] for z in sources),closed_cases=len(cases),cases=[{k:v for k,v in z.items() if k!='rows'} for z in cases],qualified_research_directions=0,new_method_tested=False,extension_sha256=sha(C/'SOURCE_POLISH_EXTENSION85_V1.md'),architecture=dict(hidden_dim=128,latent_dim=4,n_layers=2),source_objective_horizon=32)
    write(STAGE/'RESULT85_V1.json',result);write(STAGE/'PROGRESS85_V1.json',dict(status=result['status'],pid=os.getpid(),closed_cases=len(cases),qualified_sources=result['qualified_sources'],utc=utc()),replace=True);log(dict(kind='extended_source_campaign_closed',cases=len(cases),qualified_sources=result['qualified_sources']))

if __name__=='__main__':main()
