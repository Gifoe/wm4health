"""Independent-cohort causal-state target development; no oracle labels in world fitting."""
import runtime_profile
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,hashlib,json,pathlib,sys,time
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1];PREVIOUS=ROOT.parent/'cycle20';sys.path.insert(0,str(PREVIOUS/'scripts'))
import innovation_pilot as p
OUT=ROOT/'outputs/response_target_prototype'
FAMILIES=['coupled_linear','rotating','saturating','queueing']
METHODS=['joint_budget','obs_budget','reconstruction','forecast_bank','response_only','projected_bank','random_projection','wide_budget','mop_budget','projected_mop']
ACTION_BANK=np.asarray([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype='float32')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def update(state,gain,action,family):
    if family!='queueing':return p.physical_step(state,gain,action,family)
    q,v=np.asarray(state).T;u,w=np.asarray(action).T
    return np.stack([np.maximum(0,.8*q+.7*gain-(.8+.4*u)),np.maximum(0,.75*v+.2*q+.45*gain-(.5+.3*w))],axis=-1)
def generate(family,seed,n):
    if family!='queueing':return p.generate(family,seed,n)
    rng=np.random.default_rng(seed);state=np.abs(rng.normal(size=(n,2)));gain=rng.choice([.4,1.6],size=n);noise=rng.normal(size=(n,9));xs=[];ys=[];actions=[]
    for t in range(128):
        u=rng.choice([-1.,1.],size=(n,2));x=np.column_stack([state,gain,noise]);state=update(state,gain,u,family)
        gain=rng.choice([.4,1.6],size=n);noise=.98*noise+np.sqrt(1-.98**2)*rng.normal(size=noise.shape)
        if t>=64:xs.append(x);ys.append(np.column_stack([state,gain,noise]));actions.append(u)
    def pack(a):return np.stack(a,axis=1).reshape(-1,np.asarray(a[0]).shape[-1]).astype('float32')
    return dict(current=pack(xs),future=pack(ys),action=pack(actions),trajectory_id=np.repeat(np.arange(n),64))
class Mop(nn.Module):
    def __init__(self):
        super().__init__();self.dimension=4;self.encoder=p.mlp(12,4);self.predictor=p.mlp(6,8);self.inverse=p.mlp(8,2);self.router=p.mlp(6,2)
    def predict(self,z,u):return nn.functional.normalize(self.predictor(torch.cat([z,u],1)).reshape(-1,2,4),dim=-1)
def make(method):
    if method in ['mop_budget','projected_mop']:return Mop().to('cuda')
    template='obs_idm' if method=='obs_budget' else ('joint_idm_full' if method=='wide_budget' else 'joint_idm')
    return p.Model(template).to('cuda')
def encode(model,x):
    raw=model.encoder(x);return nn.functional.normalize(raw,dim=-1) if isinstance(model,Mop) else raw
def base_loss(model,x,y,u,ema):
    if not isinstance(model,Mop):return p.losses(model,x,y,u)
    raw=model.encoder(x);raw_next=model.encoder(y);z=nn.functional.normalize(raw,dim=-1);zn=nn.functional.normalize(raw_next,dim=-1)
    target=nn.functional.normalize(ema(y),dim=-1);heads=model.predict(z,u);dist=1-(heads*target[:,None,:]).sum(-1);winner=dist.argmin(1)
    forward=dist.gather(1,winner[:,None]).mean();route=nn.functional.cross_entropy(model.router(torch.cat([z,u],1)),winner)
    usage=nn.functional.one_hot(winner,2).float().mean(0);balance=(usage*torch.log((2*usage).clamp(min=1e-8))).sum()
    inverse=(model.inverse(torch.cat([z,zn],1))-u).square().mean();anti=(p.regularizer(raw)+p.regularizer(raw_next))/2
    total=forward+inverse+anti+.1*route+.01*balance
    return total,dict(prediction=forward,inverse=inverse,anti_collapse=anti,router_CE=route,hard_usage_KL_value=balance)
def native_physical(model,x,u,decoder,which,mean,scale,rotation):
    if not isinstance(model,Mop):return p.model_physical(model,x,u,decoder,which,mean,scale,rotation)
    z=encode(model,x);heads=model.predict(z,u);weights=torch.softmax(model.router(torch.cat([z,u],1)),1)
    values=torch.stack([p.probe_predict(decoder,heads[:,j],which)[:,:2] for j in range(2)],1)
    return (values*weights[:,:,None]).sum(1)*scale[:2]+mean[:2]
def forecast_bank(teacher,x):
    action=torch.tensor(ACTION_BANK,device='cuda');n=len(x)
    with torch.no_grad():return teacher.predict(x[:,None,:].expand(-1,4,-1).reshape(-1,12),action[None,:,:].expand(n,-1,-1).reshape(-1,2)).reshape(n,4,12)
def targets(teacher,train,val,seed):
    tb=forecast_bank(teacher,train['x']);vb=forecast_bank(teacher,val['x']);r=tb-tb.mean(1,keepdim=True)
    cov=(r.reshape(-1,12).double().T@r.reshape(-1,12).double()/r.shape[0]/4).cpu().numpy()
    eig,basis=np.linalg.eigh(cov);order=np.argsort(eig)[::-1];eig=np.maximum(eig[order],0);basis=basis[:,order]
    rank=int(np.searchsorted(np.cumsum(eig)/eig.sum(),.95)+1) if eig.sum()>1e-12 else 0
    projection=basis[:,:rank]@basis[:,:rank].T
    random_basis,_=np.linalg.qr(np.random.default_rng(20262160+seed).normal(size=(12,max(rank,1))))
    random_projection=random_basis[:,:rank]@random_basis[:,:rank].T
    proj=torch.tensor(projection,dtype=torch.float32,device='cuda');rand=torch.tensor(random_projection,dtype=torch.float32,device='cuda')
    cases={
        'reconstruction':(train['x'][:,None,:].expand(-1,4,-1),val['x'][:,None,:].expand(-1,4,-1)),
        'forecast_bank':(tb,vb),'response_only':(r,vb-vb.mean(1,keepdim=True)),
        'projected_bank':(tb@proj,vb@proj),'random_projection':(tb@rand,vb@rand)}
    normalized={};stats={}
    for name,(a,b) in cases.items():
        a=a.reshape(-1,48);b=b.reshape(-1,48);mean=a.mean(0);scale=torch.sqrt((a-mean).square().mean()).clamp(min=1e-6)
        normalized[name]=((a-mean)/scale,(b-mean)/scale);stats[name]=dict(mean=mean.cpu(),scale=scale.cpu())
    meta=dict(action_response_eigenvalues=eig.tolist(),selected_rank=rank,projection=projection.tolist(),random_projection=random_projection.tolist())
    return normalized,stats,meta
def train_student(method,seed,train,val,target=None):
    torch.manual_seed(seed);model=make(method);aux=p.mlp(model.dimension,48).to('cuda') if target is not None else None
    parameters=list(model.parameters())+(list(aux.parameters()) if aux is not None else [])
    optimizer=torch.optim.Adam(parameters,lr=.001);rng=torch.Generator(device='cuda').manual_seed(seed+123000)
    ema=copy.deepcopy(model.encoder).eval() if isinstance(model,Mop) else None
    if ema is not None:
        for parameter in ema.parameters():parameter.requires_grad_(False)
    steps=3000 if method in ['joint_budget','obs_budget','reconstruction','wide_budget','mop_budget'] else 1500
    history=[];tick=time.perf_counter()
    for step in range(1,steps+1):
        idx=torch.randint(len(train['x']),(256,),generator=rng,device='cuda');loss,parts=base_loss(model,train['x'][idx],train['y'][idx],train['u'][idx],ema)
        auxiliary=loss*0 if aux is None else (aux(encode(model,train['x'][idx]))-target[0][idx]).square().mean();loss=loss+auxiliary
        assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(parameters,10);optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.encoder.parameters()):a.mul_(.99).add_(b,alpha=.01)
        if step%300==0:
            with torch.no_grad():
                validation,vparts=base_loss(model,val['x'][:2048],val['y'][:2048],val['u'][:2048],ema)
                va=validation*0 if aux is None else (aux(encode(model,val['x'][:2048]))-target[1][:2048]).square().mean()
            history.append(dict(step=step,training_total=float(loss.detach()),validation_total=float(validation+va),auxiliary=float(auxiliary.detach()),
                validation_auxiliary=float(va),gradient_norm=float(norm),components={k:float(v.detach()) for k,v in parts.items()}))
    return model,aux,history,time.perf_counter()-tick
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available();assert sha(PREVIOUS/'scripts/innovation_pilot.py')=='ae0d40989255c30af55fd689ccb72336f04570bfb86a6bbdb9c8ce49a2a038b7'
    assert json.loads((PREVIOUS/'outputs/innovation_pilot/independent_verification.json').read_text())['pass']
    assert not OUT.exists();OUT.mkdir(parents=True)
    for name in ['data','checkpoints','teachers','predictions']:(OUT/name).mkdir()
    tick=time.perf_counter();config=dict(families=FAMILIES,methods=METHODS,seeds=[2021,2022,2023],plan_sha256=sha(ROOT/'PROTOTYPE_PLAN.md'),
        script_sha256=sha(pathlib.Path(__file__)),dependency_sha256=sha(PREVIOUS/'scripts/innovation_pilot.py'))
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8');registry=[];cache={};fits=[];teacher_rows=[]
    for fi,family in enumerate(FAMILIES):
        splits={name:generate(family,20262111+1000*fi+i,n) for i,(name,n) in enumerate([('train',192),('val',64),('test',128)])}
        mean=splits['train']['current'].mean(0);scale=splits['train']['current'].std(0,ddof=1);assert np.all(scale>1e-6)
        rotation,_=np.linalg.qr(np.random.default_rng(20262120+fi).normal(size=(12,12)));rotation=rotation.astype('float32');files={}
        for name,data in splits.items():
            file=OUT/'data'/(family+'_'+name+'.npz');np.savez_compressed(file,**data,mean=mean,scale=scale,rotation=rotation);files[name]=dict(file=file.name,sha256=sha(file))
        registry.append(dict(family=family,files=files));cache[family]=(splits['train'],splits['val'],mean,scale,rotation)
    (OUT/'dataset_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    for family in FAMILIES:
        raw_train,raw_val,mean,scale,rotation=cache[family];train=p.to_gpu(p.prep(raw_train,mean,scale,rotation));val=p.to_gpu(p.prep(raw_val,mean,scale,rotation))
        tm=torch.tensor(mean,device='cuda');ts=torch.tensor(scale,device='cuda');tr=torch.tensor(rotation,device='cuda')
        for seed in [2021,2022,2023]:
            teacher,teacher_history,teacher_seconds=p.train_model('fullobs',seed+5000,train,val)
            with torch.no_grad():
                forecast=p.model_physical(teacher,val['x'],val['u'],None,'linear',tm,ts,tr)
                teacher_error=float(((forecast-torch.tensor(raw_val['future'][:,:2],device='cuda'))/ts[:2]).square().mean())
            targs,targ_stats,projection_meta=targets(teacher,train,val,seed)
            teacher_file=OUT/'teachers'/(family+'_'+str(seed)+'.pt');torch.save(dict(state={k:v.cpu() for k,v in teacher.state_dict().items()},normalization=targ_stats),teacher_file)
            teacher_rows.append(dict(family=family,seed=seed,seconds=teacher_seconds,validation_physical_MSE=teacher_error,
                checkpoint=teacher_file.name,checkpoint_sha256=sha(teacher_file),history=teacher_history,**projection_meta))
            (OUT/'teacher_records.json').write_text(json.dumps(teacher_rows,indent=2),encoding='utf-8')
            del teacher
            for method in METHODS:
                start=time.perf_counter();target_name='projected_bank' if method=='projected_mop' else method
                target=targs.get(target_name);model,aux,history,seconds=train_student(method,seed,train,val,target)
                with torch.no_grad():z=encode(model,train['x'][:8192]);vz=encode(model,val['x'])
                decoder=p.probe_train(z,train['current_labels'][:8192],vz,val['current_labels'],seed+70000)
                forward=p.probe_train(torch.cat([z,train['u'][:8192]],1),train['future_labels'][:8192],torch.cat([vz,val['u']],1),val['future_labels'],seed+80000)
                dc=p.choice(decoder['scores'],[0,1]);gc=p.choice(decoder['scores'],[2]);fc=p.choice(forward['scores'],[0,1])
                with torch.no_grad():
                    ev=torch.linalg.eigvalsh(torch.cov(z.T)).clamp(min=1e-12);prob=ev/ev.sum();rank=float(torch.exp(-(prob*torch.log(prob)).sum()))
                    forecast=native_physical(model,val['x'],val['u'],decoder,dc,tm,ts,tr)
                    val_error=float(((forecast-torch.tensor(raw_val['future'][:,:2],device='cuda'))/ts[:2]).square().mean())
                key=family+'_'+method+'_'+str(seed);file=OUT/'checkpoints'/(key+'.pt')
                torch.save(dict(world_model={k:v.cpu() for k,v in model.state_dict().items()},auxiliary=None if aux is None else {k:v.cpu() for k,v in aux.state_dict().items()},
                    decoder=p.probe_state(decoder),future_probe=p.probe_state(forward),decoder_choice=dc,gain_choice=gc,future_choice=fc,method=method),file)
                row=dict(key=key,family=family,method=method,seed=seed,checkpoint=file.name,checkpoint_sha256=sha(file),student_seconds=seconds,
                    total_fit_probe_seconds=time.perf_counter()-start,history=history,validation_physical_MSE=val_error,encoder_effective_rank=rank,
                    parameter_count=sum(v.numel() for v in model.parameters()),auxiliary_parameter_count=0 if aux is None else sum(v.numel() for v in aux.parameters()),
                    teacher_seconds=teacher_seconds if target_name in ['forecast_bank','response_only','projected_bank','random_projection'] else 0,
                    decoder_validation_MSE=decoder['scores'].cpu().tolist(),successor_probe_validation_MSE=forward['scores'].cpu().tolist())
                fits.append(row);(OUT/'fit_records.json').write_text(json.dumps(fits,indent=2),encoding='utf-8')
                print(json.dumps(dict(phase='fit',completed=len(fits),total=120,key=key,seconds=row['total_fit_probe_seconds'],validation_physical_MSE=val_error)),flush=True)
                del model,aux,decoder,forward,z,vz
    assert len(fits)==120;(OUT/'training_complete.json').write_text(json.dumps(dict(fits=120,test_metrics_accessed=False,seconds=time.perf_counter()-tick)),encoding='utf-8')
    rows=[]
    for family in FAMILIES:
        data=np.load(OUT/'data'/(family+'_test.npz'),allow_pickle=False);raw={k:data[k] for k in ['current','future','action','trajectory_id']}
        mean=data['mean'];scale=data['scale'];rotation=data['rotation'];test=p.to_gpu(p.prep(raw,mean,scale,rotation))
        tm=torch.tensor(mean,device='cuda');ts=torch.tensor(scale,device='cuda');tr=torch.tensor(rotation,device='cuda')
        low=raw['current'].copy();high=low.copy();low[:,2]=.4;high[:,2]=1.6
        lx=torch.tensor((low-mean)/scale@rotation.T,device='cuda');hx=torch.tensor((high-mean)/scale@rotation.T,device='cuda')
        lo_truth=update(low[:,:2],low[:,2],raw['action'],family);hi_truth=update(high[:,:2],high[:,2],raw['action'],family)
        true_mod=(hi_truth-lo_truth)/scale[:2];gain_blind=(hi_truth+lo_truth)/2
        blind_MSE=float(np.mean(((gain_blind-raw['future'][:,:2])/scale[:2])**2))
        for fit in [r for r in fits if r['family']==family]:
            state=torch.load(OUT/'checkpoints'/fit['checkpoint'],map_location='cpu',weights_only=True);model=make(fit['method']);model.load_state_dict(state['world_model']);model.eval()
            decoder=p.restore_probe(state['decoder']);forward=p.restore_probe(state['future_probe'])
            with torch.no_grad():
                z=encode(model,test['x']);zn=encode(model,test['y']);current=p.probe_predict(decoder,z,state['decoder_choice']);gp=p.probe_predict(decoder,z,state['gain_choice'])[:,2]
                native=(native_physical(model,test['x'],test['u'],decoder,state['decoder_choice'],tm,ts,tr)-tm[:2])/ts[:2]
                oracle=p.probe_predict(decoder,zn,state['decoder_choice'])[:,:2];future=p.probe_predict(forward,torch.cat([z,test['u']],1),state['future_choice'])
                nlo=native_physical(model,lx,test['u'],decoder,state['decoder_choice'],tm,ts,tr);nhi=native_physical(model,hx,test['u'],decoder,state['decoder_choice'],tm,ts,tr)
                flo=p.probe_predict(forward,torch.cat([encode(model,lx),test['u']],1),state['future_choice']);fhi=p.probe_predict(forward,torch.cat([encode(model,hx),test['u']],1),state['future_choice'])
                nmod=((nhi-nlo)/ts[:2]).cpu().numpy();fmod=(fhi-flo).cpu().numpy();truth=test['future_labels'].cpu().numpy()
                gain_accuracy=float(((gp*ts[2]+tm[2]>1)==(test['current_labels'][:,2]*ts[2]+tm[2]>1)).float().mean())
                inv=model.inverse(torch.cat([z,zn],1));inv_accuracy=float(((inv>0)==(test['u']>0)).all(1).float().mean())
            def relative(a):return float(np.sqrt(np.mean((a-true_mod)**2)/np.mean(true_mod**2)))
            file=OUT/'predictions'/(fit['key']+'.npz')
            np.savez_compressed(file,current_probe=current.cpu().numpy(),gain_prediction=gp.cpu().numpy(),native_physical_prediction=native.cpu().numpy(),
                frozen_encoder_successor_probe=future.cpu().numpy(),oracle_successor_decoding=oracle.cpu().numpy(),true_successor=truth,
                true_modulation=true_mod,native_modulation=nmod,probe_modulation=fmod,trajectory_id=raw['trajectory_id'])
            row={k:fit[k] for k in ['key','family','method','seed','encoder_effective_rank','parameter_count','auxiliary_parameter_count']}
            row.update(gain_accuracy=gain_accuracy,inverse_joint_action_accuracy=inv_accuracy,
                native_physical_MSE=float((native-test['future_labels']).square().mean()),frozen_encoder_successor_probe_MSE=float((future-test['future_labels']).square().mean()),
                oracle_successor_decoding_MSE=float((oracle-test['future_labels']).square().mean()),
                current_physical_decoding_MSE=float((current[:,:2]-test['current_labels'][:,:2]).square().mean()),
                native_modulation_relative_RMSE=relative(nmod),frozen_encoder_modulation_relative_RMSE=relative(fmod),gain_blind_oracle_physical_MSE=blind_MSE,
                prediction_file=file.name,prediction_sha256=sha(file));rows.append(row)
            print(json.dumps(dict(phase='test',completed=len(rows),key=row['key'],gain_accuracy=gain_accuracy,response_error=row['native_modulation_relative_RMSE'])),flush=True)
    readiness={f:all(r['validation_physical_MSE']<.05 for r in teacher_rows if r['family']==f) for f in FAMILIES}
    result=dict(rows=rows,teacher_validation_readiness=readiness,elapsed_seconds=time.perf_counter()-tick,config=config,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),maximum_allocated_bytes=torch.cuda.max_memory_allocated()),
        accepted_novel_directions=0,objective_achieved=False,scope='Independent-cohort synthetic development; task adaptations, not source-native benchmark reproductions. No public release.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(completed=120,seconds=result['elapsed_seconds'],readiness=readiness)),flush=True)
if __name__=='__main__':main()
