"""Controlled task transfer: observed incoming innovations versus self-targeted representations."""
import runtime_profile
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,hashlib,json,pathlib,sys,time
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/innovation_pilot'
FAMILIES=['coupled_linear','rotating','saturating'];METHODS=['joint','joint_idm','ema_idm','gaussian_idm','mixture_idm','obs_idm','fullobs','joint_idm_full']
STEPS=1500;BATCH=256;DEVICE='cuda'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def physical_step(state,gain,action,family):
    q,p=np.asarray(state).T;u,v=np.asarray(action).T
    if family=='coupled_linear':return np.stack([.7*q+.15*p+.35*gain*u,.6*p-.1*q+.3*v],axis=-1)
    if family=='rotating':return np.stack([.9*q+.25*p+.3*u,-.25*q+.9*p+.3*gain*v],axis=-1)
    return np.stack([.7*np.tanh(q)+.15*p+.4*gain*np.tanh(u),.5*p-.2*np.tanh(q)+.25*v],axis=-1)
def generate(family,seed,n_trajectories):
    rng=np.random.default_rng(seed);state=rng.normal(size=(n_trajectories,2));gain=rng.choice([.4,1.6],size=n_trajectories)
    nuisance=rng.normal(size=(n_trajectories,9));xs=[];ys=[];us=[]
    for t in range(128):
        action=rng.choice([-1.,1.],size=(n_trajectories,2));current=np.column_stack([state,gain,nuisance])
        state=physical_step(state,gain,action,family);gain=rng.choice([.4,1.6],size=n_trajectories)
        nuisance=.98*nuisance+np.sqrt(1-.98**2)*rng.normal(size=nuisance.shape)
        future=np.column_stack([state,gain,nuisance])
        if t>=64:xs.append(current);ys.append(future);us.append(action)
    # Trajectories are contiguous and can be block-resampled without mixing trajectory identities.
    def pack(a):return np.stack(a,axis=1).reshape(-1,np.asarray(a[0]).shape[-1]).astype('float32')
    return dict(current=pack(xs),future=pack(ys),action=pack(us),trajectory_id=np.repeat(np.arange(n_trajectories),64))
def prep(split,mean,scale,rotation):
    return dict(x=((split['current']-mean)/scale@rotation.T).astype('float32'),
        y=((split['future']-mean)/scale@rotation.T).astype('float32'),u=split['action'],
        current_labels=((split['current'][:,:3]-mean[:3])/scale[:3]).astype('float32'),
        future_labels=((split['future'][:,:2]-mean[:2])/scale[:2]).astype('float32'))
def numerical_checks():
    a=generate('rotating',20262001,17);q,_=np.linalg.qr(np.random.default_rng(711).normal(size=(12,12)))
    mean=a['current'].mean(0);scale=a['current'].std(0);x=(a['current']-mean)/scale@q.T
    error=float(np.max(np.abs(x@q*scale+mean-a['current'])));assert error<1e-5
    errors={}
    for family in FAMILIES:
        current=generate(family,20262002,19);s=current['current'][:,:2];u=current['action']
        lo=physical_step(s,np.full(len(s),.4),u,family);hi=physical_step(s,np.full(len(s),1.6),u,family)
        expected=np.zeros_like(lo)
        if family=='rotating':expected[:,1]=.36*u[:,1]
        else:expected[:,0]=(1.2*.35*u[:,0] if family=='coupled_linear' else 1.2*.4*np.tanh(u[:,0]))
        e=float(np.max(np.abs(hi-lo-expected)));assert e<1e-6;errors[family]=e
        assert set(np.unique(current['current'][:,2]).round(1))==set(np.asarray([.4,1.6],dtype='float32'))
    return dict(observation_inverse_max_error=error,gain_modulation_equation_errors=errors)
def mlp(n_in,n_out):return nn.Sequential(nn.Linear(n_in,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,n_out))
class Model(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method;self.dimension=12 if method in ['fullobs','joint_idm_full'] else 4
        self.encoder=nn.Identity() if method=='fullobs' else mlp(12,self.dimension)
        self.obs=method in ['obs_idm','fullobs'];self.mixture=method=='mixture_idm';self.gaussian=method=='gaussian_idm'
        self.inverse=mlp(2*self.dimension,2) if 'idm' in method else None
        nout=12 if self.obs else self.dimension*(2 if self.mixture or self.gaussian else 1)
        self.predictor=mlp(self.dimension+2,nout)
    def predict(self,z,u):
        out=self.predictor(torch.cat([z,u],dim=-1))
        if self.mixture:return out.reshape(-1,2,self.dimension)
        return out
def regularizer(z):
    center=z-z.mean(0);cov=center.T@center/(len(z)-1);std=torch.sqrt(torch.diag(cov)+1e-4)
    off=cov-torch.diag(torch.diag(cov))
    return 10*torch.relu(1-std).mean()+off.square().sum()/z.shape[1]+.01*z.mean(0).square().mean()
def losses(model,x,y,u,ema=None):
    z=model.encoder(x);zn=model.encoder(y);target=ema(y) if ema is not None else zn
    pred=model.predict(z,u)
    if model.obs:prediction=(pred-y).square().mean()
    elif model.mixture:prediction=(pred-target[:,None,:]).square().mean(-1).min(1).values.mean()
    elif model.gaussian:
        mu,logvar=pred.chunk(2,dim=-1);logvar=logvar.clamp(-6,4)
        prediction=.5*((target-mu).square()*torch.exp(-logvar)+logvar).mean()
    else:prediction=(pred-target).square().mean()
    inverse=(model.inverse(torch.cat([z,zn],dim=-1))-u).square().mean() if model.inverse is not None else prediction*0
    anti=(regularizer(z)+regularizer(zn))/2 if model.method!='fullobs' else prediction*0
    return prediction+inverse+anti,dict(prediction=prediction,inverse=inverse,anti_collapse=anti)
def to_gpu(data):return {k:torch.tensor(v,device=DEVICE) for k,v in data.items()}
def train_model(method,seed,train,val):
    torch.manual_seed(seed);model=Model(method).to(DEVICE);ema=None
    if method=='ema_idm':
        ema=copy.deepcopy(model.encoder).eval()
        for p in ema.parameters():p.requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);rng=torch.Generator(device=DEVICE).manual_seed(seed+123000)
    history=[];tic=time.perf_counter()
    for step in range(1,STEPS+1):
        idx=torch.randint(len(train['x']),(BATCH,),generator=rng,device=DEVICE)
        loss,parts=losses(model,train['x'][idx],train['y'][idx],train['u'][idx],ema)
        assert torch.isfinite(loss),method
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for a,b in zip(ema.parameters(),model.encoder.parameters()):a.mul_(.99).add_(b,alpha=.01)
        if step%150==0:
            with torch.no_grad():v,vparts=losses(model,val['x'][:2048],val['y'][:2048],val['u'][:2048],ema)
            history.append(dict(step=step,training_total=float(loss),validation_total=float(v),gradient_norm=float(norm),
                training_components={k:float(v) for k,v in parts.items()},validation_components={k:float(v) for k,v in vparts.items()}))
    return model,history,time.perf_counter()-tic
def probe_train(a,b,va,vb,seed):
    mean=a.mean(0);scale=a.std(0).clamp(min=1e-6);aa=(a-mean)/scale;vv=(va-mean)/scale
    design=torch.cat([aa,torch.ones((len(aa),1),device=DEVICE)],1).double()
    penalty=torch.eye(design.shape[1],device=DEVICE,dtype=torch.float64)*1e-5;penalty[-1,-1]=0
    coefficients=torch.linalg.solve(design.T@design/len(design)+penalty,design.T@b.double()/len(design)).float()
    torch.manual_seed(seed);net=mlp(a.shape[1],b.shape[1]).to(DEVICE);opt=torch.optim.Adam(net.parameters(),lr=2e-3)
    rng=torch.Generator(device=DEVICE).manual_seed(seed+444)
    for _ in range(500):
        idx=torch.randint(len(a),(BATCH,),generator=rng,device=DEVICE);loss=(net(aa[idx])-b[idx]).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    with torch.no_grad():
        linear=torch.cat([vv,torch.ones((len(vv),1),device=DEVICE)],1)@coefficients
        nonlinear=net(vv);scores=torch.stack([(linear-vb).square().mean(0),(nonlinear-vb).square().mean(0)])
    return dict(mean=mean,scale=scale,coefficients=coefficients,network=net,scores=scores)
def probe_predict(probe,a,choice):
    aa=(a-probe['mean'])/probe['scale']
    if choice=='linear':return torch.cat([aa,torch.ones((len(aa),1),device=DEVICE)],1)@probe['coefficients']
    return probe['network'](aa)
def probe_state(probe):
    return dict(mean=probe['mean'].cpu(),scale=probe['scale'].cpu(),coefficients=probe['coefficients'].cpu(),
                network={k:v.cpu() for k,v in probe['network'].state_dict().items()},scores=probe['scores'].cpu())
def restore_probe(state):
    net=mlp(len(state['mean']),state['coefficients'].shape[1]).to(DEVICE);net.load_state_dict(state['network']);net.eval()
    return {k:v.to(DEVICE) for k,v in state.items() if k!='network'}|dict(network=net)
def choice(scores,columns):return ['linear','nonlinear'][int(scores[:,columns].mean(1).argmin())]
def full_physical(x,mean,scale,rotation):
    # Evaluation-only inverse observation transform; model/probe fitting never sees it.
    return x@rotation*scale+mean
def model_physical(model,x,u,decoder,decoder_choice,mean,scale,rotation):
    z=model.encoder(x);pred=model.predict(z,u)
    if model.obs:return full_physical(pred,mean,scale,rotation)[:,:2]
    if model.gaussian:pred=pred[:,:model.dimension]
    if model.mixture:
        normalized=torch.stack([probe_predict(decoder,pred[:,j],decoder_choice)[:,:2] for j in range(2)]).mean(0)
    else:normalized=probe_predict(decoder,pred,decoder_choice)[:,:2]
    return normalized*scale[:2]+mean[:2]
def main():
    start=time.perf_counter();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;assert not OUT.exists();OUT.mkdir(parents=True)
    (OUT/'checkpoints').mkdir();(OUT/'data').mkdir();(OUT/'predictions').mkdir();checks=numerical_checks()
    config=dict(families=FAMILIES,methods=METHODS,seeds=[2021,2022,2023],steps=STEPS,batch=BATCH,lr=.001,probe_steps=500,
        plan_sha256=sha(ROOT/'PILOT_PLAN.md'),script_sha256=sha(pathlib.Path(__file__)),numerical_checks=checks)
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    registry=[];data_cache={};fit_rows=[]
    # Generate split files before fitting; no test metrics or test tensors are used during fitting.
    for fi,family in enumerate(FAMILIES):
        splits={name:generate(family,20262011+1000*fi+i,n) for i,(name,n) in enumerate([('train',192),('val',64),('test',128)])}
        mean=splits['train']['current'].mean(0);scale=splits['train']['current'].std(0,ddof=1)
        rotation,_=np.linalg.qr(np.random.default_rng(20262020+fi).normal(size=(12,12)));rotation=rotation.astype('float32')
        files={}
        for name,split in splits.items():
            f=OUT/'data'/(family+'_'+name+'.npz');np.savez_compressed(f,**split,mean=mean,scale=scale,rotation=rotation)
            files[name]=dict(file=f.name,sha256=sha(f),trajectories=len(np.unique(split['trajectory_id'])),transitions=len(split['action']))
            assert len(np.unique(split['current'][:,2]))==2 and len(np.unique(split['action'],axis=0))==4
        registry.append(dict(family=family,files=files));data_cache[family]=(splits['train'],splits['val'],mean,scale,rotation)
    (OUT/'dataset_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    for family in FAMILIES:
        train_raw,val_raw,mean,scale,rotation=data_cache[family];train=to_gpu(prep(train_raw,mean,scale,rotation));val=to_gpu(prep(val_raw,mean,scale,rotation))
        for method in METHODS:
            for seed in [2021,2022,2023]:
                tic=time.perf_counter();model,history,seconds=train_model(method,seed,train,val)
                with torch.no_grad():
                    z=model.encoder(train['x'][:8192]);vz=model.encoder(val['x'])
                decoder=probe_train(z,train['current_labels'][:8192],vz,val['current_labels'],seed+70000)
                future_probe=probe_train(torch.cat([z,train['u'][:8192]],1),train['future_labels'][:8192],
                    torch.cat([vz,val['u']],1),val['future_labels'],seed+80000)
                decoder_choice=choice(decoder['scores'],[0,1]);gain_choice=choice(decoder['scores'],[2]);future_choice=choice(future_probe['scores'],[0,1])
                with torch.no_grad():
                    covariance=torch.cov(z.T);eigs=torch.linalg.eigvalsh(covariance).clamp(min=1e-12);prob=eigs/eigs.sum()
                    effective_rank=float(torch.exp(-(prob*torch.log(prob)).sum()))
                    val_p=model_physical(model,val['x'],val['u'],decoder,decoder_choice,torch.tensor(mean,device=DEVICE),torch.tensor(scale,device=DEVICE),torch.tensor(rotation,device=DEVICE))
                    val_physical_MSE=float(((val_p-torch.tensor(val_raw['future'][:,:2],device=DEVICE))/torch.tensor(scale[:2],device=DEVICE)).square().mean())
                key=family+'_'+method+'_'+str(seed);checkpoint=OUT/'checkpoints'/(key+'.pt')
                state=dict(world_model={k:v.cpu() for k,v in model.state_dict().items()},decoder=probe_state(decoder),future_probe=probe_state(future_probe),
                    decoder_choice=decoder_choice,gain_choice=gain_choice,future_choice=future_choice,method=method,seed=seed)
                torch.save(state,checkpoint)
                row=dict(key=key,family=family,method=method,seed=seed,checkpoint=checkpoint.name,checkpoint_sha256=sha(checkpoint),
                    world_training_seconds=seconds,total_fit_and_probe_seconds=time.perf_counter()-tic,history=history,
                    validation_physical_MSE=val_physical_MSE,decoder_validation_MSE=decoder['scores'].cpu().tolist(),
                    successor_probe_validation_MSE=future_probe['scores'].cpu().tolist(),decoder_choice=decoder_choice,
                    gain_choice=gain_choice,successor_probe_choice=future_choice,encoder_effective_rank=effective_rank,
                    encoder_std_min=float(z.std(0).min()),parameter_count=sum(p.numel() for p in model.parameters()))
                fit_rows.append(row);(OUT/'fit_records.json').write_text(json.dumps(fit_rows,indent=2),encoding='utf-8')
                print(json.dumps(dict(phase='fit',completed=len(fit_rows),total=72,key=key,seconds=row['total_fit_and_probe_seconds'],val_physical_MSE=val_physical_MSE)),flush=True)
                del model,decoder,future_probe,z,vz
    assert len(fit_rows)==72
    (OUT/'training_complete.json').write_text(json.dumps(dict(fits=72,seconds=time.perf_counter()-start,test_metrics_accessed=False),indent=2),encoding='utf-8')
    # Test access begins here, after all world models and diagnostic choices are fixed.
    rows=[]
    for family in FAMILIES:
        file=OUT/'data'/(family+'_test.npz');saved=np.load(file,allow_pickle=False)
        mean=saved['mean'];scale=saved['scale'];rotation=saved['rotation'];raw={k:saved[k] for k in ['current','future','action','trajectory_id']}
        test=to_gpu(prep(raw,mean,scale,rotation));tm=torch.tensor(mean,device=DEVICE);ts=torch.tensor(scale,device=DEVICE);tr=torch.tensor(rotation,device=DEVICE)
        low=raw['current'].copy();high=low.copy();low[:,2]=.4;high[:,2]=1.6
        lx=torch.tensor((low-mean)/scale@rotation.T,device=DEVICE);hx=torch.tensor((high-mean)/scale@rotation.T,device=DEVICE)
        low_truth=physical_step(low[:,:2],low[:,2],raw['action'],family);high_truth=physical_step(high[:,:2],high[:,2],raw['action'],family)
        true_modulation=(high_truth-low_truth)/scale[:2]
        for fit_row in [r for r in fit_rows if r['family']==family]:
            state=torch.load(OUT/'checkpoints'/fit_row['checkpoint'],map_location='cpu',weights_only=True)
            model=Model(fit_row['method']).to(DEVICE);model.load_state_dict(state['world_model']);model.eval()
            decoder=restore_probe(state['decoder']);forward_probe=restore_probe(state['future_probe'])
            with torch.no_grad():
                z=model.encoder(test['x']);zn=model.encoder(test['y'])
                current=probe_predict(decoder,z,state['decoder_choice']);gain_pred=probe_predict(decoder,z,state['gain_choice'])[:,2]
                native=model_physical(model,test['x'],test['u'],decoder,state['decoder_choice'],tm,ts,tr)
                oracle_next=probe_predict(decoder,zn,state['decoder_choice'])[:,:2]
                probe_next=probe_predict(forward_probe,torch.cat([z,test['u']],1),state['future_choice'])
                native_lo=model_physical(model,lx,test['u'],decoder,state['decoder_choice'],tm,ts,tr)
                native_hi=model_physical(model,hx,test['u'],decoder,state['decoder_choice'],tm,ts,tr)
                probe_lo=probe_predict(forward_probe,torch.cat([model.encoder(lx),test['u']],1),state['future_choice'])
                probe_hi=probe_predict(forward_probe,torch.cat([model.encoder(hx),test['u']],1),state['future_choice'])
                native_modulation=((native_hi-native_lo)/ts[:2]).cpu().numpy();probe_modulation=(probe_hi-probe_lo).cpu().numpy()
                ytrue=test['future_labels'].cpu().numpy();norm_native=((native-tm[:2])/ts[:2]).cpu().numpy()
                g_true=test['current_labels'][:,2];g_mse=float((gain_pred-g_true).square().mean());g_var=float(g_true.var(unbiased=False))
                gain_accuracy=float(((gain_pred*ts[2]+tm[2]>1)==(test['current_labels'][:,2]*ts[2]+tm[2]>1)).float().mean())
                inverse_accuracy=None
                if model.inverse is not None:
                    inverse=model.inverse(torch.cat([z,zn],1));inverse_accuracy=float(((inverse>0)==(test['u']>0)).all(1).float().mean())
            def relative_error(pred):return float(np.sqrt(np.mean((pred-true_modulation)**2)/np.mean(true_modulation**2)))
            predfile=OUT/'predictions'/(fit_row['key']+'.npz')
            np.savez_compressed(predfile,native_physical_prediction=norm_native,frozen_encoder_successor_probe=probe_next.cpu().numpy(),
                oracle_successor_decoding=oracle_next.cpu().numpy(),current_probe=current.cpu().numpy(),gain_prediction=gain_pred.cpu().numpy(),
                true_successor=ytrue,true_modulation=true_modulation,native_modulation=native_modulation,probe_modulation=probe_modulation,
                trajectory_id=raw['trajectory_id'])
            row={k:fit_row[k] for k in ['key','family','method','seed','encoder_effective_rank','encoder_std_min','parameter_count']}
            row.update(gain_accuracy=gain_accuracy,gain_R2=1-g_mse/g_var,inverse_joint_action_accuracy=inverse_accuracy,
                native_physical_MSE=float(np.mean((norm_native-ytrue)**2)),frozen_encoder_successor_probe_MSE=float(np.mean((probe_next.cpu().numpy()-ytrue)**2)),
                oracle_successor_decoding_MSE=float(np.mean((oracle_next.cpu().numpy()-ytrue)**2)),
                native_modulation_relative_RMSE=relative_error(native_modulation),frozen_encoder_modulation_relative_RMSE=relative_error(probe_modulation),
                prediction_file=predfile.name,prediction_sha256=sha(predfile))
            rows.append(row);print(json.dumps(dict(phase='test',completed=len(rows),key=row['key'],gain_accuracy=gain_accuracy,
                response_error=row['native_modulation_relative_RMSE'])),flush=True)
    readiness={f:bool(all(r['validation_physical_MSE']<.05 for r in fit_rows if r['family']==f and r['method']=='fullobs')) for f in FAMILIES}
    result=dict(rows=rows,full_observation_validation_readiness=readiness,config=config,
        elapsed_seconds=time.perf_counter()-start,environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),maximum_allocated_bytes=torch.cuda.max_memory_allocated()),
        accepted_novel_directions=0,objective_achieved=False,scope='Exploratory mathematical objective transfers, not native source-paper reproductions; probes use privileged labels only after encoders are frozen.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(completed=72,seconds=result['elapsed_seconds'],readiness=readiness)),flush=True)
if __name__=='__main__':main()
