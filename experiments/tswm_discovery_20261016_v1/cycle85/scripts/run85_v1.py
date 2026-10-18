from common85_v1 import *
import time, math
from scipy.linalg import lstsq

BC,ALIGN,NATIVE_HASHES=native_modules()

def new_agent(seed,nu):
    torch.manual_seed(seed)
    return BC.BCObsAgent(dict(obs_dim=2,robot_obs_dim=2,obj_obs_dim=0,lat_obs_dim=2),dict(act_dim=nu+1),'cpu',n_layers=2,hidden_dim=64,lr=3e-4)

def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)

def path_scores(pred,y,h):
    return np.array([nrmse(pred[j:j+1],y[j:j+1],h) for j in range(len(y))])

def val_scores(agent,y,u):
    p=predict(agent,y[:,0],u)
    return dict(mean32=finite_score(nrmse(p,y,32)),median32=finite_score(float(np.median(path_scores(p,y,32)))),median128=finite_score(float(np.median(path_scores(p,y,min(128,u.shape[1]))))))

def source(family,seed):
    tag=f'{family}_{seed}';d=DATA/tag;w=WEIGHTS/tag;d.mkdir(parents=True,exist_ok=True);w.mkdir(parents=True,exist_ok=True)
    ready=O/f'SOURCE_READY85_{tag}_V1.json'
    if ready.exists():
        record=json.loads(ready.read_text());state=torch.load(record['checkpoint'],map_location='cpu',weights_only=True);a=new_agent(seed,record['nu']);unpack(a,state);freeze(a)
        payload=np.load(d/'source_trainval_v1.npz',allow_pickle=False)
        return a,{k:payload[k] for k in payload.files},record
    x,u=generate(family,seed,1000,128,64);xv,uv=generate(family,seed,2000,16,128)
    mu=np.mean(x[:,:-1],axis=(0,1));scale=np.std(x[:,:-1],axis=(0,1));assert np.all(scale>.01)
    y=(x-mu)/scale;yv=(xv-mu)/scale
    payload=dict(y=y,u=u,val_y=yv,val_u=uv,mu=mu,scale=scale)
    save_npz(d/'source_trainval_v1.npz',**payload)
    a=new_agent(seed,u.shape[-1]);rng=np.random.default_rng(seed+8000);native_log=NativeLogger();begin=time.perf_counter()
    log(dict(kind='source_start',tag=tag,native_hashes=NATIVE_HASHES,source_data_sha256=sha(d/'source_trainval_v1.npz')))
    src=y[:,:-1].reshape(-1,2);nxt=y[:,1:].reshape(-1,2);acts=u.reshape(-1,u.shape[-1]);dummy=np.column_stack([acts,np.zeros(len(acts))])
    for i in range(2000):
        ix=rng.integers(0,len(src),128)
        a.update_dyn_cons(tensor(src[ix]),tensor(dummy[ix]),tensor(nxt[ix]),native_log,i+1)
    params=trainable(a,['obs_enc','obs_dec','fwd_dyn','inv_dyn']);opt=torch.optim.Adam(params,lr=3e-4)
    best=float('inf');best_state=None;best_info=None;checks=[];passed=False
    for i in range(1,16001):
        yy,uu=batch_windows(y,u,rng,32);opt.zero_grad();oe,recon=output_loss(a,yy,uu)
        z=a.obs_enc(yy[:,0]);zn=a.obs_enc(yy[:,1]);inv=F.mse_loss(a.inv_dyn(torch.cat([z,zn],dim=-1)),uu[:,0])
        loss=oe+.1*recon+.001*inv;loss.backward();opt.step()
        if i not in [2000,4000,8000,16000]:continue
        scores=val_scores(a,yv,uv);info=dict(polish_updates=i,warmup_updates=2000,loss=float(loss.detach()),**scores)
        checks.append(info);log(dict(kind='source_validation',tag=tag,**info))
        if scores['mean32']<best:best=scores['mean32'];best_state=pack(a);best_info=info
        if scores['median32']<=.15 and scores['median128']<=.35:
            passed=True;best_state=pack(a);best_info=info;break
    assert best_state is not None
    unpack(a,best_state);freeze(a);checkpoint=w/'source_selected_v1.pt';torch.save(best_state,checkpoint)
    record=dict(utc=utc(),family=family,seed=seed,nu=u.shape[-1],qualified=passed,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selected=best_info,validation_checks=checks,source_data_sha256=sha(d/'source_trainval_v1.npz'),trained_parameters=sum(p.numel() for n in ['obs_enc','obs_dec','fwd_dyn','inv_dyn'] for p in getattr(a,n).parameters()),wall_seconds=time.perf_counter()-begin)
    write(ready,record);log(dict(kind='source_closed',tag=tag,qualified=passed,selected=best_info,wall_seconds=record['wall_seconds']))
    return a,payload,record

def anchor_loss(agent,source_agent,source_anchor,target_anchor):
    with torch.no_grad():z=source_agent.obs_enc(source_anchor)
    return F.mse_loss(agent.obs_enc(target_anchor),z)+F.mse_loss(agent.obs_dec(z),target_anchor)

def adapt_neural(src,base,case,family,seed,method,lr):
    suffix=f'{family}_{seed}_{case}_{method}_{lr:.0e}';path=O/f'FIT85_{suffix}_V1.json';wp=WEIGHTS/family/str(seed)/case/(method+f'_{lr:.0e}_selected_v1.pt')
    if path.exists():
        record=json.loads(path.read_text());a=copy.deepcopy(src);unpack(a,torch.load(record['checkpoint'],map_location='cpu',weights_only=True));freeze(a);return a,record
    a=copy.deepcopy(src);rng=np.random.default_rng(np.random.SeedSequence([seed,9000]));torch.manual_seed(seed+9000)
    sa=tensor(base['anchor_source']);ta=tensor(base['anchor_target']);y=base['y'];u=base['u'];yv=base['val_y'];uv=base['val_u']
    initial=pack(a);best_state=initial;best_step=0;best=val_scores(a,yv,uv)['mean32'];history=[dict(updates=0,validation_score=best)]
    begin=time.perf_counter();parameters=0
    if method=='native_frozen_alignment':
        params=trainable(a,['obs_enc','obs_dec']);parameters=sum(p.numel() for p in params)
        aligner=ALIGN.ObsAligner(src,a,'cpu',n_layers=2,hidden_dim=64,lr=lr)
        ysource=base['source_y'];usource=base['source_u'];sx=ysource[:,:-1].reshape(-1,2);sn=ysource[:,1:].reshape(-1,2);su=usource.reshape(-1,usource.shape[-1])
        tx=y[:,:-1].reshape(-1,2);tn=y[:,1:].reshape(-1,2);tu=u.reshape(-1,u.shape[-1]);logger=NativeLogger()
        parameters+=sum(p.numel() for m in [aligner.lat_disc,aligner.src_disc,aligner.tgt_disc] for p in m.parameters())
        for iteration in range(1,1001):
            ix=rng.integers(0,len(sx),128);iy=rng.integers(0,len(tx),128)
            xx,aa,nn=tensor(sx[ix]),tensor(su[ix]),tensor(sn[ix]);tt,bb,vv=tensor(tx[iy]),tensor(tu[iy]),tensor(tn[iy])
            aligner.update_disc(xx,aa,tt,bb,logger,iteration)
            aligner.update_gen(xx,aa,nn,tt,bb,vv,logger,iteration)
            aligner.tgt_obs_enc_opt.zero_grad();aligner.tgt_obs_dec_opt.zero_grad()
            loss=10*anchor_loss(a,src,sa,ta);loss.backward();aligner.tgt_obs_enc_opt.step();aligner.tgt_obs_dec_opt.step()
            updates=iteration*2
            if updates%200==0:
                score=val_scores(a,yv,uv)['mean32'];history.append(dict(updates=updates,validation_score=score))
                if score<best:best=score;best_state=pack(a);best_step=updates
                log(dict(kind='target_validation',fit=suffix,updates=updates,score=score,best=best))
    else:
        assert method=='joint_output_error'
        params=trainable(a,['obs_enc','obs_dec','fwd_dyn']);parameters=sum(p.numel() for p in params);origin=[p.detach().clone() for p in params];opt=torch.optim.Adam(params,lr=lr)
        for updates in range(1,2001):
            yy,uu=batch_windows(y,u,rng,32);opt.zero_grad();oe,recon=output_loss(a,yy,uu)
            prior=sum(torch.sum((p-q)**2) for p,q in zip(params,origin))/parameters
            loss=oe+.1*recon+10*anchor_loss(a,src,sa,ta)+1e-4*prior;loss.backward();opt.step()
            if updates%200==0:
                score=val_scores(a,yv,uv)['mean32'];history.append(dict(updates=updates,validation_score=score))
                if score<best:best=score;best_state=pack(a);best_step=updates
                log(dict(kind='target_validation',fit=suffix,updates=updates,score=score,best=best))
    unpack(a,best_state);freeze(a);wp.parent.mkdir(parents=True,exist_ok=True);torch.save(best_state,wp)
    record=dict(utc=utc(),family=family,seed=seed,case=case,method=method,lr=lr,selected_updates=best_step,completed_encoder_decoder_updates=2000,checkpoint=str(wp),checkpoint_sha256=sha(wp),validation_score=best,validation_history=history,trained_parameters=parameters,wall_seconds=time.perf_counter()-begin,original_paper_reproduction=False)
    write(path,record);return a,record

def dictionary_specs(dim):
    exponents=[z for z in itertools.product(range(4),repeat=dim) if sum(z)<=3]
    r=np.random.default_rng(202685);freq=r.normal(size=(dim,128));phase=r.uniform(0,2*np.pi,128)
    return dict(exponents=np.asarray(exponents,dtype=np.int64),frequency=freq,phase=phase)
def features(x,kind,spec):
    if kind=='polynomial3':return np.prod(x[:,None,:]**spec['exponents'][None,:,:],axis=-1)
    return np.column_stack([np.ones(len(x)),x,np.cos(x@spec['frequency']+spec['phase'])*np.sqrt(2/128)])
def ridge_roll(y0,u,coef,kind,spec):
    pred=[];state=y0.copy()
    with np.errstate(over='ignore',invalid='ignore'):
        for t in range(u.shape[1]):
            invalid=(~np.isfinite(state).all(axis=1))|(np.max(np.abs(state),axis=1)>1e4)
            safe=state.copy();safe[invalid]=0
            nxt=state+features(np.column_stack([safe,u[:,t]]),kind,spec)@coef
            nxt[invalid]=np.nan;pred.append(nxt);state=nxt
    return np.stack(pred,axis=1)
def ordinary(srcdata,base,family,seed,case):
    data_input=np.column_stack([base['y'][:,:-1].reshape(-1,2),base['u'].reshape(-1,base['u'].shape[-1])]);target=(base['y'][:,1:]-base['y'][:,:-1]).reshape(-1,2)
    source_input=np.column_stack([srcdata['y'][:,:-1].reshape(-1,2),srcdata['u'].reshape(-1,srcdata['u'].shape[-1])]);source_target=(srcdata['y'][:,1:]-srcdata['y'][:,:-1]).reshape(-1,2)
    spec=dictionary_specs(data_input.shape[-1]);records=[];chosen={};begin=time.perf_counter()
    for kind in ['polynomial3','rff128']:
        fs=features(source_input,kind,spec);ft=features(data_input,kind,spec);width=ft.shape[1]
        cs=np.linalg.lstsq(np.vstack([fs,np.sqrt(1e-6)*np.eye(width)]),np.vstack([source_target,np.zeros((width,2))]),rcond=None)[0]
        best=float('inf');best_coefficient=None;best_meta=None
        for lam in [1e-6,1e-3,.1]:
            for prior in [0.,1e-3,.1,10.]:
                coefficient=np.linalg.solve(ft.T@ft+(lam+prior)*np.eye(width),ft.T@target+prior*cs)
                prediction=ridge_roll(base['val_y'][:,0],base['val_u'],coefficient,kind,spec)
                score=finite_score(nrmse(prediction,base['val_y'],32));record=dict(method=kind,ridge=lam,source_prior=prior,validation_score=score)
                records.append(record)
                if score<best:best=score;best_coefficient=coefficient;best_meta=record
        assert best_coefficient is not None
        chosen[kind]=dict(coef=best_coefficient,spec=spec,meta=best_meta,source_coef=cs)
    path=RAW/f'{family}_{seed}_{case}_ordinary_selected_v1.npz'
    save_npz(path,polynomial3_coef=chosen['polynomial3']['coef'],rff128_coef=chosen['rff128']['coef'],polynomial3_source_coef=chosen['polynomial3']['source_coef'],rff128_source_coef=chosen['rff128']['source_coef'],**spec)
    record=dict(utc=utc(),family=family,seed=seed,case=case,candidates=records,selected={k:v['meta'] for k,v in chosen.items()},coefficient_path=str(path),coefficient_sha256=sha(path),wall_seconds=time.perf_counter()-begin)
    write(O/f'ORDINARY85_{family}_{seed}_{case}_V1.json',record)
    return chosen,record

def evaluate_case(src,srcdata,sr,family,seed,case):
    tag=f'{family}_{seed}_{case}';complete=O/f'CASE85_{tag}_V1.json'
    if complete.exists():return json.loads(complete.read_text())
    base=target_data(family,seed,srcdata['mu'],srcdata['scale'],case);base['source_y']=srcdata['y'];base['source_u']=srcdata['u']
    data_path=DATA/f'{tag}_target_trainval_v1.npz';save_npz(data_path,**{k:v for k,v in base.items() if not k.startswith('source_')})
    models={'frozen':src};candidates=[];selected={}
    for method in ['native_frozen_alignment','joint_output_error']:
        fits=[]
        for lr in [3e-4,1e-3]:
            log(dict(kind='target_fit_start',tag=tag,method=method,lr=lr))
            a,record=adapt_neural(src,base,case,family,seed,method,lr);fits.append((a,record));candidates.append(record)
        best=min(fits,key=lambda z:z[1]['validation_score']);models[method]=best[0];selected[method]=best[1]
    dictionaries,ordinary_record=ordinary(srcdata,base,family,seed,case)
    scores={m:r['validation_score'] for m,r in selected.items()}
    scores.update({m:r['meta']['validation_score'] for m,r in dictionaries.items()})
    best_existing=min(scores,key=scores.get)
    seal=O/f'SELECTION_SEALED85_{tag}_V1.json'
    write(seal,dict(utc=utc(),family=family,seed=seed,case=case,selected_neural=selected,selected_ordinary=ordinary_record['selected'],best_existing=best_existing,validation_scores=scores,target_data_sha256=sha(data_path),source_checkpoint_sha256=sr['checkpoint_sha256'],eval_generated=False))
    # First generation/access of the independent EVAL namespace occurs after the selection seal.
    x,u=generate(family,seed,6000,8,128,case in ['dynamics','joint'],paired=True)
    truth=observe((x-srcdata['mu'])/srcdata['scale'],case in ['observation','joint'])
    rng=np.random.default_rng(np.random.SeedSequence([seed,10000]));initial=truth[:,0].copy();noise=np.repeat(rng.normal(0,.01,(8,2)),2,axis=0);initial+=noise
    arrays=dict(reference=truth,initial_observation=initial,actions=u,physical_reference=x,mu=srcdata['mu'],scale=srcdata['scale'])
    rows=[];method_summaries={}
    for name in ['frozen','native_frozen_alignment','joint_output_error','polynomial3','rff128']:
        if name in models:p=predict(models[name],initial,u)
        else:
            z=dictionaries[name];p=ridge_roll(initial,u,z['coef'],name,z['spec'])
        arrays[name+'_prediction']=p
        hs={}
        for h in [32,128]:
            metric=path_scores(p,truth,h);fail=~np.isfinite(metric);hs[str(h)]=dict(median_nrmse=finite_score(float(np.median(metric))),failed=int(fail.sum()),total=len(metric))
            for j,value in enumerate(metric):rows.append(dict(method=name,kind='trajectory',horizon=h,path=j,score=finite_score(float(value)),failed=bool(not np.isfinite(value))))
        response=[]
        for j in range(8):
            predicted=p[2*j+1]-p[2*j];actual=truth[2*j+1,1:]-truth[2*j,1:]
            bad=(not np.isfinite(predicted).all()) or np.max(np.abs(p[2*j:2*j+2]))>1e4
            score=float('inf') if bad else float(np.sqrt(np.mean((predicted-actual)**2))/max(float(np.sqrt(np.mean(actual**2))),.05))
            response.append(score);rows.append(dict(method=name,kind='action_response',horizon=128,path=j,score=finite_score(score),failed=not np.isfinite(score)))
        hs['response']=dict(median_error=finite_score(float(np.median(response))),failed=int(sum(not np.isfinite(z) for z in response)),total=8)
        method_summaries[name]=hs
    path=RAW/f'{tag}_evaluation_v1.npz';save_npz(path,**arrays)
    best_summary=method_summaries[best_existing]
    ordinary_suffices=best_summary['128']['failed']==0 and best_summary['response']['failed']==0 and best_summary['128']['median_nrmse']<=.2 and best_summary['response']['median_error']<=.3
    record=dict(utc=utc(),family=family,seed=seed,case=case,selection_seal_sha256=sha(seal),source_ready_sha256=sha(O/f'SOURCE_READY85_{family}_{seed}_V1.json'),best_existing_by_validation=best_existing,selected_existing_suffices=ordinary_suffices,summary=method_summaries,rows=rows,array_path=str(path),array_sha256=sha(path),target_data_path=str(data_path),target_data_sha256=sha(data_path),source_qualified=True,new_method_tested=False)
    write(complete,record);log(dict(kind='case_closed',tag=tag,best_existing=best_existing,suffices=ordinary_suffices,summary=method_summaries))
    assert sha(sr['checkpoint'])==sr['checkpoint_sha256'],'Source checkpoint changed'
    return record

def main():
    O.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    qualification=json.loads((O/'QUALIFICATION85_V1.json').read_text())
    assert qualification['status']=='PASS' and qualification['native_hashes']==NATIVE_HASHES
    assert qualification['protocol_sha256']==sha(C/'PROTOCOL85_V1.md')
    log(dict(kind='campaign_start',pid=os.getpid(),protocol_sha256=sha(C/'PROTOCOL85_V1.md'),source_hashes=NATIVE_HASHES,device='cpu',torch=torch.__version__,numpy=np.__version__))
    allcases=[];sources=[]
    for family in ['duffing','brusselator']:
        for seed in SEEDS:
            write(O/'PROGRESS85_V1.json',dict(status='SOURCE_OR_TARGET_FITTING',pid=os.getpid(),family=family,seed=seed,closed_cases=len(allcases),utc=utc()),replace=True)
            a,data,record=source(family,seed);sources.append(record)
            if not record['qualified']:continue
            for case in SHIFT_CASES:
                write(O/'PROGRESS85_V1.json',dict(status='TARGET_FITTING',pid=os.getpid(),family=family,seed=seed,case=case,closed_cases=len(allcases),utc=utc()),replace=True)
                allcases.append(evaluate_case(a,data,record,family,seed,case))
    result=dict(utc=utc(),status='EXISTING_REPAIR_SCREEN_COMPLETE_NOT_FINAL_RESEARCH',sources=sources,qualified_sources=sum(z['qualified'] for z in sources),closed_cases=len(allcases),cases=[{k:v for k,v in c.items() if k!='rows'} for c in allcases],qualified_research_directions=0,new_method_tested=False)
    write(O/'RESULT85_V1.json',result)
    write(O/'PROGRESS85_V1.json',dict(status=result['status'],pid=os.getpid(),closed_cases=len(allcases),qualified_sources=result['qualified_sources'],utc=utc()),replace=True)
    log(dict(kind='campaign_closed',qualified_sources=result['qualified_sources'],cases=len(allcases),result_sha256=sha(O/'RESULT85_V1.json')))

if __name__=='__main__':main()
