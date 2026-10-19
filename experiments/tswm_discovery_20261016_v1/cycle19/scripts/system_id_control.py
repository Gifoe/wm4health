"""Exploratory old-style vector-field library regression, selected on past-only data."""
import runtime_profile
import hashlib,itertools,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent/'cycle18'
OUT=ROOT/'outputs/system_id_control'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def powers(degree):
    return np.asarray([p for p in itertools.product(range(degree+1),repeat=3) if sum(p)<=degree],dtype='int64')
def features(z,pows,knots):
    a=np.asarray(z,dtype='float64').reshape(-1,3)
    out=np.prod(a[:,None,:]**pows[None,:,:],axis=2)
    if knots.size:out=np.concatenate([out,np.maximum(a[:,:,None]-knots[None,:,:],0).reshape(len(a),-1)],axis=1)
    return out
def fit(context,degree,hinges,ridge):
    mean=context.mean(0);scale=context.std(0,ddof=1);assert np.all(scale>0)
    z=(context-mean)/scale;pows=powers(degree)
    knots=np.quantile(z,np.arange(1,10)/10,axis=0).T if hinges else np.empty((3,0))
    phi=features(z[1:-1],pows,knots);feature_scale=np.sqrt(np.mean(phi**2,axis=0));feature_scale=np.maximum(feature_scale,1e-12)
    design=phi/feature_scale;target=(z[2:]-z[:-2])/2
    penalty=np.eye(design.shape[1])*ridge;constant=np.flatnonzero(pows.sum(1)==0)[0];penalty[constant,constant]=0
    coeff=np.linalg.solve(design.T@design/len(design)+penalty,design.T@target/len(design))/feature_scale[:,None]
    return dict(mean=mean,scale=scale,powers=pows,knots=knots,coeff=coeff)
def derivative(z,model):return features(z,model['powers'],model['knots'])@model['coeff']
def step(z,model):
    current=np.asarray(z,dtype='float64').reshape(-1,3).copy();h=.25
    with np.errstate(over='ignore',invalid='ignore'):
        for _ in range(4):
            k1=derivative(current,model);k2=derivative(current+h*k1/2,model)
            k3=derivative(current+h*k2/2,model);k4=derivative(current+h*k3,model)
            current+=h*(k1+2*k2+2*k3+k4)/6
            if not np.isfinite(current).all() or np.max(np.abs(current))>1e4:return None
    return current
def rollout(initial,steps,model):
    z=(np.asarray(initial).reshape(-1,3)-model['mean'])/model['scale'];result=[]
    for i in range(steps):
        z=step(z,model)
        if z is None:return np.asarray(result),i
        result.append(z*model['scale']+model['mean'])
    return np.asarray(result),None
def select(context):
    split=int(.8*len(context));train=context[:split];origins=np.arange(split,len(context)-64,32)
    assert len(origins)>0 and max(origins)+64<len(context)
    truth=np.stack([context[o+1:o+65] for o in origins],axis=1);rows=[]
    for degree,hinges,ridge in itertools.product([2,3],[False,True],[1e-8,1e-5,1e-2]):
        model=fit(train,degree,hinges,ridge);pred,failed=rollout(context[origins],64,model)
        score=float(np.mean(((pred-truth)/model['scale'])**2)) if failed is None else None
        row=dict(degree=degree,hinges=hinges,ridge=ridge,validation_MSE=score,diverged_validation_step=failed)
        rows.append(row)
    valid=[r for r in rows if r['validation_MSE'] is not None];assert valid,'Every past-only validation model diverged'
    best=min(valid,key=lambda r:r['validation_MSE']);model=fit(context,best['degree'],best['hinges'],best['ridge'])
    return model,dict(train_prefix=split,validation_initial_state_indices=origins.tolist(),validation_horizon=64,candidates=rows,selected=best)
def numerical_checks():
    # Exact regression of a degree2 vector field, independent prescribed coefficient target.
    rng=np.random.default_rng(20261904);a=rng.normal(size=(500,3));p=powers(2);coef=rng.normal(size=(len(p),3))*.01
    matrix=features(a,p,np.empty((3,0)));target=matrix@coef
    recovered=np.linalg.lstsq(matrix,target,rcond=None)[0];e=float(np.max(np.abs(recovered-coef)));assert e<1e-12
    # Analytic stable diagonal linear flow over64 sample indices checks batching and index convention.
    coef=np.zeros((len(p),3));rates=np.asarray([-.05,-.03,-.01])
    for j in range(3):
        power=np.zeros(3,dtype='int64');power[j]=1;idx=np.flatnonzero(np.all(p==power,axis=1))[0];coef[idx,j]=rates[j]
    model=dict(mean=np.zeros(3),scale=np.ones(3),powers=p,knots=np.empty((3,0)),coeff=coef)
    initial=np.asarray([[1.,2.,3.],[-1.,.3,2.]]);pred,failed=rollout(initial,64,model)
    exact=initial[None,:,:]*np.exp(np.arange(1,65)[:,None,None]*rates[None,None,:])
    err=float(np.max(np.abs(pred-exact)));assert failed is None and err<1e-9,err
    return dict(exact_library_coefficient_error=e,linear_analytic_RK4_max_error=err)
def main():
    started=time.perf_counter();assert not OUT.exists();OUT.mkdir(parents=True)
    checks=numerical_checks();sys.path.insert(0,str(BASE/'upstream/DynaMix-python/src'))
    import torch
    from dynamix.metrics.metrics import geometrical_misalignment,temporal_misalignment,MASE
    from path_event_audit import summarize,compare
    tasks=[('lorenz63_original','lorenz63',2000,0,2000),('chua_original','chua',1028,0,1028),
        ('sprottM_original','sprottM',512,0,512),('chua_common_short','chua',2000,972,1028),('chua_common_long','chua',2000,0,1028)]
    rows=[];rng=np.random.default_rng(20261905)
    for label,name,origin,start,reference_length in tasks:
        tick=time.perf_counter();source=BASE/'upstream/DynaMix-python/notebooks/test_data'/(name+'.npy')
        data=np.load(source,allow_pickle=False).astype('float32').astype('float64')
        context=data[start:origin];reference_context=data[origin-reference_length:origin];before=sha(source)
        # Future is not accessed until all specification selection and fitting has finished.
        model,selection=select(context.copy());prediction,failed=rollout(context[-1],10000,model)
        prediction=prediction[:,0,:] if len(prediction) else np.empty((0,3));truth=data[origin:origin+10000]
        file=OUT/(label+'.npz');np.savez_compressed(file,prediction=prediction,truth=truth,context=context,reference_context=reference_context,**model)
        reloaded=np.load(file,allow_pickle=False);stored={k:reloaded[k] for k in model};replay,failed_replay=rollout(context[-1],min(64,len(prediction)),stored)
        assert failed_replay is None and np.array_equal(replay[:,0,:],prediction[:64]);assert sha(source)==before
        row=dict(task=label,system=name,origin=origin,context_start=start,context_length=len(context),
            selection=selection,diverged_at_future_index=failed,array_file=file.name,array_sha256=sha(file),source_sha256=before)
        if failed is None:
            p=torch.tensor(prediction,dtype=torch.float32);t=torch.tensor(truth,dtype=torch.float32)
            summary=summarize(prediction[:,0],reference_context[:,0]);reference=summarize(truth[:,0],reference_context[:,0])
            row['global_metrics']=dict(D_stsp=float(geometrical_misalignment(p,t,n_bins=30)),
                D_H=None if (p.std(0)<1e-8).any() else float(temporal_misalignment(p,t,smoothing=20)),
                MASE_true_denominator=float(MASE(p,t,steps=10)))
            row['event_summaries']=summary[0];row['event_comparison']=compare(summary,reference,rng)
        row['seconds']=time.perf_counter()-tick;rows.append(row)
        print(json.dumps({k:row[k] for k in ['task','seconds','diverged_at_future_index']}),flush=True)
    def clean(v):
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    result=dict(rows=rows,numerical_checks=checks,elapsed_seconds=time.perf_counter()-started,
        script_sha256=sha(pathlib.Path(__file__)),plan_sha256=sha(ROOT/'SYSTEM_ID_CONTROL_PLAN.md'),
        event_helper_sha256=sha(ROOT/'scripts/path_event_audit.py'),environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__),
        protocol='Exploratory per-context dense ridge library regression; past-only hyperparameter selection; not native SINDy reproduction or compute-matched universal-model comparison.',
        accepted_novel_directions=0,objective_achieved=False)
    (OUT/'results.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(completed=len(rows),seconds=result['elapsed_seconds'])),flush=True)
if __name__=='__main__':main()
