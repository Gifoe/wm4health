from common86_v1 import *
import copy,time,platform

STAGE=O/'paired_v1';STAGE.mkdir(parents=True,exist_ok=True)
DEST=Path(CONFIG['artifact_root'])/'cycle86'/'paired_v1';DEST.mkdir(parents=True,exist_ok=True)
WEIGHTS=Path(CONFIG['checkpoint_root'])/'cycle86'/'paired_v1';WEIGHTS.mkdir(parents=True,exist_ok=True)

class Predictor(torch.nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.net=torch.nn.Sequential(torch.nn.Linear(16,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,3 if kind=='generator' else 32))
    def forward(self,x):
        out=self.net(x.flatten(1))
        if self.kind=='direct':return out.reshape(-1,16,2)
        p=.8*torch.tanh(out);a=torch.stack([torch.stack([p[:,0],p[:,1]],dim=-1),torch.stack([p[:,1],p[:,2]],dim=-1)],dim=1);a2=a@a;a3=a2@a;a4=a3@a;t=torch.eye(2)[None]+.05*a+.00125*a2+(.05**3/6)*a3+(.05**4/24)*a4;s=x[:,-1];states=[]
        for _ in range(16):s=torch.bmm(t,s[:,:,None]).squeeze(-1);states.append(s)
        return torch.stack(states,dim=1)
def flow(a,initial,times):
    values,q=np.linalg.eigh(a);z=np.einsum('nji,nj->ni',q,initial);return np.einsum('nij,ntj->nti',q,z[:,None]*np.exp(values[:,None]*times[None,:,None]))
def window_data(seed,d,ids,resample=False):
    initial=d['initial'][ids,0].copy();rng=np.random.default_rng(np.random.SeedSequence([seed,3000]))
    allinit=rng.uniform(-1,1,(len(d['a']),2))
    if resample:initial=allinit[ids]
    clean=flow(d['a'][ids],initial,np.arange(24)*.05);noise=np.random.default_rng(np.random.SeedSequence([seed,4001 if resample else 4000])).normal(0,.01,(len(d['a']),24,2))[ids];y=clean+noise;assert np.isfinite(y).all();return y.astype(np.float32),clean
def source_data(seed):
    r=json.loads((O/f'POPULATION86_{seed}_V1.json').read_text());assert sha(r['array_path'])==r['array_sha256'];d=np.load(r['array_path'],allow_pickle=False);selected=d['p1_3_mask'];rng=np.random.default_rng(np.random.SeedSequence([seed,5000]));train_selected=rng.choice(np.flatnonzero(selected&(d['split']==0)),1024,replace=False);train_all=rng.choice(np.flatnonzero(d['split']==0),1024,replace=False);valid_selected=np.flatnonzero(selected&(d['split']==1))[:384];valid_all=np.flatnonzero(d['split']==1)[:384];answer={}
    for arm,train,val,resample in [('whole_filter',train_selected,valid_selected,False),('resampled_ic',train_selected,valid_selected,True),('prefix_all',train_all,valid_all,False)]:
        y,clean=window_data(seed,d,train,resample);yv,vc=window_data(seed,d,val,resample);answer[arm]=dict(train_ids=train,val_ids=val,train_y=y,val_y=yv,train_clean=clean,val_clean=vc)
    p=DEST/f'source_{seed}_v1.npz';save(p,**{arm+'_'+k:v for arm,ds in answer.items() for k,v in ds.items()});return answer,p
def mse(model,x,y):
    with torch.no_grad():return float(torch.mean((model(x)-y)**2))
def fit(seed,kind,arm,ds,initial):
    model=Predictor(kind);model.load_state_dict(initial);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);x=torch.from_numpy(ds['train_y'][:,:8]);y=torch.from_numpy(ds['train_y'][:,8:]);xv=torch.from_numpy(ds['val_y'][:,:8]);yv=torch.from_numpy(ds['val_y'][:,8:]);rng=np.random.default_rng(np.random.SeedSequence([seed,6000]));trace=[];best=None;schedules=[];start=time.perf_counter()
    def check(step):
        nonlocal best
        score=mse(model,xv,yv);path=WEIGHTS/f'{seed}_{kind}_{arm}_step{step:04}_v1.pt'
        with path.open('xb') as f:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step),f)
        row=dict(step=step,val_mse=score,checkpoint_path=str(path),checkpoint_sha256=sha(path));trace.append(row)
        if best is None or score<best['val_mse']:best=dict(row)
    check(0)
    for step in range(1,3001):
        ids=rng.integers(0,1024,64);schedules.append(ids);prediction=model(x[ids]);loss=torch.mean((prediction-y[ids])**2);assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%300==0:check(step);log(dict(kind='paired_fit_progress',seed=seed,architecture=kind,arm=arm,step=step,val_mse=trace[-1]['val_mse']))
    persistence=float(torch.mean((xv[:,-1,None]-yv)**2));record=dict(seed=seed,architecture=kind,arm=arm,trace=trace,selected=best,persistence_val_mse=persistence,source_qualified=best['val_mse']<persistence,parameters=sum(p.numel() for p in model.parameters()),updates=3000,elapsed_seconds=time.perf_counter()-start,schedule_sha256=hashlib.sha256(np.asarray(schedules,dtype=np.int64).tobytes()).hexdigest(),initial_state_sha256=hashlib.sha256(b''.join(initial[k].numpy().tobytes() for k in sorted(initial))).hexdigest());write(STAGE/f'FIT86_{seed}_{kind}_{arm}_V1.json',record);log(dict(kind='paired_fit_closed',**{k:v for k,v in record.items() if k not in ['trace','selected']},selected=best));return record
def linear_predict(x,lam):
    a=x[:,:-1].reshape(-1,7,2);b=np.diff(x,axis=1);w=[];out=[]
    for aa,bb,xx in zip(a,b,x):
        delta=np.linalg.lstsq(np.vstack([aa,np.sqrt(lam)*np.eye(2)]),np.vstack([bb,np.zeros((2,2))]),rcond=None)[0];t=np.eye(2)+delta;s=xx[-1].astype(np.float64).copy();states=[]
        for _ in range(16):s=s@t;states.append(s.copy())
        out.append(states);w.append(t)
    return np.array(out),np.array(w)
def scores(pred,truth):
    failed=(~np.isfinite(pred).all(axis=(1,2)))|(np.max(np.abs(pred),axis=(1,2))>1e4);mse=np.mean((pred-truth)**2,axis=(1,2));nrmse=np.sqrt(mse)/np.maximum(np.sqrt(np.mean(truth**2,axis=(1,2))),.1);mse[failed]=1e12;nrmse[failed]=1e12;return mse,nrmse,failed
def main():
    log(dict(kind='paired_training_start',pid=os.getpid(),protocol_sha256=sha(C/'PAIRED_PRETRAIN86_V1.md')));fits=[];sources=[];linear=[]
    for seed in SEEDS:
        data,path=source_data(seed);sources.append(dict(seed=seed,path=str(path),sha256=sha(path)))
        for kind in ['generator','direct']:
            torch.manual_seed(seed+(0 if kind=='generator' else 10));initial=copy.deepcopy(Predictor(kind).state_dict())
            for arm in ['whole_filter','resampled_ic','prefix_all']:fits.append(fit(seed,kind,arm,data[arm],initial))
        ds=data['prefix_all'];records=[]
        for lam in [0.,1e-4,.01,.1,1.]:
            p,w=linear_predict(ds['val_y'][:,:8].astype(np.float64),lam);records.append(dict(ridge=lam,val_mse=float(np.mean((p-ds['val_y'][:,8:])**2))))
        linear.append(dict(seed=seed,candidates=records,selected=min(records,key=lambda r:r['val_mse'])))
    seal=STAGE/'SELECTION_SEALED86_V1.json';write(seal,dict(utc=utc(),fits=fits,sources=sources,linear=linear,eval_generated=False,protocol_sha256=sha(C/'PAIRED_PRETRAIN86_V1.md')))
    from sample86_v1 import population
    seed=2026864;rates,angles,a,initial=population(seed);n=1024;rates,angles,a,initial=rates[:n],angles[:n],a[:n],initial[:n];dummy=dict(a=a,initial=initial);ids=np.arange(n);noisy,clean=window_data(seed,dummy,ids);allpaths=flow(a,initial[:,0],np.linspace(0,4,81));mask=np.max(np.abs(allpaths),axis=(1,2))<3.;truth=clean[:,8:];history=noisy[:,:8];saved=dict(rates=rates,a=a,initial=initial,history=history,reference=truth,selected_population=mask);results=[]
    for r in fits:
        model=Predictor(r['architecture']);model.load_state_dict(torch.load(r['selected']['checkpoint_path'],map_location='cpu',weights_only=False)['model']);model.eval()
        with torch.no_grad():p=model(torch.from_numpy(history)).numpy().astype(np.float64)
        label=f"{r['seed']}_{r['architecture']}_{r['arm']}";saved[label+'_prediction']=p;mm,nn,ff=scores(p,truth);saved[label+'_mse']=mm;saved[label+'_nrmse']=nn;saved[label+'_failed']=ff
        summaries={}
        for name,which in [('all',np.ones(n,dtype=bool)),('selected',mask),('rejected',~mask)]:summaries[name]=dict(n=int(which.sum()),mean_mse=float(mm[which].mean()),median_nrmse=float(np.median(nn[which])),mean_nrmse=float(nn[which].mean()),failed=int(ff[which].sum()))
        results.append(dict(label=label,seed=r['seed'],architecture=r['architecture'],arm=r['arm'],source_qualified=r['source_qualified'],summary=summaries))
    for label,p in [('persistence',np.repeat(history[:,-1,None],16,axis=1).astype(np.float64))]+[(f"linear_{r['seed']}",linear_predict(history.astype(np.float64),r['selected']['ridge'])[0]) for r in linear]:
        saved[label+'_prediction']=p;mm,nn,ff=scores(p,truth);saved[label+'_mse']=mm;saved[label+'_nrmse']=nn;saved[label+'_failed']=ff;results.append(dict(label=label,architecture='ordinary',arm=label,summary={name:dict(n=int(which.sum()),mean_mse=float(mm[which].mean()),median_nrmse=float(np.median(nn[which])),mean_nrmse=float(nn[which].mean()),failed=int(ff[which].sum())) for name,which in [('all',np.ones(n,dtype=bool)),('selected',mask),('rejected',~mask)]}))
    p=DEST/'evaluation_2026864_v1.npz';save(p,**saved);out=dict(status='PAIRED_FACTUAL_LEARNING_COMPLETE',utc=utc(),fits=fits,sources=sources,linear=linear,results=results,eval_array_path=str(p),eval_array_sha256=sha(p),selection_seal_sha256=sha(seal),new_eval_seed=2026864,released_checkpoint_tested=False,new_method_validated=False,accepted_research_directions=0,local_result_payload_files_written=0);write(STAGE/'RESULT86_PAIRED_V1.json',out);log(dict(kind='paired_training_complete',fits=len(fits),qualified=sum(r['source_qualified'] for r in fits),evaluated_models=len(results)))
if __name__=='__main__':main()
