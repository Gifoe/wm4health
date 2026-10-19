"""Ordinary canonical block laws and AR, no new neural training."""
import pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent;sys.path.insert(0,str(B/'cycle34/scripts'))
from sundial_common34 import *
import pandas as pd,time,datetime
O=R/'outputs/kernel_controls_v1';assert not O.exists()
Q=R/'outputs/native_queries_v2';verified=json.loads((Q/'verification.json').read_text(encoding='utf-8'))
assert verified['passed'] and verified['results_sha256']==sha(Q/'result.json')
D=json.loads((Q/'result.json').read_text(encoding='utf-8'));O.mkdir()
@torch.inference_mode()
def enc(model,h):
    mu=h.mean(-1,keepdim=True);sd=h.std(-1,keepdim=True,unbiased=False)+1e-5
    z=model.model(input_ids=(h-mu)/sd,return_dict=True,use_cache=False).last_hidden_state[:,-1]
    return {'z':z,'mean':mu,'scale':sd}
def crps(samples,y):
    a=np.asarray(samples,dtype='float64');a=a.reshape(len(a),-1);b=np.asarray(y,dtype='float64').reshape(-1)
    n=len(a);order=np.sort(a,axis=0);coeff=(2*np.arange(n)-n+1)[:,None]
    return float((np.mean(np.abs(a-b),axis=0)-np.sum(coeff*order,axis=0)/(n*(n-1))).mean())
def score(a,y):
    return {'hourly_crps':crps(a,y),'mean_crps':crps(a.mean(-1),np.mean(y)),
            'terminal_crps':crps(a[:,-1],y[-1]),'ramp_crps':crps(np.max(np.abs(np.diff(a,axis=-1)),axis=-1),np.max(np.abs(np.diff(y))))}
def arfit(h):
    mu=float(h.mean());sd=float(h.std())+1e-5;z=(h-mu)/sd
    features=np.column_stack([np.ones(len(z)-8)]+[z[8-j:len(z)-j] for j in range(1,9)])
    target=z[8:];pen=np.diag([0.]+[.001]*8);coef=np.linalg.solve(features.T@features+pen,features.T@target)
    resid=target-features@coef;sigma=float(np.sqrt(np.mean(resid*resid)))
    return coef,sigma,mu,sd
def arsample(h,fit,rng):
    coef,sigma,mu,sd=fit;past=np.repeat(((h[-8:]-mu)/sd)[None],256,axis=0);result=[]
    for k in range(192):
        point=coef[0]+past[:,-np.arange(1,9)]@coef[1:]+sigma*rng.normal(size=256);result.append(point);past=np.column_stack([past[:,1:],point])
    return np.stack(result,axis=1)*sd+mu
begin=time.monotonic();validate_sources();model=load_model();before=state_sha(model);rows=[];fits=[];files=[]
with torch.inference_mode():
    # Fixed warmup, excluded from recorded arm latency.
    h=torch.sin(torch.arange(672,device='cuda',dtype=torch.float32)/13)[None]
    from_noise(model,enc(model,h),torch.zeros((2,1,720),device='cuda'),192);torch.cuda.synchronize()
for index,meta in enumerate(D['inputs']):
    if meta['excluded'] is not None:continue
    task=meta['dataset'];end=meta['end_index_exclusive'];old=Q/f'{task}_{end}';h=np.load(old/'history.npy')
    path=B/'cycle34/assets/fev'/('solar_with_weather/15T/train-00000-of-00001.parquet' if task=='solar' else 'jena_weather/10T/train-00000-of-00001.parquet')
    allvalues=np.asarray(pd.read_parquet(path).iloc[0][meta['field']],dtype='float32');truth=allvalues[end:end+192]
    if len(truth)!=192 or not np.isfinite(truth).all():
        rows.append({'dataset':task,'end':end,'excluded_future_nonfinite':True});continue
    mu=float(h.mean());sd=float(h.std())+1e-5;folder=O/f'{task}_{end}';folder.mkdir()
    at=time.perf_counter();fit=arfit(h);fit_seconds=time.perf_counter()-at
    fits.append({'dataset':task,'end':end,'seconds':fit_seconds,'coefficient':fit[0].tolist(),'sigma_z':fit[1]})
    for rep in [2026391,2026392]:
        seed(rep+index*100);noise=torch.randn(256,1,720,device='cuda');predictions={};durations={}
        with torch.inference_mode():
            torch.cuda.synchronize();t=time.perf_counter();ht=torch.tensor(h[None],device='cuda')
            a=from_noise(model,enc(model,ht),noise,192).squeeze(0);torch.cuda.synchronize();durations['native_joint']=time.perf_counter()-t
            predictions['native_joint']=a.cpu().numpy();prior=np.load(old/f'direct_{rep}.npy')[:,:192]
            assert np.max(np.abs(predictions['native_joint']-prior))<2e-4
            for block in [96,8]:
                torch.cuda.synchronize();t=time.perf_counter();past=torch.tensor(np.repeat(h[None],256,axis=0),device='cuda');segments=[]
                for step in range(192//block):
                    if step==0:piece=from_noise(model,enc(model,ht),noise,block).squeeze(0)
                    else:
                        seed(rep+index*100+block*100000+step+9000000)
                        e=torch.randn(1,256,720,device='cuda');piece=from_noise(model,enc(model,past),e,block).squeeze(1)
                    past=torch.cat([past,piece],dim=1);segments.append(piece)
                result=torch.cat(segments,dim=1);torch.cuda.synchronize();durations[f'canonical{block}']=time.perf_counter()-t
                predictions[f'canonical{block}']=result.cpu().numpy()
        t=time.perf_counter();predictions['ar8']=arsample(h,fit,np.random.default_rng(rep+index*100));durations['ar8']=time.perf_counter()-t
        for arm,a in predictions.items():
            assert a.shape==(256,192) and np.isfinite(a).all()
            rows.append({'dataset':task,'end':end,'rep':rep,'arm':arm,'excluded_future_nonfinite':False,
                         **score((a.astype('float64')-mu)/sd,(truth.astype('float64')-mu)/sd),'prediction_seconds':durations[arm],
                         'head_calls':{'native_joint':1,'canonical96':2,'canonical8':24,'ar8':0}[arm]})
        p=folder/f'predictions_{rep}.npz';np.savez_compressed(p,history=h,truth=truth,**predictions);files.append({'path':p.relative_to(O).as_posix(),'sha256':sha(p)})
        print(json.dumps({'context':folder.name,'rep':rep,'arms_complete':len(rows)}),flush=True)
after=state_sha(model);assert before==after
out={'status':'completed_pending_independent_verification','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'seconds':time.monotonic()-begin,'rows':rows,'ar_context_fits':fits,'files':files,'neural_updates':0,
     'model_state_before':before,'model_state_after':after,'test_scored':False,'exploratory_early_context_future_scored':True,
     'not_a_new_method':True,'accepted_directions':0,'source_query_result_sha256':sha(Q/'result.json'),
     'plan_sha256':sha(R/'ORDINARY_KERNEL_CONTROL_PLAN.md'),'script_sha256':sha(pathlib.Path(__file__))}
dump(O/'result.json',out);print(json.dumps({'status':out['status'],'rows':len(rows),'seconds':out['seconds']}),flush=True)
