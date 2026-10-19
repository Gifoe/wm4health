"""Frozen ordinary controls for a cross-system observation-policy audit."""
import runtime_profile
import copy, datetime, hashlib, json, pathlib, sys, time, traceback
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/acquisition_controls'
L=96; H=24

def dump(path, data):
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')

def observed_features(hist, calendar, beta, q, uniform, fully=None):
    """MAR uses only the last revealed value; MNAR uses current sensor value."""
    b,n=hist.shape; assert n==L and calendar.shape==(b,L,4)
    masks=np.empty((b,L),np.float32);filled=np.empty_like(masks);gaps=np.empty_like(masks)
    last=hist[:,0].copy();gap=np.zeros(b,np.float32)
    if fully is None:fully=np.zeros(b,dtype=bool)
    for t in range(L):
        trigger=np.where(q>0.5,hist[:,t],last)
        p=1/(1+np.exp(-beta[:,t]*np.clip(trigger,-3,3)))
        m=(uniform[:,t]<p)|fully
        if t==0:m[:]=True
        last=np.where(m,hist[:,t],last);gap=np.where(m,0,gap+1)
        masks[:,t]=m;filled[:,t]=last;gaps[:,t]=np.log1p(gap)/np.log1p(L)
    features=np.concatenate([np.stack([hist*masks,masks,filled,gaps,beta/3,np.broadcast_to(q[:,None],hist.shape)],-1),calendar],-1).astype(np.float32)
    assert np.isfinite(features).all()
    return features

def load_series(key):
    if key=='ETTh1':
        path=ROOT.parent/'cycle5/data/ETTh1.csv';frame=pd.read_csv(path)
        dates=pd.to_datetime(frame['date']);values=frame['OT'].to_numpy(float)
    elif key=='BikeSharing':
        path=ROOT/'data/hour.csv';frame=pd.read_csv(path)
        dates=pd.to_datetime(frame['dteday'])+pd.to_timedelta(frame['hr'],unit='h');values=frame['cnt'].to_numpy(float)
    else:raise ValueError(key)
    index=pd.DatetimeIndex(dates);assert index.is_unique and index.is_monotonic_increasing
    full=pd.date_range(index[0],index[-1],freq='h');series=pd.Series(values,index=index).reindex(full)
    raw=series.to_numpy(float);a=int(len(raw)*.6);b=int(len(raw)*.8)
    mean=float(np.nanmean(raw[:a]));std=float(np.nanstd(raw[:a]));assert std>0
    scaled=((raw-mean)/std).astype(np.float32)
    hour=full.hour.to_numpy();week=full.dayofweek.to_numpy()*24+hour
    cal=np.stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*week/168),np.cos(2*np.pi*week/168)],-1).astype(np.float32)
    valid=np.isfinite(scaled);prefix=np.concatenate([[0],np.cumsum(~valid)])
    roots=np.arange(L,len(raw)-H+1);roots=roots[(prefix[roots+H]-prefix[roots-L])==0]
    splits={'train':roots[roots+H<=a],'val':roots[(roots>=a)&(roots+H<=b)],'test':roots[roots>=b]}
    assert all(len(r)>100 for r in splits.values())
    meta={'key':key,'source_bytes':path.stat().st_size,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'full_hour_grid_length':len(raw),'original_rows':len(values),'naturally_absent_hours':int((~valid).sum()),
          'normalization_mean':mean,'normalization_std':std,'normalization_rows':'first60percent only',
          'train_end_index':a,'val_end_index':b,'root_counts':{k:len(v) for k,v in splits.items()},
          'discard_rule':'any originally absent hour in historical96/future24 window',
          'date_timezone':'source naive calendar; no UTC or physical elapsed-time claim'}
    return scaled,cal,splits,meta

def gather(values,calendar,roots):
    hidx=roots[:,None]+np.arange(-L,0)[None,:];fidx=roots[:,None]+np.arange(H)[None,:]
    hist=values[hidx];return hist,calendar[hidx],calendar[fidx],values[fidx]

def make_queries(values,calendar,roots,seed,full_grid):
    roots=roots[::24];hist,hcal,fcal,y=gather(values,calendar,roots)
    rng=np.random.default_rng(seed);base=rng.uniform(-3,3,len(roots)).astype(np.float32);u=rng.uniform(size=hist.shape).astype(np.float32)
    queries={}
    for q in [0,1]:
        for schedule in (['constant','single_switch','fast_switch'] if full_grid else ['constant']):
            sign=np.ones(L,np.float32)
            if schedule=='single_switch':sign[L//2:]=-1
            elif schedule=='fast_switch':sign=(-1.)**(np.arange(L)//8)
            beta=base[:,None]*sign[None,:]
            feat=observed_features(hist,hcal,beta,np.full(len(roots),q,np.float32),u)
            queries[f'q{q}_{schedule}']={'features':feat,'future_calendar':fcal,'target':y,'roots':roots,
                'metadata':{'q':q,'schedule':schedule,'mean_observation_fraction':float(feat[:,:,1].mean()),'roots':len(roots)}}
    feat=observed_features(hist,hcal,np.zeros_like(hist),np.zeros(len(roots),np.float32),u,fully=np.ones(len(roots),bool))
    queries['fully_observed']={'features':feat,'future_calendar':fcal,'target':y,'roots':roots,
        'metadata':{'q':0,'schedule':'full','mean_observation_fraction':1.,'roots':len(roots)}}
    return queries

class OrdinaryGRU(nn.Module):
    def __init__(self):
        super().__init__();self.rnn=nn.GRU(10,32,num_layers=2,batch_first=True)
        self.head=nn.Sequential(nn.Linear(32+H*4,64),nn.SiLU(),nn.Linear(64,H))
    def forward(self,x,fc):
        _,h=self.rnn(x);return self.head(torch.cat([h[-1],fc.flatten(1)],-1))

def score(model,queries,save_predictions=False):
    model.eval();rows={};predictions={}
    with torch.no_grad():
        for key,q in queries.items():
            preds=[]
            for i in range(0,len(q['target']),128):
                x=torch.from_numpy(q['features'][i:i+128]).cuda();fc=torch.from_numpy(q['future_calendar'][i:i+128]).cuda()
                preds.append(model(x,fc).cpu().numpy())
            pred=np.concatenate(preds);delta=pred-q['target'];assert np.isfinite(pred).all()
            rows[key]={**q['metadata'],'MAE':float(np.mean(np.abs(delta))),'MSE':float(np.mean(delta**2))}
            if save_predictions:predictions[key]=pred
    return rows,predictions

def validation_score(rows):
    return .4*rows['q0_constant']['MAE']+.4*rows['q1_constant']['MAE']+.2*rows['fully_observed']['MAE']

def train_one(key,arm,seed,values,calendar,splits,val,test,config):
    begin=time.perf_counter();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=OrdinaryGRU().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    rng=np.random.default_rng(7000+seed);best=float('inf');best_weights=None;curves=[]
    initial,_=score(model,val);curves.append({'update':0,'val_source_MAE':validation_score(initial)})
    for step in range(1,config['updates']+1):
        model.train();roots=rng.choice(splits['train'],config['batch'],replace=True)
        hist,hcal,fcal,y=gather(values,calendar,roots);b=len(roots)
        proposals=rng.uniform(-3,3,(b,L)).astype(np.float32);q=rng.integers(0,2,b).astype(np.float32)
        seg=int(rng.choice([16,24,32,48]));fully=rng.uniform(size=b)<.2;u=rng.uniform(size=hist.shape).astype(np.float32)
        beta=proposals[:,np.arange(L)//seg] if arm=='augmented' else np.repeat(proposals[:,:1],L,axis=1)
        beta=beta.copy();beta[fully]=0
        feat=observed_features(hist,hcal,beta,q,u,fully)
        optimizer.zero_grad(set_to_none=True);pred=model(torch.from_numpy(feat).cuda(),torch.from_numpy(fcal).cuda())
        loss=torch.mean(torch.abs(pred-torch.from_numpy(y).cuda()));assert torch.isfinite(loss)
        loss.backward();grad=float(nn.utils.clip_grad_norm_(model.parameters(),1.));assert np.isfinite(grad);optimizer.step()
        if step%200==0:
            rows,_=score(model,val);v=validation_score(rows);curves.append({'update':step,'train_batch_MAE':float(loss.detach().cpu()),'val_source_MAE':v,'gradient_norm_before_clip':grad})
            if v<best:best=v;best_weights=copy.deepcopy({k:t.detach().cpu() for k,t in model.state_dict().items()});selected=step
            print(json.dumps({'dataset':key,'arm':arm,'seed':seed,'update':step,'val_MAE':v,'seconds':time.perf_counter()-begin}),flush=True)
    stem=f'{key}_{arm}_seed{seed}'
    torch.save({'state_dict':{k:t.detach().cpu() for k,t in model.state_dict().items()},'config':config},OUT/(stem+'.last.pt'))
    model.load_state_dict(best_weights);tests,preds=score(model,test,True)
    torch.save({'state_dict':best_weights,'config':config,'selected_update':selected,'best_source_val_MAE':best},OUT/(stem+'.pt'))
    np.savez_compressed(OUT/(stem+'_predictions.npz'),**preds)
    dump(OUT/(stem+'_curves.json'),curves)
    result={'dataset':key,'arm':arm,'seed':seed,'active_parameters':sum(p.numel() for p in model.parameters()),
            'best_source_val_MAE':best,'selected_update':selected,'initial_val_MAE':curves[0]['val_source_MAE'],
            'last_val_MAE':curves[-1]['val_source_MAE'],'elapsed_seconds':time.perf_counter()-begin,'tests':tests,
            'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated()}
    print(json.dumps({'completed':stem,'selected_update':selected,'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    return result

def sanity():
    rng=np.random.default_rng(199);hist=rng.normal(size=(7,L)).astype(np.float32);cal=np.zeros((7,L,4),np.float32)
    beta=np.ones_like(hist)*2;u=rng.uniform(size=hist.shape);q=np.zeros(7,np.float32)
    a=observed_features(hist,cal,beta,q,u);changed=hist.copy();changed[a[:,:,1]==0]+=100
    b=observed_features(changed,cal,beta,q,u)
    assert np.array_equal(a,b),'MAR leaked a hidden current value'
    assert np.all(a[:,0,1]==1) and np.all(a[:,:,3]>=0)
    full=observed_features(hist,cal,beta,q,u,fully=np.ones(7,bool));assert np.array_equal(full[:,:,0],hist) and np.all(full[:,:,1]==1)
    return {'MAR_unobserved_value_invariance':True,'first_value_revealed':True,'fully_observed_identity':True,
            'future_values_not_arguments_to_observation_features':True}

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'results.json').exists(),'Completed experiment must not rerun'
    assert torch.cuda.is_available();torch.set_num_threads(1)
    config={'datasets':['ETTh1','BikeSharing'],'arms':['source','augmented'],'seeds':[2021,2022,2023],
        'updates':2000,'batch':64,'lr':.001,'L':L,'H':H,'hidden':32,'layers':2,'checkpoint_selection':'best common source-policy validation MAE, every200updates',
        'validation_weights':{'q0_constant':.4,'q1_constant':.4,'fully_observed':.2},'validation_seed':9101,'test_seed':9102,
        'test_root_stride':24,'scope':'ordinary controls on controlled missing-context transfer; no new method or original-paper reproduction',
        'plan_sha256':hashlib.sha256((ROOT/'ACQUISITION_CONTROL_PLAN.md').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    dump(OUT/'config.json',config);dump(OUT/'sanity.json',sanity())
    dump(OUT/'environment.json',{'python_torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
        'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'threads':torch.get_num_threads(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    results=[];datasets=[];baselines={};begin=time.perf_counter()
    for key in config['datasets']:
        values,cal,splits,meta=load_series(key);datasets.append(meta)
        val=make_queries(values,cal,splits['val'],9101,False);test=make_queries(values,cal,splits['test'],9102,True)
        for split,queries in [('val',val),('test',test)]:
            arrays={}
            for group,q in queries.items():
                for name in ['features','future_calendar','target','roots']:arrays[group+'__'+name]=q[name]
            np.savez_compressed(OUT/f'{key}_{split}_queries.npz',**arrays)
        baselines[key]={}
        for group,q in test.items():
            reference={}
            for name,p in [('persistence',np.repeat(q['features'][:,-1,2,None],H,axis=1)),('past_day',q['features'][:,-H:,2])]:
                d=p-q['target'];reference[name]={'MAE':float(np.mean(abs(d))),'MSE':float(np.mean(d*d))}
            baselines[key][group]=reference
        dump(OUT/'datasets.json',datasets)
        for seed in config['seeds']:
            for arm in config['arms']:
                row=train_one(key,arm,seed,values,cal,splits,val,test,config);results.append(row)
                dump(OUT/'partial_results.json',{'config':config,'rows':results,'datasets':datasets,'reference_controls':baselines})
    report={'config':config,'rows':results,'datasets':datasets,'reference_controls':baselines,'elapsed_seconds':time.perf_counter()-begin,
            'scientific_jobs_complete':True,'objective_achieved':False,'novel_direction_accepted':False}
    dump(OUT/'results.json',report);print(json.dumps({'fits_complete':len(results),'elapsed_seconds':report['elapsed_seconds'],'objective_achieved':False}),flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    class Tee:
        def __init__(self,original,log):self.original=original;self.log=log
        def write(self,value):self.original.write(value);self.log.write(value)
        def flush(self):self.original.flush();self.log.flush()
    with (OUT/'stdout.log').open('x',encoding='utf-8') as log:
        sys.stdout=Tee(sys.__stdout__,log)
        try:main()
        except BaseException:
            with (OUT/'stderr.log').open('a',encoding='utf-8') as err:traceback.print_exc(file=err)
            raise
        finally:sys.stdout.flush();sys.stdout=sys.__stdout__
