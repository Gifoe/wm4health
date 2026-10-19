"""Independent observation-mask/label/checkpoint replay; no scientific refitting."""
import runtime_profile
import hashlib,json,pathlib
import numpy as np
import pandas as pd
import torch
from acquisition_controls import OrdinaryGRU
from residual_controls import ResidualGRU

ROOT=pathlib.Path(__file__).resolve().parents[1];L=96;H=24

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def independent_series(key,meta):
    if key=='ETTh1':
        path=ROOT.parent/'cycle5/data/ETTh1.csv';f=pd.read_csv(path);idx=pd.DatetimeIndex(pd.to_datetime(f['date']));y=f['OT'].to_numpy(float)
    else:
        path=ROOT/'data/hour.csv';f=pd.read_csv(path);idx=pd.DatetimeIndex(pd.to_datetime(f['dteday'])+pd.to_timedelta(f['hr'],unit='h'));y=f['cnt'].to_numpy(float)
    assert sha(path)==meta['source_sha256']
    grid=pd.date_range(idx.min(),idx.max(),freq='h');raw=pd.Series(y,index=idx).reindex(grid).to_numpy(float)
    a=int(.6*len(raw));b=int(.8*len(raw));mean=np.nanmean(raw[:a]);std=np.nanstd(raw[:a])
    assert abs(mean-meta['normalization_mean'])<1e-10 and abs(std-meta['normalization_std'])<1e-10
    assert np.sum(~np.isfinite(raw))==meta['naturally_absent_hours']
    z=((raw-mean)/std).astype(np.float32)
    slot=grid.dayofweek.to_numpy()*24+grid.hour.to_numpy()
    hour=grid.hour.to_numpy()
    cal=np.stack([np.sin(hour*np.pi/12),np.cos(hour*np.pi/12),np.sin(slot*np.pi/84),np.cos(slot*np.pi/84)],-1).astype(np.float32)
    counts=np.convolve(np.isfinite(raw).astype(int),np.ones(L+H,dtype=int),'valid')
    allroots=np.flatnonzero(counts==L+H)+L
    splits={'train':allroots[allroots+H<=a],'val':allroots[(allroots>=a)&(allroots+H<=b)],'test':allroots[allroots>=b]}
    assert {k:len(v) for k,v in splits.items()}==meta['root_counts']
    med=np.array([np.nanmedian(raw[:a][slot[:a]==s]) for s in range(168)])
    return z,cal,splits,(med-mean)/std,slot

def verify_queries(folder,key,meta):
    z,cal,splits,median,slots=independent_series(key,meta);checks=[];calendar_reference={}
    for split in ['val','test']:
        arrays=np.load(folder/f'{key}_{split}_queries.npz',allow_pickle=False)
        groups=sorted({n.split('__')[0] for n in arrays.files})
        expected=splits[split][::24];b=len(expected);rng=np.random.default_rng(9101 if split=='val' else 9102)
        beta0=rng.uniform(-3,3,b).astype(np.float32);u=rng.uniform(size=(b,L)).astype(np.float32)
        for group in groups:
            x=arrays[group+'__features'];r=arrays[group+'__roots'];y=arrays[group+'__target'];fc=arrays[group+'__future_calendar']
            assert np.array_equal(r,expected) and np.all(np.diff(r)>=H)
            hist=z[r[:,None]+np.arange(-L,0)];truth=z[r[:,None]+np.arange(H)]
            assert np.array_equal(y,truth) and np.isfinite(x).all() and np.isfinite(y).all()
            assert np.allclose(x[:,:,6:],cal[r[:,None]+np.arange(-L,0)],atol=2e-7)
            assert np.allclose(fc,cal[r[:,None]+np.arange(H)],atol=2e-7)
            mask=x[:,:,1].astype(bool);assert np.all(mask[:,0])
            assert np.array_equal(x[:,:,0],np.where(mask,hist,0))
            last_index=np.maximum.accumulate(np.where(mask,np.arange(L)[None,:],-1),axis=1)
            fill=np.take_along_axis(hist,last_index,axis=1)
            assert np.array_equal(fill,x[:,:,2])
            gaps=np.log1p(np.arange(L)[None,:]-last_index)/np.log1p(L)
            assert np.allclose(gaps,x[:,:,3],atol=1e-7)
            if group=='fully_observed':assert mask.all() and np.all(x[:,:,4:6]==0)
            else:
                q=int(group[1]);assert np.all(x[:,:,5]==q)
                signs=np.ones(L,np.float32)
                if group.endswith('single_switch'):signs[L//2:]=-1
                elif group.endswith('fast_switch'):signs=(-1.)**(np.arange(L)//8)
                beta=beta0[:,None]*signs[None,:]
                assert np.allclose(beta/3,x[:,:,4],atol=1e-7)
                prior_index=np.concatenate([np.zeros((b,1),int),last_index[:,:-1]],axis=1)
                trigger=hist if q else np.take_along_axis(hist,prior_index,axis=1)
                p=1/(1+np.exp(-beta*np.clip(trigger,-3,3)));m_expected=u<p;m_expected[:,0]=True
                assert np.array_equal(mask,m_expected),'Sampling law mismatch'
            checks.append({'dataset':key,'split':split,'group':group,'roots':len(r),'mask_and_labels_verified':True})
            if split=='test':
                forecast=median[slots[r[:,None]+np.arange(H)]];delta=forecast-y
                calendar_reference[group]={'MAE':float(np.mean(abs(delta))),'MSE':float(np.mean(delta*delta)),
                    'scope':'descriptive training-only hour-of-week median reference,not selected by test score'}
    return checks,calendar_reference

def main():
    torch.set_num_threads(1);allchecks=[];replays=[];calendar={};maxdiff=0.;backend_diagnostics=[];precision_controls=[]
    for family,cls in [('acquisition_controls',OrdinaryGRU),('residual_controls',ResidualGRU)]:
        folder=ROOT/'outputs'/family;result=read(folder/'results.json');assert len(result['rows'])==12
        cfg=result['config'];assert cfg['updates']==2000 and cfg['seeds']==[2021,2022,2023]
        assert cfg['plan_sha256']==sha(ROOT/'ACQUISITION_CONTROL_PLAN.md')
        assert cfg['script_sha256']==sha(ROOT/'scripts/acquisition_controls.py')
        for meta in result['datasets']:
            checks,reference=verify_queries(folder,meta['key'],meta);allchecks.extend(checks);calendar[meta['key']]=reference
        for row in result['rows']:
            key=row['dataset'];stem=f"{key}_{row['arm']}_seed{row['seed']}";model=cls();checkpoint=torch.load(folder/(stem+'.pt'),map_location='cpu',weights_only=True)
            model.load_state_dict(checkpoint['state_dict']);model.eval();assert sum(p.numel() for p in model.parameters())==row['active_parameters']
            curves=read(folder/(stem+'_curves.json'));candidates=curves[1:];chosen=min(candidates,key=lambda r:r['val_source_MAE'])
            assert chosen['update']==row['selected_update']==checkpoint['selected_update']
            assert abs(chosen['val_source_MAE']-row['best_source_val_MAE'])<1e-9
            assert all(np.isfinite(r['val_source_MAE']) for r in curves)
            arrays=np.load(folder/f'{key}_test_queries.npz',allow_pickle=False);stored=np.load(folder/(stem+'_predictions.npz'),allow_pickle=False);precision_checked=False
            for group,metrics in row['tests'].items():
                x=arrays[group+'__features'];fc=arrays[group+'__future_calendar'];y=arrays[group+'__target']
                with torch.no_grad():pred=model(torch.from_numpy(x),torch.from_numpy(fc)).numpy()
                error=float(np.max(abs(pred-stored[group])));maxdiff=max(maxdiff,error)
                if error>=2e-5:
                    record={'stem':stem,'family':family,'group':group,'CPU_max_error':error,
                        'CPU_mean_absolute_difference':float(np.mean(abs(pred-stored[group]))),
                        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_precision':torch.get_float32_matmul_precision()}
                    gpu=cls().cuda();gpu.load_state_dict(checkpoint['state_dict']);gpu.eval();parts=[]
                    with torch.no_grad():
                        for i in range(0,len(y),128):parts.append(gpu(torch.from_numpy(x[i:i+128]).cuda(),torch.from_numpy(fc[i:i+128]).cuda()).cpu().numpy())
                    gp=np.concatenate(parts);gpu_error=float(np.max(abs(gp-stored[group])))
                    record['same_GPU_replay_max_error']=gpu_error;backend_diagnostics.append(record)
                    assert gpu_error<5e-6,'Saved prediction cannot be reproduced on original backend'
                    assert error<.01,'CPU discrepancy requires a separate numerical audit'
                    if not precision_checked:
                        parts=[]
                        with torch.no_grad(),torch.backends.cudnn.flags(allow_tf32=False):
                            for i in range(0,len(y),128):parts.append(gpu(torch.from_numpy(x[i:i+128]).cuda(),torch.from_numpy(fc[i:i+128]).cuda()).cpu().numpy())
                        no_tf32=np.concatenate(parts);control_error=float(np.max(abs(no_tf32-pred)))
                        precision_controls.append({'family':family,'stem':stem,'group':group,'GPU_no_cudnn_TF32_vs_CPU_max_error':control_error})
                        precision_checked=True
                delta=stored[group].astype(float)-y.astype(float)
                assert abs(float(np.mean(abs(delta)))-metrics['MAE'])<3e-7
                assert abs(float(np.mean(delta*delta))-metrics['MSE'])<2e-6
                assert abs(float(x[:,:,1].mean())-metrics['mean_observation_fraction'])<1e-7
            replays.append({'family':family,'dataset':key,'arm':row['arm'],'seed':row['seed'],'selected_update':row['selected_update'],'verified':True})
    for key in ['ETTh1','BikeSharing']:
        for split in ['val','test']:
            a=np.load(ROOT/f'outputs/acquisition_controls/{key}_{split}_queries.npz',allow_pickle=False)
            b=np.load(ROOT/f'outputs/residual_controls/{key}_{split}_queries.npz',allow_pickle=False)
            assert a.files==b.files and all(np.array_equal(a[n],b[n]) for n in a.files)
    pin=read(ROOT/'sources/timeflies/SOURCE_PIN.json')
    for r in pin['files']:assert sha(ROOT/'sources/timeflies'/r['path'])==r['sha256']
    variant=read(ROOT/'outputs/residual_controls/variant.json')
    assert variant['wrapper_sha256']==sha(ROOT/'scripts/residual_controls.py') and not variant['new_method_claimed']
    out={'all_checks_passed':True,'fits_replayed':len(replays),'query_groups_independently_verified':len(allchecks),
        'maximum_CPU_vs_GPU_prediction_error':maxdiff,'query_arrays_identical_across_families':True,
        'sampling_and_no_unobserved_value_leak_checks':allchecks,'checkpoint_replays':replays,
        'calendar_median_reference':calendar,'backend_replay_diagnostics':backend_diagnostics,'precision_controls':precision_controls,
        'maximum_saved_vs_original_GPU_replay_error':max(r['same_GPU_replay_max_error'] for r in backend_diagnostics),
        'maximum_no_cudnn_TF32_vs_CPU_control_error':max(r['GPU_no_cudnn_TF32_vs_CPU_max_error'] for r in precision_controls),
        'verification_scope':'Independent raw-mask/label/source/checkpoint/metric checks. Original GPU replay is authoritative; CPU backend differences are retained explicitly. No training rerun.',
        'objective_achieved':False,'novel_direction_accepted':False}
    (ROOT/'outputs/evidence_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['sampling_and_no_unobserved_value_leak_checks','checkpoint_replays','calendar_median_reference','backend_replay_diagnostics','precision_controls']}))

if __name__=='__main__':main()
