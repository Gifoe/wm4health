"""Load TEST only after the entire fixed training matrix is sealed."""
from native_common33 import *
import time
EVAL=ROOT/'outputs/test_evaluation_v1'

def metric(pred,true):
    e=pred.astype(np.float64)-true.astype(np.float64)
    return {'mae':float(np.abs(e).mean()),'mse':float((e*e).mean()),
            'per_horizon_mae':np.abs(e).mean((0,2)).tolist()}

def predict(head,pack,kind='pred'):
    rows=[];head.eval()
    with torch.inference_mode():
        for b in range(0,len(pack['y']),512):rows.append(forward_head(head,pack,slice(b,b+512),kind=kind).cpu().numpy())
    out=np.concatenate(rows);assert np.isfinite(out).all();return out

def main():
    assert not EVAL.exists();EVAL.mkdir(parents=True)
    start=time.perf_counter();source_check()
    sealed=FITS/'training_complete.json';completion=json.loads(sealed.read_text())
    assert completion['status']=='complete_all_24_fits_test_not_loaded' and len(completion['fits'])==24
    lock=json.loads((FITS/'source_lock.json').read_text())
    assert sha(ROOT/'scripts/native_common33.py')==lock['common_sha256']
    assert completion['source_lock_sha256']==sha(FITS/'source_lock.json')
    for fit in completion['fits']:
        f=FITS/fit['dataset']/f"{fit['arm']}_seed{fit['seed']}"
        assert sha(f/'best_forecast.pt')==fit['best_forecast_sha256']
        assert sha(f/'best_objective.pt')==fit['best_objective_sha256']
        assert fit['test_not_loaded']
    records=[];testmeta={}
    def add(ds,method,selection,seedvalue,pred,true,origins,oracle=None,extra=None):
        name=f'{method}_seed{seedvalue}_{selection}.npz';path=EVAL/ds/name
        payload={'pred':pred,'true':true,'origins':origins}
        if oracle is not None:payload['oracle_target_code_decode']=oracle
        np.savez_compressed(path,**payload)
        row={'dataset':ds,'method':method,'selection':selection,'seed':seedvalue,'windows':len(origins),
              'path':path.relative_to(EVAL).as_posix(),'sha256':sha(path),**metric(pred,true)}
        if oracle is not None:row['oracle_target_code_reconstruction_mae']=metric(oracle,true)['mae']
        if extra:row.update(extra)
        records.append(row)
    for ds in ['NP','PJM']:
        model,cfg,original,provenance=load_native(ds);values,origins,data=load_data(ds)
        o=origins['test'];n=len(o);d=cfg['latent_dim'];p=cfg['decoder_patch_len'];folder=EVAL/ds/'cache';folder.mkdir(parents=True)
        shapes={'pred':(n,24,d),'target':(n,24,d),'prefix':(n,p-1,d),'mean':(n,1,1),'scale':(n,1,1),'y':(n,24,1)}
        maps={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=np.float32,shape=s) for k,s in shapes.items()}
        np.save(folder/'origins.npy',o)
        with torch.inference_mode():
            for b in range(0,n,128):
                ix=o[b:b+128]
                hist=torch.tensor(np.stack([values[t-168:t] for t in ix]),device='cuda')
                future=torch.tensor(np.stack([values[t:t+24,1:] for t in ix]),device='cuda')
                y=torch.tensor(np.stack([values[t:t+24,:1] for t in ix]),device='cuda')
                out=model(hist,future_target=y,x_future=future);stats=out['target_revin_stats']
                vals={'pred':out['h_pred'],'target':out['h_target'],'prefix':out['h_hist'][:,-(p-1):],
                      'mean':stats['mean'],'scale':stats['stdev'],'y':y}
                for k,x in vals.items():assert torch.isfinite(x).all();maps[k][b:b+len(ix)]=x.cpu().numpy()
        for mm in maps.values():mm.flush()
        del maps
        pack={k:torch.tensor(np.load(folder/(k+'.npy'),mmap_mode='r'),device='cuda') for k in shapes}
        true=pack['y'].cpu().numpy()
        reference=make_head(cfg,'native_pred');reference.load_state_dict({k[len('decoder.'):]:t for k,t in original.items() if k.startswith('decoder.')},strict=True)
        pred=predict(reference,pack);oracle=predict(reference,pack,kind='target')
        old=np.load(OLD/'results/worldts'/f'{ds}_H24_seed2021/predictions.npz')
        assert np.array_equal(old['origins'],o) and np.array_equal(old['true'],true)
        oldmetrics=metric(old['pred'],old['true']);nowmetrics=metric(pred,true)
        metricdiff=max(abs(oldmetrics[k]-nowmetrics[k]) for k in ['mae','mse']);assert metricdiff<2e-5
        # Fresh full-model forecast route with the ORIGINAL complete stage2
        # checkpoint, compared to the separated cached decoder route.
        model.load_state_dict(original,strict=True);model.eval()
        fixtures=np.array([0,7,318,n-1]);ff=o[fixtures]
        hist=torch.tensor(np.stack([values[t-168:t] for t in ff]),device='cuda')
        fut=torch.tensor(np.stack([values[t:t+24,1:] for t in ff]),device='cuda')
        with torch.inference_mode():direct=model.forecast(hist,x_future=fut).cpu().numpy()
        directdiff=float(np.abs(direct-pred[fixtures]).max())
        assert directdiff/(1+np.abs(direct).max())<2e-5
        add(ds,'original_stage2_reference','original',2021,pred,true,o,oracle,{'reference_old_metric_max_abs':metricdiff,
            'cached_vs_full_forecast_fixture_max_abs':directdiff,'checkpoint_sha256':provenance['stage2_sha256']})
        simple={'last_value':np.repeat(values[o-1,:1][:,None,:],24,axis=1),
                'seasonal_24':np.stack([values[t-24:t,:1] for t in o]),
                'seasonal_168':np.stack([values[t+np.arange(24)-168,:1] for t in o])}
        for name,out in simple.items():add(ds,name,'fixed',0,out,true,o)
        del model,reference;torch.cuda.empty_cache()
        for fit in [row for row in completion['fits'] if row['dataset']==ds]:
            arm=fit['arm'];s=fit['seed'];fitfolder=FITS/ds/f'{arm}_seed{s}'
            for selection,checkpoint in [('forecast','best_forecast.pt'),('objective','best_objective.pt')]:
                ck=torch.load(fitfolder/checkpoint,map_location='cpu',weights_only=True)
                head=make_head(cfg,arm);head.load_state_dict(ck['state_dict'],strict=True)
                pred=predict(head,pack);oracle=predict(head,pack,kind='target')
                add(ds,arm,selection,s,pred,true,o,oracle,{'checkpoint_sha256':sha(fitfolder/checkpoint),
                    'selected_epoch':ck['meta']['selected_epoch'],'parameters':fit['parameters'],
                    'fit_seconds':fit['seconds'],'cap_still_improving':fit['cap_still_improving']})
                del head;torch.cuda.empty_cache()
        cachefiles=[{'path':f.relative_to(EVAL).as_posix(),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(folder.glob('*.npy'))]
        testmeta[ds]={'config':cfg,'data':data,'cache_files':cachefiles,'provenance':provenance}
        del pack;torch.cuda.empty_cache()
    assert len(records)==56
    result={'status':'complete_pending_independent_verification','records':records,'datasets':testmeta,
       'training_seal_sha256':sha(sealed),'source_lock_sha256':sha(FITS/'source_lock.json'),
       'evaluation_script_sha256':sha(__file__),'common_sha256':sha(ROOT/'scripts/native_common33.py'),
       'test_generated_only_after_all_training_sealed':True,'seconds':time.perf_counter()-start,
       'interpretation':'ordinary frozen readout attribution on previously studied real data; not new model or fresh independent cohort'}
    dump(EVAL/'result.json',result)
    print(json.dumps({'status':result['status'],'records':len(records),'seconds':result['seconds']}),flush=True)
if __name__=='__main__':main()
