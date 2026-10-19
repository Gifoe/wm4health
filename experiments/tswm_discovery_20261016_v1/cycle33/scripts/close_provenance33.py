"""Supplementary immutable provenance closure, not an experimental change."""
from native_common33 import *
import time
EVAL=ROOT/'outputs/test_evaluation_v1'

def main():
    start=time.perf_counter();path=EVAL/'provenance_closure.json';assert not path.exists()
    proof=json.loads((EVAL/'verification.json').read_text());assert proof['status']=='passed'
    er=json.loads((EVAL/'result.json').read_text());tc=json.loads((FITS/'training_complete.json').read_text())
    lock=json.loads((FITS/'source_lock.json').read_text());complete=json.loads((CACHE/'complete.json').read_text())
    assert sha(CACHE/'complete.json')==lock['cache_complete_sha256']
    assert sha(ROOT/'frozen_source_manifest.json')==lock['source_manifest_sha256']
    assert sha(ROOT/'scripts/cache_native33.py')==complete['cache_script_sha256']
    assert sha(FITS/'source_lock.json')==tc['source_lock_sha256']==er['source_lock_sha256']
    checks=[];maximum=0.;replayed=0
    for ds in ['NP','PJM']:
        cm=json.loads((CACHE/ds/'manifest.json').read_text())
        assert cm==complete['datasets'][ds]
        assert cm['data']==er['datasets'][ds]['data']
        assert cm['config']==er['datasets'][ds]['config']
        assert cm['provenance']==er['datasets'][ds]['provenance']
        model,cfg,_,prov=load_native(ds);values,origins,data=load_data(ds)
        assert prov==cm['provenance'] and data==cm['data']
        assert state_sha(model.state_dict())==cm['frozen_state_before']==cm['frozen_state_after']
        for split in ['train','val']:
            folder=CACHE/ds/split;o=origins[split]
            y=np.load(folder/'y.npy',mmap_mode='r')
            assert np.array_equal(y,np.stack([values[t:t+24,:1] for t in o]))
            fixtures=np.array([0,7,173,len(o)//2,len(o)-1]);fo=o[fixtures]
            hist=torch.tensor(np.stack([values[t-168:t] for t in fo]),device='cuda')
            future=torch.tensor(np.stack([values[t:t+24,1:] for t in fo]),device='cuda')
            target=torch.tensor(y[fixtures],device='cuda')
            with torch.inference_mode():out=model(hist,future_target=target,x_future=future)
            fresh={'pred':out['h_pred'],'target':out['h_target'],'prefix':out['h_hist'][:,-(cfg['decoder_patch_len']-1):],
                   'mean':out['target_revin_stats']['mean'],'scale':out['target_revin_stats']['stdev']}
            for name,arr in fresh.items():
                saved=np.load(folder/(name+'.npy'),mmap_mode='r')[fixtures];now=arr.cpu().numpy()
                error=float(np.abs(now-saved).max());maximum=max(maximum,error)
                assert error/(1+np.abs(now).max())<2e-5
                replayed+=1
        del model;torch.cuda.empty_cache();checks.append(ds)
    for fit in tc['fits']:
        folder=FITS/fit['dataset']/f"{fit['arm']}_seed{fit['seed']}"
        assert json.loads((folder/'result.json').read_text())==fit
        assert fit['source_lock_sha256']==sha(FITS/'source_lock.json')
        hist=json.loads((folder/'history.json').read_text());tail=[r['forecast_mae'] for r in hist[-5:]]
        assert fit['cap_still_improving']==(min(tail[1:])<tail[0]*.995)
        for selection,filename in [('forecast','best_forecast.pt'),('objective','best_objective.pt')]:
            ck=torch.load(folder/filename,map_location='cpu',weights_only=True)
            row=next(r for r in er['records'] if r['dataset']==fit['dataset'] and r['method']==fit['arm'] and
                r['seed']==fit['seed'] and r['selection']==selection)
            assert row['checkpoint_sha256']==sha(folder/filename)
            assert ck['meta']['dataset']==fit['dataset'] and ck['meta']['arm']==fit['arm'] and ck['meta']['seed']==fit['seed']
            assert ck['meta']['source_lock_sha256']==fit['source_lock_sha256']
            assert row['selected_epoch']==ck['meta']['selected_epoch']
    result={'status':'passed','script_sha256':sha(__file__),'evaluation_sha256':sha(EVAL/'result.json'),
       'verification_sha256':sha(EVAL/'verification.json'),'training_seal_sha256':sha(FITS/'training_complete.json'),
       'train_test_old_weight_data_config_provenance_equal':checks,'trainval_feature_fixtures_checked':replayed,
       'maximum_trainval_feature_fixture_abs_error':maximum,'all_24_fit_metadata_and_48_selection_records_equal':True,
       'all_trainval_cached_targets_checked_against_raw':True,'seconds':time.perf_counter()-start}
    dump(path,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
