"""Raw-data, split, source, schedule, checkpoint, inference and score audit."""
from native_common33 import *
import time,math
EVAL=ROOT/'outputs/test_evaluation_v1'
def main():
    start=time.perf_counter();source_check()
    er=json.loads((EVAL/'result.json').read_text());tc=json.loads((FITS/'training_complete.json').read_text())
    lock=json.loads((FITS/'source_lock.json').read_text())
    assert sha(FITS/'training_complete.json')==er['training_seal_sha256']
    assert sha(ROOT/'scripts/native_common33.py')==lock['common_sha256']==er['common_sha256']
    assert sha(ROOT/'scripts/train_readouts33.py')==lock['script_sha256']
    assert sha(ROOT/'scripts/evaluate_readouts33.py')==er['evaluation_script_sha256']
    assert sha(ROOT/'NATIVE_READOUT_PLAN.md')==lock['plan_sha256']
    assert len(tc['fits'])==24 and len(er['records'])==56
    checked_scores=0;scoreerror=0.;replayerror=0.;cacheerror=0.;schedulechecks=0
    for ds in ['NP','PJM']:
        cfg=er['datasets'][ds]['config'];meta=er['datasets'][ds]['data']
        # Independent named pivot and train-only normalization.
        frame=pd.read_csv(OLD/'data'/f'{ds}.csv');names=frame['cols'].drop_duplicates().tolist()
        wide=frame.pivot(index='date',columns='cols',values='data')[['OT']+names[:-1]]
        raw=wide.to_numpy(dtype=np.float64);tr=int(int(len(raw)*.8)*.875);tv=int(len(raw)*.8)
        mean=raw[:tr].mean(0);sd=raw[:tr].std(0);values=((raw-mean)/sd).astype(np.float32)
        assert np.allclose([meta['y_mean'][0]]+meta['cov_mean'],mean,rtol=0,atol=1e-12)
        assert np.allclose([meta['y_scale'][0]]+meta['cov_scale'],sd,rtol=0,atol=1e-12)
        expected_o=np.arange(tv,len(raw)-24+1,dtype=np.int64)
        fixture=np.array([0,7,318,len(expected_o)-1]);fo=expected_o[fixture]
        true=np.stack([values[t:t+24,:1] for t in expected_o])
        for file in er['datasets'][ds]['cache_files']:assert sha(EVAL/file['path'])==file['sha256']
        cf=EVAL/ds/'cache';small={name:torch.tensor(np.load(cf/(name+'.npy'),mmap_mode='r')[fixture],device='cuda')
              for name in ['pred','target','prefix','mean','scale','y']}
        # Native fresh source forward from manually normalized raw rows.
        model,_,_,prov=load_native(ds)
        hist=torch.tensor(np.stack([values[t-168:t] for t in fo]),device='cuda')
        future=torch.tensor(np.stack([values[t:t+24,1:] for t in fo]),device='cuda')
        target=torch.tensor(true[fixture],device='cuda')
        with torch.inference_mode():out=model(hist,future_target=target,x_future=future)
        fresh={'pred':out['h_pred'],'target':out['h_target'],'prefix':out['h_hist'][:,-(cfg['decoder_patch_len']-1):],
               'mean':out['target_revin_stats']['mean'],'scale':out['target_revin_stats']['stdev'],'y':target}
        for name in fresh:
            error=float((fresh[name]-small[name]).abs().max());cacheerror=max(cacheerror,error)
            assert error/(1+float(fresh[name].abs().max()))<2e-5,(ds,name,error)
        del model;torch.cuda.empty_cache()
        cm=json.loads((CACHE/ds/'manifest.json').read_text())
        for row in cm['files']:assert sha(CACHE/ds/row['path'])==row['sha256']
        trainorig=np.load(CACHE/ds/'train/origins.npy');valor=np.load(CACHE/ds/'val/origins.npy')
        assert np.array_equal(trainorig,np.arange(168,tr-24+1))
        assert np.array_equal(valor,np.arange(tr,tv-24+1))
        assert trainorig[-1]+24<=tr and valor[-1]+24<=tv
        for s in SEEDS:
            orders=np.load(FITS/ds/f'orders_seed{s}.npy')
            for e in range(20):
                expected=np.random.default_rng(s+733000+e).permutation(len(trainorig)).astype(np.int32)
                assert np.array_equal(orders[e],expected);schedulechecks+=1
            nh=[]
            for arm in ARMS:
                folder=FITS/ds/f'{arm}_seed{s}';r=json.loads((folder/'result.json').read_text());h=json.loads((folder/'history.json').read_text())
                assert len(h)==21 and r['updates']==20*math.ceil(len(trainorig)/32)
                assert r['order_sha256']==sha(FITS/ds/f'orders_seed{s}.npy')
                assert r['history_sha256']==sha(folder/'history.json')
                if arm.startswith('native_'):nh.append(r['initial_state_sha256'])
                for e,row in enumerate(h):
                    assert row['updates']==e*math.ceil(len(trainorig)/32)
                    expectedlr=r['base_lr'] if e<=4 else r['base_lr']*.9**(e-4)
                    assert abs(row['lr']-expectedlr)<1e-14
                assert r['best_epoch']==min(range(21),key=lambda i:h[i]['forecast_mae'])
                assert r['best_objective_epoch']==min(range(21),key=lambda i:h[i]['selection_objective_secondary'])
                for name,key in [('best_forecast.pt','best_forecast_sha256'),('best_objective.pt','best_objective_sha256'),('last.pt','last_sha256')]:
                    assert sha(folder/name)==r[key]
                ck=torch.load(folder/'best_forecast.pt',map_location='cpu',weights_only=True)
                seed(s)
                gs=(ck['state_dict']['feature_mean'].cuda(),ck['state_dict']['feature_scale'].cuda()) if arm=='global_pred' else None
                head=make_head(cfg,arm,global_stats=gs)
                assert state_sha(head.state_dict())==r['initial_state_sha256']
                assert sum(p.numel() for p in head.parameters())==r['parameters']
                head.load_state_dict(ck['state_dict'],strict=True);head.eval()
                with torch.inference_mode():replay=forward_head(head,small,slice(None)).cpu().numpy()
                record=next(row for row in er['records'] if row['dataset']==ds and row['method']==arm and row['seed']==s and row['selection']=='forecast')
                saved=np.load(EVAL/record['path'])['pred'][fixture]
                err=float(np.abs(saved-replay).max());replayerror=max(replayerror,err)
                assert err/(1+np.abs(saved).max())<2e-5
                del head;torch.cuda.empty_cache()
            assert nh[0]==nh[1]
        for row in [r for r in er['records'] if r['dataset']==ds]:
            path=EVAL/row['path'];assert sha(path)==row['sha256'];a=np.load(path)
            assert np.array_equal(a['origins'],expected_o) and np.array_equal(a['true'],true)
            assert np.isfinite(a['pred']).all()
            error=a['pred'].astype(np.float64)-true.astype(np.float64)
            for name,val in [('mae',np.abs(error).mean()),('mse',np.square(error).mean())]:
                diff=abs(float(val)-row[name]);scoreerror=max(scoreerror,diff);assert diff<1e-12;checked_scores+=1
            assert np.allclose(np.abs(error).mean((0,2)),row['per_horizon_mae'],rtol=0,atol=1e-12)
            if 'oracle_target_code_decode' in a:
                err=np.abs(a['oracle_target_code_decode'].astype(np.float64)-true).mean()
                assert abs(float(err)-row['oracle_target_code_reconstruction_mae'])<1e-12
        del small;torch.cuda.empty_cache()
    result={'status':'passed','evaluation_sha256':sha(EVAL/'result.json'),'training_seal_sha256':sha(FITS/'training_complete.json'),
      'verifier_sha256':sha(__file__),'score_checks':checked_scores,'schedule_checks':schedulechecks,
      'all_24_initializations_and_selections_checked':True,'independent_raw_data_and_cache_checked':True,
      'maximum_metric_abs_error':scoreerror,'maximum_cached_native_fixture_abs_error':cacheerror,
      'maximum_selected_head_fixture_abs_error':replayerror,'seconds':time.perf_counter()-start,
      'scope':'old real-data frozen-readout attribution; not novel method or independent confirmation'}
    assert not (EVAL/'verification.json').exists();dump(EVAL/'verification.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
