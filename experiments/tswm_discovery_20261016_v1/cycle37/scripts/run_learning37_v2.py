"""Registered ordinary-control learning experiment; no new-method or test claims."""
from common37_v2 import *
import time,datetime,traceback,joblib
from sklearn.ensemble import HistGradientBoostingRegressor
O=ROOT/'outputs/learning_v2'
assert not O.exists(),'Preserve completed or running experiment'
pre=json.loads((ROOT/'outputs/preflight_v2/result.json').read_text(encoding='utf-8'))
assert pre['passed'] and pre['common_sha256']==sha(ROOT/'scripts/common37_v2.py') and pre['protocol_sha256']==sha(ROOT/'LEARNING_PROTOCOL.md')
O.mkdir(parents=True);rows=[];initials={};started=time.monotonic()
def status(**kw):
    rec={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'worker_pid':os.getpid(),'completed_rows':len(rows),**kw}
    tmp=O/'status.tmp';tmp.write_text(json.dumps(rec,indent=2),encoding='utf-8');tmp.replace(O/'status.json')
def record(folder,arm,samples,y,origins):
    assert samples.shape==(len(y),128,H) and np.isfinite(samples).all()
    sc=scores(samples,y);summ={k:float(v.mean()) for k,v in sc.items()}
    np.savez_compressed(folder/(arm+'.npz'),samples=samples,targets=y,origins=origins,**{'score_'+k:v for k,v in sc.items()})
    clipped=summary(np.clip(samples,0,1),y)
    item={'dataset':folder.parent.name,'fit':folder.name,'arm':arm,'summary':summ,'clipped_sensitivity':clipped}
    rows.append(item);return item
try:
    for name in ['wind','solar']:
        groups,meta=dataset(name);fit,cal,val=groups['fit'],groups['cal'],groups['val']
        D=O/name;D.mkdir();(D/'data.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        np.savez_compressed(D/'registered_origins.npz',**{k:groups[k]['origins'] for k in groups})
        nearest=neighbours(cal,val);np.save(D/'cal_analogue_indices.npy',nearest)
        for sd in SEEDS:
            schedule=np.random.default_rng(sd).integers(len(fit['y']),size=(STEPS,BATCH),dtype=np.int32)
            np.save(D/f'schedule_{sd}.npy',schedule)
            for kind in ['quantile_mlp','kite_l1','kite_l2']:
                F=D/f'{kind}_{sd}';F.mkdir();seed(sd);model=make_model(kind,fit)
                initial=state_sha(model)
                if kind=='kite_l1':initials[(name,sd)]=initial
                if kind=='kite_l2':assert initial==initials[(name,sd)]
                optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001,foreach=False)
                gt=tensors(fit);plan=torch.as_tensor(schedule.astype('int64'),device='cuda')
                curve=[];best=float('inf');best_step=None;fit_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                def validate(step):
                    samples=predict(model,kind,val,meta,n=64,rng_seed=sd+80000)
                    return float(fair_crps(samples,val['y']).mean())
                for step in range(STEPS+1):
                    if step>0:
                        seed(sd+1000000+step);model.train();optimizer.zero_grad(set_to_none=True)
                        v=loss(model,kind,gt,plan[step-1]);assert torch.isfinite(v),(name,kind,step)
                        v.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        assert torch.isfinite(grad);optimizer.step()
                    if step%500==0:
                        value=validate(step);entry={'step':step,'val_crps':value,'elapsed_seconds':time.monotonic()-fit_start,
                                                'last_loss':None if step==0 else float(v.detach().cpu())}
                        curve.append(entry)
                        if value<best:
                            best=value;best_step=step
                            torch.save({'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'step':step},F/'selected.pt')
                        (F/'curve.json').write_text(json.dumps(curve,indent=2),encoding='utf-8')
                        status(phase='training',dataset=name,model=kind,seed=sd,step=step,best_step=best_step,best_val_crps=best)
                        print(json.dumps({'dataset':name,'model':kind,'seed':sd,**entry,'best_step':best_step}),flush=True)
                checkpoint=torch.load(F/'selected.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state']);model.eval()
                selected_hash=state_sha(model);torch.cuda.synchronize();sampling_start=time.perf_counter()
                vs=predict(model,kind,val,meta,n=128,rng_seed=sd+90000)
                torch.cuda.synchronize();sample_seconds=time.perf_counter()-sampling_start
                cs=predict(model,kind,cal,meta,n=128,rng_seed=sd+91000)
                np.savez_compressed(F/'calibration.npz',samples=cs,targets=cal['y'],origins=cal['origins'])
                if kind=='quantile_mlp':
                    vq=predict(model,kind,val,meta,return_quantiles=True);cq=predict(model,kind,cal,meta,return_quantiles=True)
                    variants=quantile_paths(vq,cq,cal['y'],nearest,n=128,rng_seed=sd+92000)
                    proof=max(float(np.max(np.abs(np.sort(x,axis=1)-np.sort(vs,axis=1)))) for x in variants.values())
                    assert proof==0
                else:variants={'native':vs};proof=None
                for arm,arr in variants.items():
                    record(F,arm,arr,val['y'],val['origins']);corrected,offset=calibrate(arr,cs,cal['y'])
                    record(F,arm+'_calibrated',corrected,val['y'],val['origins']);np.save(F/(arm+'_calibration_offset.npy'),offset)
                if kind!='quantile_mlp':
                    model.flow_matching.num_sampling_steps=50
                    higher=predict(model,kind,val,meta,n=128,rng_seed=sd+90000)
                    record(F,'native_50steps',higher,val['y'],val['origins']);model.flow_matching.num_sampling_steps=10
                assert state_sha(model)==selected_hash
                early=min(x['val_crps'] for x in curve if x['step']<=3000)
                cap=(early-best)/max(abs(early),1e-12)>.01
                info={'dataset':name,'kind':kind,'seed':sd,'updates':STEPS,'batch_size':BATCH,'initial_state_sha256':initial,
                    'selected_state_sha256':selected_hash,'checkpoint_sha256':sha(F/'selected.pt'),'selected_step':best_step,
                    'best_selection_val_crps_64':best,'parameter_count':sum(p.numel() for p in model.parameters()),
                    'schedule_sha256':sha(D/f'schedule_{sd}.npy'),'fit_seconds':time.monotonic()-fit_start,
                    'sample_128_val_seconds':sample_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'last_1000_improvement_gt_1pct':cap,'selected_initial_state':best_step==0,
                    'quantile_variant_sorted_sample_max_abs':proof,'inference_state_unchanged':True,'test_scored':False}
                (F/'fit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
                (O/'partial_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                del model,optimizer,gt,plan;torch.cuda.empty_cache()
        # Strong ordinary tree mean + empirical residual paths; same available inputs.
        F=D/'hgb';F.mkdir();status(phase='hgb',dataset=name)
        def matrix(g):return np.concatenate([np.repeat(flat(g),H,axis=0),np.tile(np.eye(H,dtype='float32'),(len(g['y']),1))],axis=1)
        hgb_start=time.monotonic();est=HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,learning_rate=.05,
            min_samples_leaf=20,l2_regularization=1,max_bins=63,early_stopping=False,random_state=2026371)
        est.fit(matrix(fit),fit['y'].reshape(-1));cp=est.predict(matrix(cal)).reshape(-1,H);vp=est.predict(matrix(val)).reshape(-1,H)
        residual=cal['y']-cp;rng=np.random.default_rng(2026371);variants={}
        for mode in ['unconditional','analogue']:
            s=np.empty((len(vp),128,H));ind=np.empty_like(s)
            for i in range(len(vp)):
                pool=np.arange(len(residual)) if mode=='unconditional' else nearest[i]
                chosen=rng.choice(pool,128,replace=True);s[i]=vp[i]+residual[chosen]
                for h in range(H):ind[i,:,h]=s[i,rng.permutation(128),h]
            assert np.array_equal(np.sort(s,axis=1),np.sort(ind,axis=1))
            record(F,mode+'_joint',s,val['y'],val['origins']);record(F,mode+'_iid',ind,val['y'],val['origins'])
        joblib.dump(est,F/'model.joblib');np.savez_compressed(F/'point_and_calibration.npz',val_mean=vp,cal_mean=cp,cal_residual=residual,
            cal_targets=cal['y'],val_targets=val['y'],cal_origins=cal['origins'],val_origins=val['origins'])
        (F/'fit.json').write_text(json.dumps({'seconds':time.monotonic()-hgb_start,'n_iter':est.n_iter_,
            'no_val_hyperparameter_selection':True,'test_scored':False},indent=2),encoding='utf-8')
        (O/'partial_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        del groups,fit,cal,val,est;torch.cuda.empty_cache()
    final={'status':'completed','elapsed_seconds':time.monotonic()-started,'neural_fits':12,'hgb_fits':2,
        'neural_updates':12*STEPS,'rows':rows,'accepted_research_directions':0,'test_scored':False,
        'protocol_sha256':sha(ROOT/'LEARNING_PROTOCOL.md'),'common_sha256':sha(ROOT/'scripts/common37_v2.py'),
        'runner_sha256':sha(pathlib.Path(__file__)),'raw_archive_sha256':sha(ROOT/'data_intake/GEFCom2014_original.zip')}
    (O/'results.json').write_text(json.dumps(final,indent=2),encoding='utf-8');status(phase='completed',elapsed_seconds=final['elapsed_seconds'])
    print(json.dumps({'status':'completed','seconds':final['elapsed_seconds'],'rows':len(rows)}),flush=True)
except BaseException:
    status(phase='failed',error=traceback.format_exc());raise
