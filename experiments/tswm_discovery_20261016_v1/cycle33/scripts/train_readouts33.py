"""24 fixed-budget decoder refits. TEST is never loaded by this worker."""
from native_common33 import *
import time,math,traceback

def evaluate(head,pack):
    head.eval();absolute=squared=reconstruction=0.;count=0
    with torch.no_grad():
        for b in range(0,len(pack['y']),512):
            idx=slice(b,b+512);target=pack['y'][idx]
            output=forward_head(head,pack,idx)
            oracle=forward_head(head,pack,idx,kind='target')
            error=(output-target).double()
            absolute+=float(error.abs().sum());squared+=float((error*error).sum())
            reconstruction+=float((oracle-target).double().abs().sum());count+=target.numel()
    return {'forecast_mae':absolute/count,'forecast_mse':squared/count,'target_code_reconstruction_mae':reconstruction/count}

def save(path,head,meta):
    state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    tmp=path.with_name(path.name+'.tmp');torch.save({'state_dict':state,'meta':meta},tmp);tmp.replace(path)

def main():
    assert not FITS.exists();FITS.mkdir(parents=True)
    begin=time.perf_counter();source_check()
    cm=json.loads((CACHE/'complete.json').read_text())
    assert cm['status']=='complete_trainval_only'
    assert cm['common_sha256']==sha(ROOT/'scripts/native_common33.py')
    assert cm['plan_sha256']==sha(ROOT/'NATIVE_READOUT_PLAN.md')
    lock={'plan_sha256':sha(ROOT/'NATIVE_READOUT_PLAN.md'),'source_manifest_sha256':sha(ROOT/'frozen_source_manifest.json'),
          'cache_complete_sha256':sha(CACHE/'complete.json'),'script_sha256':sha(__file__),
          'common_sha256':sha(ROOT/'scripts/native_common33.py'),'datasets':['NP','PJM'],
          'seeds':SEEDS,'arms':ARMS,'epochs':20,'batch_size':32,'test_not_loaded':True,'pid':os.getpid()}
    dump(FITS/'source_lock.json',lock);completed=[]
    for ds in ['NP','PJM']:
        manifest=json.loads((CACHE/ds/'manifest.json').read_text());cfg=manifest['config']
        for row in manifest['files']:assert sha(CACHE/ds/row['path'])==row['sha256']
        train=cache_arrays(ds,'train');val=cache_arrays(ds,'val')
        n=len(train['y']);steps_per_epoch=math.ceil(n/32);base_lr=float(cfg['stage2_lr'])
        # Frozen feature standardization is estimated only once on TRAIN.
        f=global_features(train);global_stats=(f.mean(0),f.std(0,unbiased=False).clamp_min(1e-4));del f
        for s in SEEDS:
            orderpath=FITS/ds/f'orders_seed{s}.npy';orderpath.parent.mkdir(parents=True,exist_ok=True)
            orders=np.stack([np.random.default_rng(s+733000+e).permutation(n).astype(np.int32) for e in range(20)])
            np.save(orderpath,orders);order_hash=sha(orderpath)
            native_init_hash=None
            for arm in ARMS:
                folder=FITS/ds/f'{arm}_seed{s}';assert not folder.exists();folder.mkdir(parents=True)
                seed(s);torch.cuda.reset_peak_memory_stats();fit_start=time.perf_counter()
                head=make_head(cfg,arm,global_stats=global_stats)
                initial_hash=state_sha(head.state_dict())
                if arm=='native_mix':native_init_hash=initial_hash
                if arm=='native_pred':assert initial_hash==native_init_hash
                alpha=.3 if arm=='native_mix' else 0.
                optimizer=torch.optim.Adam(head.parameters(),lr=base_lr)
                history=[];best_forecast=float('inf');best_objective=float('inf');updates=0
                fitmeta={'dataset':ds,'arm':arm,'seed':s,'base_lr':base_lr,'alpha':alpha,
                         'latent_dim':cfg['latent_dim'],'prefix_length':cfg['decoder_patch_len']-1,
                         'order_sha256':order_hash,'initial_state_sha256':initial_hash,
                         'parameters':sum(p.numel() for p in head.parameters()),'source_lock_sha256':sha(FITS/'source_lock.json'),
                         'global_explicit_stats_and_all_future_tokens':arm=='global_pred'}
                for epoch in range(21):
                    train_loss=None;lr=lr_for_epoch(base_lr,max(0,epoch-1));epoch_start=time.perf_counter()
                    if epoch:
                        for group in optimizer.param_groups:group['lr']=lr
                        head.train();ix=torch.tensor(orders[epoch-1],dtype=torch.long,device='cuda');total=torch.zeros((),device='cuda')
                        for b in range(0,n,32):
                            batch=ix[b:b+32]
                            optimizer.zero_grad(set_to_none=True)
                            pred=forward_head(head,train,batch)
                            loss=(pred-train['y'][batch]).abs().mean()
                            if alpha:
                                recon=forward_head(head,train,batch,kind='target')
                                loss=loss+alpha*(recon-train['y'][batch]).abs().mean()
                            assert torch.isfinite(loss)
                            loss.backward();optimizer.step();total+=loss.detach()*len(batch);updates+=1
                        train_loss=float(total/n)
                    metrics=evaluate(head,val);objective=metrics['forecast_mae']+alpha*metrics['target_code_reconstruction_mae']
                    row={'epoch':epoch,'updates':updates,'lr':lr,'training_objective':train_loss,**metrics,
                         'selection_objective_secondary':objective,'seconds':time.perf_counter()-epoch_start}
                    history.append(row)
                    if metrics['forecast_mae']<best_forecast:
                        best_forecast=metrics['forecast_mae'];best_epoch=epoch
                        save(folder/'best_forecast.pt',head,{**fitmeta,'selected_epoch':epoch,'selection':'validation_forecast_mae','metrics':metrics})
                    if objective<best_objective:
                        best_objective=objective;best_obj_epoch=epoch
                        save(folder/'best_objective.pt',head,{**fitmeta,'selected_epoch':epoch,'selection':'validation_native_objective_secondary','metrics':metrics})
                    dump(folder/'history.json',history)
                    dump(FITS/'status.json',{'status':'running','pid':os.getpid(),'dataset':ds,'arm':arm,'seed':s,'epoch':epoch,
                         'completed_fits':len(completed),'planned_fits':24,'best_validation_forecast_mae':best_forecast,
                         'elapsed_seconds':time.perf_counter()-begin,'test_not_loaded':True})
                    if epoch==0 or epoch%5==0:print(json.dumps({'dataset':ds,'arm':arm,'seed':s,**row}),flush=True)
                assert updates==20*steps_per_epoch
                save(folder/'last.pt',head,{**fitmeta,'selected_epoch':20,'selection':'fixed_last','metrics':history[-1]})
                tail=[x['forecast_mae'] for x in history[-5:]]
                result={**fitmeta,'status':'complete','best_epoch':best_epoch,'best_objective_epoch':best_obj_epoch,
                        'best_validation_forecast_mae':best_forecast,'best_validation_objective':best_objective,
                        'updates':updates,'steps_per_epoch':steps_per_epoch,'epochs':20,
                        'last_validation_mae':history[-1]['forecast_mae'],
                        'cap_still_improving':min(tail[1:])<tail[0]*.995,
                        'seconds':time.perf_counter()-fit_start,'peak_allocated_including_shared_cache_bytes':torch.cuda.max_memory_allocated(),
                        'history_sha256':sha(folder/'history.json'),'best_forecast_sha256':sha(folder/'best_forecast.pt'),
                        'best_objective_sha256':sha(folder/'best_objective.pt'),'last_sha256':sha(folder/'last.pt'),
                        'test_not_loaded':True}
                dump(folder/'result.json',result);completed.append(result)
                print(json.dumps({'fit_complete':ds+'/'+folder.name,'completed':len(completed),'best_epoch':best_epoch,'best_validation':best_forecast,'seconds':result['seconds']}),flush=True)
                del head,optimizer;torch.cuda.empty_cache()
        del train,val,global_stats;torch.cuda.empty_cache()
    source_check();assert len(completed)==24
    dump(FITS/'training_complete.json',{'status':'complete_all_24_fits_test_not_loaded','fits':completed,
          'source_lock_sha256':sha(FITS/'source_lock.json'),'seconds':time.perf_counter()-begin})
    dump(FITS/'status.json',{'status':'complete_all_24_fits_test_not_loaded','pid':os.getpid(),'completed_fits':24,'planned_fits':24,
          'seconds':time.perf_counter()-begin,'test_not_loaded':True})

if __name__=='__main__':
    try:main()
    except Exception:
        if FITS.exists():dump(FITS/'failure.json',{'status':'failed','pid':os.getpid(),'traceback':traceback.format_exc()})
        raise
