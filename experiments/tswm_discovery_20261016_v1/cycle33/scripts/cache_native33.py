"""Float32 frozen native features, only TRAIN/VAL before all fits lock."""
from native_common33 import *
import time
def main():
    assert not CACHE.exists();CACHE.mkdir(parents=True)
    source=source_check();start=time.perf_counter();meta={}
    for ds in ['NP','PJM']:
        seed(2026330);model,cfg,original,provenance=load_native(ds)
        values,origins,data=load_data(ds);d=cfg['latent_dim'];p=cfg['decoder_patch_len'];T=168;H=24
        before=state_sha(model.state_dict());replays=[]
        with torch.inference_mode():
            for split in ['train','val']:
                folder=CACHE/ds/split;folder.mkdir(parents=True);o=origins[split];n=len(o)
                shapes={'pred':(n,H,d),'target':(n,H,d),'prefix':(n,p-1,d),'mean':(n,1,1),'scale':(n,1,1),'y':(n,H,1)}
                maps={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=np.float32,shape=s) for k,s in shapes.items()}
                np.save(folder/'origins.npy',o)
                for b in range(0,n,128):
                    ix=o[b:b+128]
                    hist=torch.tensor(np.stack([values[t-T:t] for t in ix]),device='cuda')
                    fut=torch.tensor(np.stack([values[t:t+H,1:] for t in ix]),device='cuda')
                    y=torch.tensor(np.stack([values[t:t+H,:1] for t in ix]),device='cuda')
                    out=model(hist,future_target=y,x_future=fut)
                    stats=out['target_revin_stats']
                    tensors={'pred':out['h_pred'],'target':out['h_target'],'prefix':out['h_hist'][:,-(p-1):],
                             'mean':stats['mean'],'scale':stats['stdev'],'y':y}
                    for k,x in tensors.items():assert torch.isfinite(x).all();maps[k][b:b+len(ix)]=x.cpu().numpy()
                    if b==0:
                        changed=model(hist,future_target=y+12345,x_future=fut)
                        diff=float((changed['h_pred']-out['h_pred']).abs().max());assert diff==0
                        cached=model.decode(out['h_pred'],target_revin_stats=stats,history_tokens=tensors['prefix'])
                        direct=model.decode(out['h_pred'],target_revin_stats=stats,history_tokens=out['h_hist'])
                        error=float((cached-direct).abs().max());assert error==0
                        replays.append({'split':split,'future_target_invariance':diff,'prefix_vs_full_history_decode':error,'batch':len(ix)})
                for mm in maps.values():mm.flush()
                del maps
            assert state_sha(model.state_dict())==before
        files=[]
        for f in sorted((CACHE/ds).rglob('*.npy')):files.append({'path':f.relative_to(CACHE/ds).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)})
        meta[ds]={'config':cfg,'provenance':provenance,'data':data,'source':source,'frozen_state_before':before,'frozen_state_after':state_sha(model.state_dict()),
                   'cache_dtype':'float32','replays':replays,'files':files,'test_features_generated':False}
        dump(CACHE/ds/'manifest.json',meta[ds]);del model;torch.cuda.empty_cache()
        print(json.dumps({'dataset':ds,'cache':'complete_trainval_only','counts':data['counts'],'bytes':sum(r['bytes'] for r in files)}),flush=True)
    dump(CACHE/'complete.json',{'status':'complete_trainval_only','datasets':meta,'seconds':time.perf_counter()-start,
        'cache_script_sha256':sha(__file__),'common_sha256':sha(ROOT/'scripts/native_common33.py'),'plan_sha256':sha(ROOT/'NATIVE_READOUT_PLAN.md')})
if __name__=='__main__':main()
