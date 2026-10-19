"""Fixed18-fit real-data migration; all model selection precedes held-out testing."""
import runtime_profile
import hashlib,json,pathlib,platform,time
import numpy as np
import torch
import real_models as m
ROOT=m.ROOT;OUT=ROOT/'outputs/real_comparators';DATA=ROOT/'outputs/real_data'
GROUPS={'current':[0,1],'RPM':[2,3],'pressure':[4,5,6,7],'microphone':[8],'signals':[9,10]}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ridge_features(hist,u):return np.column_stack([hist[:,-8:].reshape(len(hist),-1),u.reshape(len(u),-1)]).astype('float64')
def ridge_fit(train,val):
    x=ridge_features(train['x'].cpu().numpy(),train['u'].cpu().numpy());v=ridge_features(val['x'].cpu().numpy(),val['u'].cpu().numpy())
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);x=np.column_stack([(x-mean)/scale,np.ones(len(x))]);v=np.column_stack([(v-mean)/scale,np.ones(len(v))])
    y=train['y'].cpu().numpy().reshape(len(x),-1);truth=val['y'].cpu().numpy();xx=x.T@x/len(x);xy=x.T@y/len(x);candidates=[];models=[]
    for penalty in [1e-6,1e-4,1e-2,.1,1,10,100]:
        ridge=np.eye(x.shape[1])*penalty;ridge[-1,-1]=0;coeff=np.linalg.solve(xx+ridge,xy);pred=(v@coeff).reshape(truth.shape)
        score=float(np.mean((pred-truth)**2));candidates.append(dict(penalty=penalty,validation_MSE=score));models.append(coeff)
    selected=int(np.argmin([v['validation_MSE'] for v in candidates]));file=OUT/'ridge_model.npz'
    np.savez_compressed(file,coefficients=models[selected],mean=mean,scale=scale)
    record=dict(candidates=candidates,selected=candidates[selected],model_file=file.name,model_sha256=sha(file));(OUT/'ridge_selection.json').write_text(json.dumps(record,indent=2),encoding='utf-8');return record
def metrics(pred,truth,last,scale,ids,names):
    error=pred-truth;baseline=(last[:,None,:]-truth)**2;denominator=float(np.mean(baseline));mse=float(np.mean(error**2))
    result=dict(MSE=mse,MAE=float(np.mean(np.abs(error))),persistence_MSE=denominator,MSE_ratio_to_persistence=mse/max(denominator,1e-12),
        horizon_MSE=np.mean(error**2,axis=(0,2)).tolist(),channel_MSE=np.mean(error**2,axis=(0,1)).tolist(),
        raw_MAE_per_channel=np.mean(np.abs(error)*scale[:11],axis=(0,1)).tolist(),groups={k:float(np.mean(error[:,:,c]**2)) for k,c in GROUPS.items()},recordings=[])
    for i,name in enumerate(names):
        use=ids==i;result['recordings'].append(dict(file=str(name),windows=int(use.sum()),MSE=float(np.mean(error[use]**2)),MAE=float(np.mean(np.abs(error[use])))))
    return result
def evaluate(model,saved,native):
    pred=[];oracle=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(saved['history']),128):
            x=torch.tensor(saved['history'][start:start+128],device='cuda');u=torch.tensor(saved['future_conditions'][start:start+128],device='cuda')
            yhat=model.forecast(x,x_future=u);assert torch.isfinite(yhat).all();pred.append(yhat.cpu().numpy())
            if native:
                # Privileged future measurements are used only for an explicitly separate diagnostic.
                y=torch.tensor(saved['target'][start:start+128],device='cuda');out=model(input=x,x_future=u,future_target=y)
                decoded=model.decode(out['h_target'],target_revin_stats=out['target_revin_stats'],history_tokens=out['h_hist']);oracle.append(decoded.cpu().numpy())
    return np.concatenate(pred),None if not native else np.concatenate(oracle)
def main():
    m.setup();assert read(ROOT/'outputs/preflight.json')['pass_'];assert not OUT.exists();OUT.mkdir(parents=True)
    for name in ['checkpoints','predictions']:(OUT/name).mkdir()
    tick=time.perf_counter();registry=read(DATA/'registry.json')
    assert sha(ROOT/'EXPERIMENT_PLAN.md')==registry['plan_sha256'];assert sha(ROOT/'native_source_pin.json')==registry['source_pin_sha256']
    for r in registry['splits']:assert sha(DATA/r['file'])==r['sha256']
    config=dict(methods=m.METHODS,seeds=[2021,2022,2023],model_configs={k:m.config(k) for k in m.METHODS if k.startswith('worldts')},
        script_sha256=sha(pathlib.Path(__file__)),model_source_sha256=sha(ROOT/'scripts/real_models.py'),plan_sha256=registry['plan_sha256'],data_registry_sha256=sha(DATA/'registry.json'))
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8');train=m.load_split('train');val=m.load_split('val');fits=[]
    ridge_record=ridge_fit(train,val)
    for method in m.METHODS:
        for seed in [2021,2022,2023]:
            start=time.perf_counter();torch.manual_seed(seed);model=m.make(method);stages=['latent','decoder'] if method.startswith('worldts') and not method.endswith('endtoend') else ['supervised'];histories=[]
            for j,stage in enumerate(stages):
                record=m.stage_fit(model,stage,train,val,seed+1000*j);histories.append(record)
                file=OUT/'checkpoints'/(method+'_'+str(seed)+'_'+stage+'.pt');torch.save(dict(model=m.state(model),method=method,seed=seed),file)
                record.update(checkpoint=file.name,checkpoint_sha256=sha(file))
            row=dict(key=method+'_'+str(seed),method=method,seed=seed,stages=histories,checkpoint=histories[-1]['checkpoint'],checkpoint_sha256=histories[-1]['checkpoint_sha256'],
                seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),model_state_sha256=m.fingerprint(model),
                information_interface=dict(history_raw_values=64*16,known_future_command_values=16*5,target_latent_values=64*m.config(method)['latent_dim'] if method.startswith('worldts') else None,
                    condition_latent_values=64*m.config(method)['condition_dim'] if method.startswith('worldts') else None,RevIN_sidestream_values=32 if method in ['worldts32','worldts4','worldts32_endtoend'] else 0))
            fits.append(row);(OUT/'fit_records.json').write_text(json.dumps(fits,indent=2),encoding='utf-8');print(json.dumps(dict(phase='fit',completed=len(fits),total=18,key=row['key'],seconds=row['seconds'],best_epochs=[s['best_epoch'] for s in histories])),flush=True)
            del model;torch.cuda.empty_cache()
    assert len(fits)==18;(OUT/'training_complete.json').write_text(json.dumps(dict(fits=18,test_metrics_accessed=False,seconds=time.perf_counter()-tick)),encoding='utf-8')
    rows=[]
    for name in ['test','wave_matched','wave_acquisition']:
        saved=np.load(DATA/(name+'.npz'),allow_pickle=False);truth=saved['target'];last=saved['history'][:,-1,:11]
        ridge=np.load(OUT/ridge_record['model_file'],allow_pickle=False);xx=ridge_features(saved['history'],saved['future_conditions']);xx=np.column_stack([(xx-ridge['mean'])/ridge['scale'],np.ones(len(xx))])
        controls={'persistence':np.broadcast_to(last[:,None,:],truth.shape).copy(),'ridge_arx':(xx@ridge['coefficients']).reshape(truth.shape)}
        for method,pred in controls.items():
            file=OUT/'predictions'/(name+'_'+method+'.npz');np.savez_compressed(file,prediction=pred,recording_id=saved['recording_id'],origin=saved['origin'])
            rows.append(dict(split=name,method=method,seed=None,**metrics(pred,truth,last,saved['scale'],saved['recording_id'],saved['recording_names']),prediction_file=file.name,prediction_sha256=sha(file)))
        for fit in fits:
            state=torch.load(OUT/'checkpoints'/fit['checkpoint'],map_location='cpu',weights_only=True);model=m.make(fit['method']);model.load_state_dict(state['model'])
            pred,oracle=evaluate(model,saved,fit['method'].startswith('worldts'));file=OUT/'predictions'/(name+'_'+fit['key']+'.npz')
            arrays=dict(prediction=pred,recording_id=saved['recording_id'],origin=saved['origin'])
            if oracle is not None:arrays['privileged_future_target_decode']=oracle
            np.savez_compressed(file,**arrays)
            row=dict(split=name,method=fit['method'],seed=fit['seed'],**metrics(pred,truth,last,saved['scale'],saved['recording_id'],saved['recording_names']),prediction_file=file.name,prediction_sha256=sha(file))
            if oracle is not None:row['privileged_future_decode_MSE']=float(np.mean((oracle-truth)**2))
            rows.append(row);print(json.dumps(dict(phase='test',split=name,key=fit['key'],MSE=row['MSE'],persistence_ratio=row['MSE_ratio_to_persistence'])),flush=True);del model
    result=dict(rows=rows,config=config,elapsed_seconds=time.perf_counter()-tick,environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),maximum_allocated_bytes=torch.cuda.max_memory_allocated()),
        accepted_novel_directions=0,objective_achieved=False,scope='Native architecture/new-data migration,explicit objective controls and ordinary conditional baselines;not original paper benchmark reproduction or intervention identification.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(completed=18,seconds=result['elapsed_seconds'])),flush=True)
if __name__=='__main__':main()
