"""Independent result arithmetic, native loss equation and checkpoint validation replay."""
import runtime_profile
import copy,hashlib,json,pathlib,time
import numpy as np
import torch
import real_models as m
ROOT=m.ROOT;OUT=ROOT/'outputs/real_comparators';DATA=ROOT/'outputs/real_data'
GROUPS={'current':[0,1],'RPM':[2,3],'pressure':[4,5,6,7],'microphone':[8],'signals':[9,10]}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def archive(path):
    with np.load(path,allow_pickle=False) as file:return {k:file[k] for k in file.files}
def independent_latent(a,b):
    flat_a=a.reshape(len(a),-1);flat_b=b.reshape(len(b),-1)
    normalized_a=flat_a/torch.linalg.vector_norm(flat_a,dim=1,keepdim=True).clamp_min(1e-8)
    normalized_b=flat_b/torch.linalg.vector_norm(flat_b,dim=1,keepdim=True).clamp_min(1e-8)
    cosine=(1-(normalized_a*normalized_b).sum(1)).mean()
    def reg(z):
        z=z.reshape(-1,z.shape[-1]);z=z-z.mean(0);std=torch.sqrt((z*z).mean(0)+1e-4);cov=z.T@z/(len(z)-1)
        off=cov-torch.diag(torch.diag(cov));return torch.clamp(1-std,min=0).mean()+.04*(off*off).sum()/z.shape[1]
    return 10*(a-b).abs().mean()+15*cosine+.05*(reg(a)+reg(b))
def main():
    tick=time.perf_counter();receipt=OUT/'independent_verification.json';assert not receipt.exists();m.setup()
    status=read(ROOT/'outputs/jobs/real_conditional_comparator_training/status.json');assert status['status']=='completed' and status['exit_code']==0
    result=read(OUT/'results.json');fits=read(OUT/'fit_records.json');assert len(fits)==18 and len(result['rows'])==60
    cfg=result['config'];assert sha(ROOT/'scripts/train_real_comparators.py')==cfg['script_sha256']==status['script_sha256']
    assert sha(ROOT/'scripts/real_models.py')==cfg['model_source_sha256'] and sha(ROOT/'EXPERIMENT_PLAN.md')==cfg['plan_sha256']
    assert sha(DATA/'registry.json')==cfg['data_registry_sha256'];assert read(DATA/'independent_verification.json')['pass']
    datasets={v['split']:archive(DATA/v['file']) for v in read(DATA/'registry.json')['splits']}
    stage_times=[];pred_times=[];arithmetic=[];relative=[];replays=[];validation=[];loss_errors=[];checks=[];source_errors=[];corrected_rows=[];worst_field=None;worst_difference=-1
    for fit in fits:
        for stage in fit['stages']:
            file=OUT/'checkpoints'/stage['checkpoint'];assert sha(file)==stage['checkpoint_sha256'];stage_times.append(file.stat().st_mtime_ns)
            best=min(stage['history'],key=lambda a:a['validation_criterion']);assert best['epoch']==stage['best_epoch'] and best['validation_criterion']==stage['best_validation']
        if len(fit['stages'])==2:
            first=torch.load(OUT/'checkpoints'/fit['stages'][0]['checkpoint'],weights_only=True,map_location='cpu')['model']
            last=torch.load(OUT/'checkpoints'/fit['checkpoint'],weights_only=True,map_location='cpu')['model']
            assert all(torch.equal(v,last[k]) for k,v in first.items() if not k.startswith('decoder.'))
        model=m.make(fit['method']);state=torch.load(OUT/'checkpoints'/fit['checkpoint'],weights_only=True,map_location='cpu');model.load_state_dict(state['model']);model.eval()
        assert m.fingerprint(model)==fit['model_state_sha256']
        v=datasets['val'];scores=[]
        with torch.no_grad():
            for start in range(0,len(v['history']),128):
                x=torch.tensor(v['history'][start:start+128],device='cuda');u=torch.tensor(v['future_conditions'][start:start+128],device='cuda');y=torch.tensor(v['target'][start:start+128],device='cuda')
                score=(model.forecast(x,x_future=u)-y).square().mean();scores.append((len(x),float(score)))
            actual=sum(n*a for n,a in scores)/sum(n for n,a in scores);validation.append(abs(actual-fit['stages'][-1]['best_validation']))
            if len(fit['stages'])==2:
                out=model(input=x,x_future=u,future_target=y);score=independent_latent(out['h_pred'],out['h_target']);source=m.latent_loss()(out['h_pred'],out['h_target'])['loss'];loss_errors.append(abs(float(score-source)))
        del model
    for row in result['rows']:
        saved=datasets[row['split']];file=OUT/'predictions'/row['prediction_file'];assert sha(file)==row['prediction_sha256'];pred_times.append(file.stat().st_mtime_ns);a=archive(file)
        assert np.array_equal(a['recording_id'],saved['recording_id']) and np.array_equal(a['origin'],saved['origin'])
        pred=a['prediction'].astype('float64');truth=saved['target'].astype('float64');err=pred-truth;baseline=saved['history'][:,-1,None,:11].astype('float64')-truth
        source_err=a['prediction']-saved['target'];source_base=saved['history'][:,-1,None,:11]-saved['target'];corrected=copy.deepcopy(row)
        source_metrics={'MSE':float(np.mean(source_err**2)),'MAE':float(np.mean(np.abs(source_err))),'persistence_MSE':float(np.mean(source_base**2)),
            'horizon_MSE':np.mean(source_err**2,axis=(0,2)),'channel_MSE':np.mean(source_err**2,axis=(0,1)),
            'raw_MAE_per_channel':np.mean(np.abs(source_err)*saved['scale'][:11],axis=(0,1))}
        source_metrics['MSE_ratio_to_persistence']=source_metrics['MSE']/max(source_metrics['persistence_MSE'],1e-12)
        metrics={'MSE':float(np.mean(err**2)),'MAE':float(np.mean(np.abs(err))),'persistence_MSE':float(np.mean(baseline**2)),
            'horizon_MSE':np.mean(err**2,axis=(0,2)),'channel_MSE':np.mean(err**2,axis=(0,1)),
            'raw_MAE_per_channel':np.mean(np.abs(err)*saved['scale'][:11],axis=(0,1))}
        metrics['MSE_ratio_to_persistence']=metrics['MSE']/max(metrics['persistence_MSE'],1e-12)
        for k,value in metrics.items():
            delta=np.abs(np.asarray(value)-np.asarray(row[k]));arithmetic.extend(delta.ravel().tolist());relative.extend((delta/np.maximum(1,np.abs(row[k]))).ravel().tolist())
            source_errors.extend(np.abs(np.asarray(source_metrics[k])-np.asarray(row[k])).ravel().tolist())
            corrected[k]=np.asarray(value).tolist()
            magnitude=float(np.max(delta/np.maximum(1,np.abs(row[k]))))
            if magnitude>worst_difference:worst_difference=magnitude;worst_field=dict(split=row['split'],method=row['method'],seed=row['seed'],field=k,relative_difference=magnitude)
        for group,cols in GROUPS.items():
            score=float(np.mean(err[:,:,cols]**2));error=abs(score-row['groups'][group]);arithmetic.append(error);relative.append(error/max(1,abs(row['groups'][group])))
            source_errors.append(abs(float(np.mean(source_err[:,:,cols]**2))-row['groups'][group]));corrected['groups'][group]=score
        for rec in row['recordings']:
            i=list(saved['recording_names']).index(rec['file']);use=saved['recording_id']==i;assert int(use.sum())==rec['windows']
            for k,value in [('MSE',float(np.mean(err[use]**2))),('MAE',float(np.mean(np.abs(err[use]))))]:
                error=abs(value-rec[k]);arithmetic.append(error);relative.append(error/max(1,abs(rec[k])))
                source_value=float(np.mean(source_err[use]**2)) if k=='MSE' else float(np.mean(np.abs(source_err[use])))
                source_errors.append(abs(source_value-rec[k]));next(c for c in corrected['recordings'] if c['file']==rec['file'])[k]=value
        if 'privileged_future_target_decode' in a:
            score=float(np.mean((a['privileged_future_target_decode'].astype('float64')-truth)**2));error=abs(score-row['privileged_future_decode_MSE']);arithmetic.append(error);relative.append(error/max(1,abs(row['privileged_future_decode_MSE'])))
            source_errors.append(abs(float(np.mean((a['privileged_future_target_decode']-saved['target'])**2))-row['privileged_future_decode_MSE']));corrected['privileged_future_decode_MSE']=score
        if row['method']=='persistence':replays.append(float(np.max(np.abs(pred-saved['history'][:,-1,None,:11]))))
        elif row['method']=='ridge_arx':
            model=archive(OUT/'ridge_model.npz');x=np.concatenate([saved['history'][:,-8:].reshape(len(pred),128),saved['future_conditions'].reshape(len(pred),80)],axis=1).astype('float64')
            design=np.concatenate([(x-model['mean'])/model['scale'],np.ones((len(x),1))],axis=1);actual=(design@model['coefficients']).reshape(pred.shape);replays.append(float(np.max(np.abs(actual-pred))))
        else:
            fit=next(v for v in fits if v['method']==row['method'] and v['seed']==row['seed']);state=torch.load(OUT/'checkpoints'/fit['checkpoint'],weights_only=True,map_location='cpu');model=m.make(row['method']);model.load_state_dict(state['model']);model.eval()
            with torch.no_grad():
                x=torch.tensor(saved['history'][:128],device='cuda');u=torch.tensor(saved['future_conditions'][:128],device='cuda');actual=model.forecast(x,x_future=u).cpu().numpy();replays.append(float(np.max(np.abs(actual-pred[:128]))))
                if 'privileged_future_target_decode' in a:
                    y=torch.tensor(saved['target'][:128],device='cuda');out=model(input=x,x_future=u,future_target=y);value=model.decode(out['h_target'],target_revin_stats=out['target_revin_stats'],history_tokens=out['h_hist']).cpu().numpy()
                    replays.append(float(np.max(np.abs(value-a['privileged_future_target_decode'][:128]))))
            del model
        checks.append(dict(split=row['split'],method=row['method'],seed=row['seed'],prediction_sha256=sha(file)))
        corrected_rows.append(corrected)
    ridge=read(OUT/'ridge_selection.json');assert sha(OUT/ridge['model_file'])==ridge['model_sha256'];assert ridge['selected']==min(ridge['candidates'],key=lambda r:r['validation_MSE'])
    rm=archive(OUT/ridge['model_file']);tr=datasets['train'];va=datasets['val']
    def design(s):
        value=np.concatenate([s['history'][:,-8:].reshape(len(s['history']),128),s['future_conditions'].reshape(len(s['history']),80)],axis=1).astype('float64')
        return np.column_stack([(value-rm['mean'])/rm['scale'],np.ones(len(value))])
    xx=design(tr);yy=tr['target'].reshape(len(xx),-1);penalty=rm['coefficients']*ridge['selected']['penalty'];penalty[-1]=0
    normal_error=float(np.max(np.abs(xx.T@(xx@rm['coefficients']-yy)/len(xx)+penalty)));assert normal_error<1e-8
    val_pred=(design(va)@rm['coefficients']).reshape(va['target'].shape);ridge_val_error=abs(float(np.mean((val_pred-va['target'])**2))-ridge['selected']['validation_MSE']);assert ridge_val_error<1e-12
    marker=(OUT/'training_complete.json').stat().st_mtime_ns;assert max(stage_times)<=marker<=min(pred_times) and (OUT/'ridge_selection.json').stat().st_mtime_ns<=marker
    diagnostics=dict(metric_relative_max_error=max(relative),worst_field=worst_field,original_dtype_arithmetic_max_error=max(source_errors),checkpoint_replay_max_error=max(replays),validation_max_error=max(validation),latent_equation_max_error=max(loss_errors),validation_by_fit=[dict(key=f['key'],error=e) for f,e in zip(fits,validation)])
    print(json.dumps(diagnostics),flush=True)
    assert max(source_errors)<1e-12 and max(replays)<2e-5 and max(validation)<1e-7 and max(loss_errors)<1e-5
    precision=copy.deepcopy(result);precision['rows']=corrected_rows
    precision['precision_note']=dict(original_results_sha256=sha(OUT/'results.json'),aggregation='All errors/reductions recomputed from original arrays in float64;training/checkpoint choices and predictions unchanged.',
        maximum_difference_from_original_dtype_reductions=max(arithmetic),maximum_relative_difference=max(relative),worst_field=worst_field)
    precision_file=OUT/'precision_corrected_results.json';assert not precision_file.exists();precision_file.write_text(json.dumps(precision,indent=2),encoding='utf-8')
    report=dict(pass_=True,prediction_hashes=60,model_fits=18,stage_checkpoint_hashes=len(stage_times),metrics_recomputed=len(arithmetic),
        metric_absolute_max_error=max(arithmetic),metric_relative_max_error=max(relative),first128_checkpoint_replay_max_error=max(replays),
        original_dtype_arithmetic_max_error=max(source_errors),float64_aggregation_correction_file=precision_file.name,float64_aggregation_sha256=sha(precision_file),worst_float32_reduction_discrepancy=worst_field,
        full_validation_forecast_MSE_max_error=max(validation),independent_latent_equation_max_error=max(loss_errors),
        ridge_normal_equation_max_residual=normal_error,ridge_validation_MSE_error=ridge_val_error,
        all_selection_precedes_test=True,frozen_stage1_states_unchanged_through_decoder_training=True,checks=checks,
        seconds=time.perf_counter()-tick,script_sha256=sha(pathlib.Path(__file__)),scope='All result arithmetic/hashes;all validation forecasts;first128 neural samples/privileged diagnostic per split;full ridge/persistence replay. Shares native neural module definitions,not novelty confirmation.')
    report['pass']=report.pop('pass_');receipt.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)
if __name__=='__main__':main()
