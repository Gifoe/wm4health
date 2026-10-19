"""Post-hoc frozen-checkpoint path replacements; no training or test tuning."""
import runtime_profile
import json,pathlib,time
import numpy as np
import torch
import real_models as m
ROOT=m.ROOT;MAIN=ROOT/'outputs/real_comparators';OUT=ROOT/'outputs/readout_path_control';DATA=ROOT/'outputs/real_data'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def archive(p):
    with np.load(p,allow_pickle=False) as s:return {k:s[k] for k in s.files}
def score(pred,truth):
    err=pred.astype('float64')-truth.astype('float64')
    return dict(MSE=float(np.mean(err**2)),MAE=float(np.mean(np.abs(err))),channel_MSE=np.mean(err**2,axis=(0,1)).tolist(),horizon_MSE=np.mean(err**2,axis=(0,2)).tolist())
def main():
    m.setup();assert read(MAIN/'independent_verification.json')['pass'];assert not OUT.exists();OUT.mkdir(parents=True)
    tick=time.perf_counter();train=archive(DATA/'train.npz');sets={k:archive(DATA/(k+'.npz')) for k in ['test','wave_matched','wave_acquisition']}
    fits=[v for v in read(MAIN/'fit_records.json') if v['method'].startswith('worldts')];assert len(fits)==12
    originals={(v['split'],v['method'],v['seed']):v for v in read(MAIN/'results.json')['rows']};rows=[];replays=[]
    for split,saved in sets.items():
        pred=np.broadcast_to(saved['history'][:,:,:11].mean(1)[:,None,:],saved['target'].shape).copy();file=OUT/(split+'_history_mean.npz');np.savez_compressed(file,prediction=pred)
        rows.append(dict(split=split,method='history_mean64',seed=None,control='history_mean',**score(pred,saved['target']),prediction_file=file.name,prediction_sha256=m.sha(file)))
    for fit in fits:
        file=MAIN/'checkpoints'/fit['checkpoint'];assert m.sha(file)==fit['checkpoint_sha256'];state=torch.load(file,map_location='cpu',weights_only=True);model=m.make(fit['method']);model.load_state_dict(state['model']);model.eval()
        hp=None;hh=None;count=0
        with torch.no_grad():
            for start in range(0,len(train['history']),128):
                x=torch.tensor(train['history'][start:start+128],device='cuda');u=torch.tensor(train['future_conditions'][start:start+128],device='cuda');out=model(input=x,x_future=u)
                a=out['h_pred'].double().sum(0);b=out['h_hist'].double().sum(0);hp=a if hp is None else hp+a;hh=b if hh is None else hh+b;count+=len(x)
            hp=(hp/count).float()[None];hh=(hh/count).float()[None]
        mean_file=OUT/(fit['key']+'_train_means.npz');np.savez_compressed(mean_file,future_latent_mean=hp.cpu().numpy(),history_latent_mean=hh.cpu().numpy())
        for split,saved in sets.items():
            arrays={'original':[],'mean_future':[],'mean_both':[]}
            with torch.no_grad():
                for start in range(0,len(saved['history']),128):
                    x=torch.tensor(saved['history'][start:start+128],device='cuda');u=torch.tensor(saved['future_conditions'][start:start+128],device='cuda');out=model(input=x,x_future=u)
                    def decode(future,history):return model.decode(future,target_revin_stats=out['target_revin_stats'],history_tokens=history).cpu().numpy()
                    arrays['original'].append(decode(out['h_pred'],out['h_hist']));arrays['mean_future'].append(decode(hp.expand(len(x),-1,-1),out['h_hist']));arrays['mean_both'].append(decode(hp.expand(len(x),-1,-1),hh.expand(len(x),-1,-1)))
            arrays={k:np.concatenate(v) for k,v in arrays.items()};reference=originals[split,fit['method'],fit['seed']];original=archive(MAIN/'predictions'/reference['prediction_file'])['prediction']
            error=float(np.max(np.abs(original-arrays['original'])));replays.append(error);assert error<2e-5
            for kind,pred in arrays.items():
                dest=OUT/(split+'_'+fit['key']+'_'+kind+'.npz');np.savez_compressed(dest,prediction=pred)
                row=dict(split=split,method=fit['method'],seed=fit['seed'],control=kind,**score(pred,saved['target']),prediction_file=dest.name,prediction_sha256=m.sha(dest),
                    checkpoint_sha256=fit['checkpoint_sha256'],train_means_file=mean_file.name,train_means_sha256=m.sha(mean_file),original_forecast_max_replay_error=error)
                rows.append(row)
            print(json.dumps(dict(completed_native_split=len(replays),method=fit['method'],seed=fit['seed'],split=split)),flush=True)
    result=dict(rows=rows,seconds=time.perf_counter()-tick,script_sha256=m.sha(pathlib.Path(__file__)),plan_sha256=m.sha(ROOT/'ARCHITECTURE_CONTROL_PLAN.md'),
        original_forecast_max_replay_error=max(replays),scope='Post-hoc fixed readout-path replacements,not test-tuned methods,information-theoretic absence or causal-world interventions.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(completed_rows=len(rows),seconds=result['seconds'],replay_error=max(replays))),flush=True)
if __name__=='__main__':main()
