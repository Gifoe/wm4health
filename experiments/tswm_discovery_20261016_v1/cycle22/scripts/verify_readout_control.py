"""Independent arithmetic and fixed-mean latent checkpoint replay."""
import runtime_profile
import json,pathlib,time
import numpy as np
import torch
import real_models as m
ROOT=m.ROOT;OUT=ROOT/'outputs/readout_path_control';MAIN=ROOT/'outputs/real_comparators';DATA=ROOT/'outputs/real_data'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def archive(p):
    with np.load(p,allow_pickle=False) as s:return {k:s[k] for k in s.files}
def main():
    tick=time.perf_counter();dest=OUT/'independent_verification.json';assert not dest.exists();m.setup();result=read(OUT/'results.json');assert len(result['rows'])==111
    assert m.sha(ROOT/'scripts/readout_path_control.py')==result['script_sha256'];assert m.sha(ROOT/'ARCHITECTURE_CONTROL_PLAN.md')==result['plan_sha256']
    datasets={k:archive(DATA/(k+'.npz')) for k in ['train','test','wave_matched','wave_acquisition']};fits=read(MAIN/'fit_records.json');arithmetic=[];replay=[];means=[]
    for row in result['rows']:
        file=OUT/row['prediction_file'];assert m.sha(file)==row['prediction_sha256'];pred=archive(file)['prediction'];truth=datasets[row['split']]['target'];err=pred.astype('float64')-truth.astype('float64')
        values=dict(MSE=float(np.mean(err**2)),MAE=float(np.mean(np.abs(err))),channel_MSE=np.mean(err**2,axis=(0,1)),horizon_MSE=np.mean(err**2,axis=(0,2)))
        for key,value in values.items():arithmetic.extend(np.abs(np.asarray(value)-np.asarray(row[key])).ravel().tolist())
        if row['control']=='history_mean':
            expectation=datasets[row['split']]['history'][:,:,:11].astype('float64').mean(1)[:,None,:];replay.append(float(np.max(np.abs(pred-expectation))))
    for fit in [v for v in fits if v['method'].startswith('worldts')]:
        file=MAIN/'checkpoints'/fit['checkpoint'];assert m.sha(file)==fit['checkpoint_sha256'];state=torch.load(file,map_location='cpu',weights_only=True);model=m.make(fit['method']);model.load_state_dict(state['model']);model.eval()
        meanfile=OUT/(fit['key']+'_train_means.npz');s=archive(meanfile);hp=None;hh=None;count=0;tr=datasets['train']
        with torch.no_grad():
            for start in range(0,len(tr['history']),128):
                x=torch.tensor(tr['history'][start:start+128],device='cuda');u=torch.tensor(tr['future_conditions'][start:start+128],device='cuda');out=model(input=x,x_future=u)
                a=out['h_pred'].double().sum(0);b=out['h_hist'].double().sum(0);hp=a if hp is None else hp+a;hh=b if hh is None else hh+b;count+=len(x)
            means.append(float(np.max(np.abs((hp/count).float().cpu().numpy()[None]-s['future_latent_mean']))));means.append(float(np.max(np.abs((hh/count).float().cpu().numpy()[None]-s['history_latent_mean']))))
            hp=torch.tensor(s['future_latent_mean'],device='cuda');hh=torch.tensor(s['history_latent_mean'],device='cuda')
            for split in ['test','wave_matched','wave_acquisition']:
                saved=datasets[split];x=torch.tensor(saved['history'][:128],device='cuda');u=torch.tensor(saved['future_conditions'][:128],device='cuda');out=model(input=x,x_future=u)
                for kind in ['original','mean_future','mean_both']:
                    row=next(v for v in result['rows'] if v['method']==fit['method'] and v['seed']==fit['seed'] and v['split']==split and v['control']==kind)
                    assert m.sha(meanfile)==row['train_means_sha256']
                    pred=archive(OUT/row['prediction_file'])['prediction'][:128]
                    actual=model.decode(out['h_pred'] if kind=='original' else hp.expand(len(x),-1,-1),target_revin_stats=out['target_revin_stats'],history_tokens=hh.expand(len(x),-1,-1) if kind=='mean_both' else out['h_hist']).cpu().numpy()
                    replay.append(float(np.max(np.abs(actual-pred))))
        del model
    assert max(arithmetic)<1e-12 and max(replay)<2e-6 and max(means)<1e-7
    report=dict(pass_=True,prediction_hashes=111,metrics_recomputed=len(arithmetic),metric_max_error=max(arithmetic),train_mean_recomputation_max_error=max(means),
        mean_checkpoints=12,first128_fixed_readout_replay_max_error=max(replay),seconds=time.perf_counter()-tick,script_sha256=m.sha(pathlib.Path(__file__)),
        scope='Full raw prediction arithmetic/hashes,training-only means recomputed and first128 neural path replacements per split. Shares native module definitions,not a causal intervention or MI proof.')
    report['pass']=report.pop('pass_');dest.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
