"""Independent generator/metric arithmetic plus checkpoint replay of the frozen pilot."""
import runtime_profile
import hashlib,json,pathlib,time
import numpy as np
import torch
import innovation_pilot as p
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/innovation_pilot'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    tick=time.perf_counter();receipt=OUT/'independent_verification.json';assert not receipt.exists()
    status=json.loads((ROOT/'outputs/jobs/causal_innovation_objective_pilot/status.json').read_text());assert status['status']=='completed' and status['exit_code']==0
    config=json.loads((OUT/'config.json').read_text());result=json.loads((OUT/'results.json').read_text());fits=json.loads((OUT/'fit_records.json').read_text())
    assert len(result['rows'])==len(fits)==72 and len({r['key'] for r in fits})==72
    assert sha(ROOT/'scripts/innovation_pilot.py')==config['script_sha256']==status['script_sha256']
    assert sha(ROOT/'PILOT_PLAN.md')==config['plan_sha256']
    registry=json.loads((OUT/'dataset_registry.json').read_text());max_equation=0.;max_inverse=0.;datasets={}
    for item in registry:
        family=item['family'];datasets[family]={}
        for name,row in item['files'].items():
            file=OUT/'data'/row['file'];assert sha(file)==row['sha256'];saved=np.load(file,allow_pickle=False)
            state=saved['current'];future=saved['future'];action=saved['action'];ids=saved['trajectory_id']
            assert state.shape==future.shape==(row['transitions'],12) and action.shape==(len(state),2)
            assert np.all(np.isfinite(state)) and np.all(np.isfinite(future));assert np.array_equal(ids,np.repeat(np.arange(row['trajectories']),64))
            # Independent equations, do not call the generator's physical_step.
            q,v,g=state[:,0].astype('float64'),state[:,1].astype('float64'),state[:,2].astype('float64');u,w=action.T
            if family=='coupled_linear':expected=np.column_stack([.7*q+.15*v+.35*g*u,.6*v-.1*q+.3*w])
            elif family=='rotating':expected=np.column_stack([.9*q+.25*v+.3*u,-.25*q+.9*v+.3*g*w])
            else:expected=np.column_stack([.7*np.tanh(q)+.15*v+.4*g*np.tanh(u),.5*v-.2*np.tanh(q)+.25*w])
            error=float(np.max(np.abs(expected-future[:,:2])));assert error<1e-6;max_equation=max(max_equation,error)
            assert set(np.unique(g).round(1))=={.4,1.6} and set(np.unique(action))=={-1.,1.}
            mean=saved['mean'];scale=saved['scale'];rotation=saved['rotation'];assert np.max(np.abs(rotation@rotation.T-np.eye(12)))<1e-6
            x=(state-mean)/scale@rotation.T;inverse=x@rotation*scale+mean
            error=float(np.max(np.abs(inverse-state)));assert error<1e-5;max_inverse=max(max_inverse,error)
            if name=='train':assert np.array_equal(mean,state.mean(0)) and np.array_equal(scale,state.std(0,ddof=1))
            datasets[family][name]={k:saved[k] for k in saved.files}
        # Independent streams: exact observation rows cannot be shared across splits.
        train=datasets[family]['train']['current'];val=datasets[family]['val']['current'];test=datasets[family]['test']['current']
        hashes=[set(hashlib.sha256(row.tobytes()).digest() for row in x) for x in [train,val,test]]
        assert not (hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2])
    complete=OUT/'training_complete.json';assert json.loads(complete.read_text())['test_metrics_accessed'] is False
    assert max((OUT/'checkpoints'/r['checkpoint']).stat().st_mtime_ns for r in fits)<=complete.stat().st_mtime_ns
    assert min((OUT/'predictions'/r['prediction_file']).stat().st_mtime_ns for r in result['rows'])>=complete.stat().st_mtime_ns
    rows_by_key={r['key']:r for r in fits};max_metric_error=0.;tested=0;max_replay=0.;replays=0
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    for row in result['rows']:
        fit=rows_by_key[row['key']];ckpt=OUT/'checkpoints'/fit['checkpoint'];assert sha(ckpt)==fit['checkpoint_sha256']
        file=OUT/'predictions'/row['prediction_file'];assert sha(file)==row['prediction_sha256'];pred=np.load(file,allow_pickle=False)
        data=datasets[row['family']]['test'];mean=data['mean'];scale=data['scale'];rotation=data['rotation']
        truth=(data['future'][:,:2]-mean[:2])/scale[:2];assert np.array_equal(pred['true_successor'],truth)
        assert np.array_equal(pred['trajectory_id'],data['trajectory_id'])
        gtruth=(data['current'][:,2]-mean[2])/scale[2];u=data['action']
        delta=np.zeros_like(truth)
        if row['family']=='rotating':delta[:,1]=.36*u[:,1]/scale[1]
        elif row['family']=='coupled_linear':delta[:,0]=.42*u[:,0]/scale[0]
        else:delta[:,0]=.48*np.tanh(u[:,0])/scale[0]
        assert np.max(np.abs(delta-pred['true_modulation']))<1e-6
        expected={
            'gain_accuracy':np.mean((pred['gain_prediction']*scale[2]+mean[2]>1)==(data['current'][:,2]>1)),
            'gain_R2':1-np.mean((pred['gain_prediction'].astype('float64')-gtruth)**2)/np.var(gtruth.astype('float64')),
            'native_physical_MSE':np.mean((pred['native_physical_prediction'].astype('float64')-truth)**2),
            'frozen_encoder_successor_probe_MSE':np.mean((pred['frozen_encoder_successor_probe'].astype('float64')-truth)**2),
            'oracle_successor_decoding_MSE':np.mean((pred['oracle_successor_decoding'].astype('float64')-truth)**2)}
        for key,array in [('native_modulation_relative_RMSE','native_modulation'),('frozen_encoder_modulation_relative_RMSE','probe_modulation')]:
            expected[key]=np.sqrt(np.mean((pred[array].astype('float64')-pred['true_modulation'])**2)/np.mean(pred['true_modulation'].astype('float64')**2))
        for key,value in expected.items():
            error=abs(float(value)-row[key]);assert error<1e-5,(row['key'],key,error);max_metric_error=max(max_metric_error,error);tested+=1
        state=torch.load(ckpt,map_location='cpu',weights_only=True)
        assert state['decoder_choice']==['linear','nonlinear'][np.argmin(np.asarray(fit['decoder_validation_MSE'])[:,:2].mean(1))]
        assert state['gain_choice']==['linear','nonlinear'][np.argmin(np.asarray(fit['decoder_validation_MSE'])[:,2])]
        assert state['future_choice']==['linear','nonlinear'][np.argmin(np.asarray(fit['successor_probe_validation_MSE']).mean(1))]
        # One full-size deterministic replay per specification on the first family.
        if row['family']=='coupled_linear' and row['seed']==2021:
            model=p.Model(row['method']).to('cuda');model.load_state_dict(state['world_model']);model.eval();decoder=p.restore_probe(state['decoder'])
            raw={k:data[k] for k in ['current','future','action','trajectory_id']};test=p.to_gpu(p.prep(raw,mean,scale,rotation))
            tm=torch.tensor(mean,device='cuda');ts=torch.tensor(scale,device='cuda');tr=torch.tensor(rotation,device='cuda')
            with torch.no_grad():native=p.model_physical(model,test['x'],test['u'],decoder,state['decoder_choice'],tm,ts,tr)
            replay=((native-tm[:2])/ts[:2]).cpu().numpy();error=float(np.max(np.abs(replay-pred['native_physical_prediction'])))
            assert error==0,(row['key'],error);max_replay=max(max_replay,error);replays+=1
    info=dict(pass_=True,source_and_plan_hashes_unchanged=True,dataset_files_verified=9,checkpoints_verified=72,prediction_files_verified=72,
        independent_physical_equation_max_error=max_equation,observation_transform_inverse_max_error=max_inverse,
        test_metric_values_recomputed=tested,max_metric_arithmetic_error=max_metric_error,
        native_checkpoint_full_batch_replays=replays,max_replay_error=max_replay,
        all_training_and_probe_choices_before_test_predictions=True,split_rows_disjoint=True,
        seconds=time.perf_counter()-tick,script_sha256=sha(pathlib.Path(__file__)),
        limits='Arithmetic and implementation checks do not establish native-paper failure, novelty, physical prevalence or a new method.')
    info['pass']=info.pop('pass_');receipt.write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
