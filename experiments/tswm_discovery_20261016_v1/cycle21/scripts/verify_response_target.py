"""Independent data/metric arithmetic and full-batch checkpoint replay."""
import runtime_profile
import hashlib,json,pathlib,time
import numpy as np
import torch
import response_target_prototype as r
p=r.p;ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/response_target_prototype'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def equation(x,g,u,f):
    q,v=x.T;a,b=u.T
    if f=='coupled_linear':return np.column_stack([.7*q+.15*v+.35*g*a,.6*v-.1*q+.3*b])
    if f=='rotating':return np.column_stack([.9*q+.25*v+.3*a,-.25*q+.9*v+.3*g*b])
    if f=='saturating':return np.column_stack([.7*np.tanh(q)+.15*v+.4*g*np.tanh(a),.5*v-.2*np.tanh(q)+.25*b])
    return np.column_stack([np.maximum(0,.8*q+.7*g-(.8+.4*a)),np.maximum(0,.75*v+.2*q+.45*g-(.5+.3*b))])
def main():
    tick=time.perf_counter();receipt=OUT/'independent_verification.json';assert not receipt.exists()
    status=json.loads((ROOT/'outputs/jobs/response_target_development/status.json').read_text());assert status['status']=='completed' and status['exit_code']==0
    result=json.loads((OUT/'results.json').read_text());config=result['config'];assert len(result['rows'])==120
    assert sha(ROOT/'scripts/response_target_prototype.py')==config['script_sha256']==status['script_sha256']
    assert sha(ROOT/'PROTOTYPE_PLAN.md')==config['plan_sha256']
    assert sha(ROOT.parent/'cycle20/scripts/innovation_pilot.py')==config['dependency_sha256']
    fits=json.loads((OUT/'fit_records.json').read_text());fit_by={v['key']:v for v in fits};teachers=json.loads((OUT/'teacher_records.json').read_text())
    assert len(fits)==120 and len(teachers)==12
    for row in teachers:assert sha(OUT/'teachers'/row['checkpoint'])==row['checkpoint_sha256']
    datasets={};physical_errors=[];inverse_errors=[];data_hashes=[]
    for entry in json.loads((OUT/'dataset_registry.json').read_text()):
        splits={}
        for split,record in entry['files'].items():
            file=OUT/'data'/record['file'];assert sha(file)==record['sha256'];data_hashes.append(sha(file));s=np.load(file,allow_pickle=False);splits[split]=s
            expected=equation(s['current'][:,:2],s['current'][:,2],s['action'],entry['family'])
            physical_errors.append(float(np.max(np.abs(expected-s['future'][:,:2]))))
            mixed=((s['current']-s['mean'])/s['scale']@s['rotation'].T).astype('float32')
            inverse_errors.append(float(np.max(np.abs(mixed@s['rotation']*s['scale']+s['mean']-s['current']))))
            assert len(np.unique(s['action'],axis=0))==4 and len(np.unique(s['current'][:,2]))==2
        train=splits['train'];assert np.array_equal(train['mean'],train['current'].mean(0));assert np.array_equal(train['scale'],train['current'].std(0,ddof=1))
        signatures=[set(map(bytes,np.ascontiguousarray(splits[n]['current']).view(np.dtype((np.void,48))).ravel())) for n in ['train','val','test']]
        assert not signatures[0]&signatures[1] and not signatures[0]&signatures[2] and not signatures[1]&signatures[2]
        datasets[entry['family']]=splits['test']
    assert max(physical_errors)<1e-6 and max(inverse_errors)<5e-6
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    arithmetic=[];replay=[];inverse_accuracy=[];rows=[];checkpoint_times=[];prediction_times=[]
    for row in result['rows']:
        fit=fit_by[row['key']];cf=OUT/'checkpoints'/fit['checkpoint'];pf=OUT/'predictions'/row['prediction_file']
        assert sha(cf)==fit['checkpoint_sha256'] and sha(pf)==row['prediction_sha256'];checkpoint_times.append(cf.stat().st_mtime_ns);prediction_times.append(pf.stat().st_mtime_ns)
        state=torch.load(cf,map_location='cpu',weights_only=True);pred=np.load(pf,allow_pickle=False);s=datasets[row['family']]
        for name,columns in [('decoder_choice',[0,1]),('gain_choice',[2]),('future_choice',[0,1])]:
            probe=state['future_probe'] if name=='future_choice' else state['decoder'];scores=probe['scores'].numpy()
            assert state[name]==['linear','nonlinear'][int(scores[:,columns].mean(1).argmin())]
        true=((s['future'][:,:2]-s['mean'][:2])/s['scale'][:2]).astype('float32');current=((s['current'][:,:2]-s['mean'][:2])/s['scale'][:2]).astype('float32')
        assert np.array_equal(true,pred['true_successor']) and np.array_equal(s['trajectory_id'],pred['trajectory_id'])
        lo=s['current'].copy();hi=lo.copy();lo[:,2]=.4;hi[:,2]=1.6
        lower=equation(lo[:,:2],lo[:,2],s['action'],row['family']);upper=equation(hi[:,:2],hi[:,2],s['action'],row['family'])
        true_mod=(upper-lower)/s['scale'][:2];assert np.max(np.abs(true_mod-pred['true_modulation']))<1e-6
        metrics={'gain_accuracy':float(np.mean((pred['gain_prediction']*s['scale'][2]+s['mean'][2]>1)==(s['current'][:,2]>1))),
            'native_physical_MSE':float(np.mean((pred['native_physical_prediction']-true)**2)),
            'current_physical_decoding_MSE':float(np.mean((pred['current_probe'][:,:2]-current)**2)),
            'frozen_encoder_successor_probe_MSE':float(np.mean((pred['frozen_encoder_successor_probe']-true)**2)),
            'oracle_successor_decoding_MSE':float(np.mean((pred['oracle_successor_decoding']-true)**2)),
            'gain_blind_oracle_physical_MSE':float(np.mean((((lower+upper)/2-s['future'][:,:2])/s['scale'][:2])**2))}
        for key,array in [('native_modulation_relative_RMSE','native_modulation'),('frozen_encoder_modulation_relative_RMSE','probe_modulation')]:
            metrics[key]=float(np.sqrt(np.mean((pred[array]-true_mod)**2)/np.mean(true_mod**2)))
        arithmetic.extend(abs(metrics[k]-row[k]) for k in metrics)
        model=r.make(row['method']);model.load_state_dict(state['world_model']);model.eval();decoder=p.restore_probe(state['decoder']);forward=p.restore_probe(state['future_probe'])
        def tensor(a):return torch.tensor(a,device='cuda')
        mean=tensor(s['mean']);scale=tensor(s['scale']);rotation=tensor(s['rotation']);u=tensor(s['action'])
        x=tensor(((s['current']-s['mean'])/s['scale']@s['rotation'].T).astype('float32'));y=tensor(((s['future']-s['mean'])/s['scale']@s['rotation'].T).astype('float32'))
        def latent(a):
            value=model.encoder(a);return torch.nn.functional.normalize(value,dim=-1) if isinstance(model,r.Mop) else value
        def decoded(z,probe,choice):return p.probe_predict(probe,z,choice)
        def native(a):
            z=latent(a);v=model.predict(z,u)
            if isinstance(model,r.Mop):
                heads=torch.stack([decoded(v[:,j],decoder,state['decoder_choice'])[:,:2] for j in range(2)],1)
                return (heads*torch.softmax(model.router(torch.cat([z,u],1)),1)[:,:,None]).sum(1)
            if model.obs:return ((v@rotation*scale+mean)[:,:2]-mean[:2])/scale[:2]
            return decoded(v,decoder,state['decoder_choice'])[:,:2]
        with torch.no_grad():
            z=latent(x);zn=latent(y);checks={'current_probe':decoded(z,decoder,state['decoder_choice']),
                'gain_prediction':decoded(z,decoder,state['gain_choice'])[:,2],
                'native_physical_prediction':native(x),'oracle_successor_decoding':decoded(zn,decoder,state['decoder_choice'])[:,:2],
                'frozen_encoder_successor_probe':decoded(torch.cat([z,u],1),forward,state['future_choice'])}
            lx=tensor(((lo-s['mean'])/s['scale']@s['rotation'].T).astype('float32'));hx=tensor(((hi-s['mean'])/s['scale']@s['rotation'].T).astype('float32'))
            checks['native_modulation']=native(hx)-native(lx)
            checks['probe_modulation']=decoded(torch.cat([latent(hx),u],1),forward,state['future_choice'])-decoded(torch.cat([latent(lx),u],1),forward,state['future_choice'])
            for key,value in checks.items():replay.append(float(np.max(np.abs(value.cpu().numpy()-pred[key]))))
            inv=model.inverse(torch.cat([z,zn],1));acc=float(((inv>0)==(u>0)).all(1).float().mean());inverse_accuracy.append(abs(acc-row['inverse_joint_action_accuracy']))
        rows.append(dict(key=row['key'],checkpoint_sha256=sha(cf),prediction_sha256=sha(pf),metrics_recomputed=metrics,inverse_action_accuracy=acc))
        del model,decoder,forward,x,y,z,zn
    assert max(arithmetic)<2e-6 and max(replay)<2e-5 and max(inverse_accuracy)<1e-7
    marker=(OUT/'training_complete.json').stat().st_mtime_ns;assert max(checkpoint_times)<=marker<=min(prediction_times)
    report=dict(pass_=True,dataset_hashes=len(data_hashes),checkpoint_hashes=120,prediction_hashes=120,teacher_hashes=12,
        metrics_recomputed=len(arithmetic),metric_max_error=max(arithmetic),checkpoint_replay_max_error=max(replay),
        inverse_accuracy_max_error=max(inverse_accuracy),physical_equation_max_error=max(physical_errors),observation_inverse_max_error=max(inverse_errors),
        all_fit_and_probe_choices_precede_test_predictions=True,rows=rows,seconds=time.perf_counter()-tick,script_sha256=sha(pathlib.Path(__file__)),
        scope='Independent arithmetic, raw equations, provenance and full-batch model replay; shares neural module definitions, not novelty confirmation.')
    report['pass']=report.pop('pass_');receipt.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
