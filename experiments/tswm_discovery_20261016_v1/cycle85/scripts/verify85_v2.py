"""Replay selected models and independently reconstruct metrics and continuous truth."""
from common85_v1 import *
from scipy.integrate import solve_ivp

BC,ALIGN,HASHES=native_modules()
if len(sys.argv)>1:
    assert sys.argv[1]=='capacity'
    import common85_v1 as base85
    O=C/'outputs'/'capacity_v2';base85.O=O
    DATA=Path(CONFIG['data_root'])/'cycle85'/'anchored_screen_capacity_v2'
def model(nu,state):
    weight_shapes=[v.shape for k,v in state['obs_enc'].items() if k.endswith('weight')]
    a=BC.BCObsAgent(dict(obs_dim=2,robot_obs_dim=2,obj_obs_dim=0,lat_obs_dim=int(weight_shapes[-1][0])),dict(act_dim=nu+1),'cpu',n_layers=len(weight_shapes)-1,hidden_dim=int(weight_shapes[0][0]),lr=3e-4)
    unpack(a,state);freeze(a);return a
def powers(a,exponents):
    cols=[]
    for exponent in exponents:
        c=np.ones(len(a))
        for j,p in enumerate(exponent):
            for _ in range(int(p)):c*=a[:,j]
        cols.append(c)
    return np.stack(cols,axis=1)
def phi(a,kind,spec):
    if kind=='polynomial3':return powers(a,spec['exponents'])
    return np.concatenate([np.ones((len(a),1)),a,np.sqrt(2/128)*np.cos(a@spec['frequency']+spec['phase'])],axis=1)
def roll(y0,u,coef,kind,spec):
    state=y0.copy();out=[]
    with np.errstate(over='ignore',invalid='ignore'):
        for command in np.swapaxes(u,0,1):
            bad=(~np.isfinite(state).all(axis=1))|(np.max(np.abs(state),axis=1)>1e4)
            bounded=state.copy();bounded[bad]=0.
            state=state+phi(np.concatenate([bounded,command],axis=1),kind,spec)@coef;state[bad]=np.nan;out.append(state.copy())
    return np.stack(out,axis=1)
def metric(p,y,h):
    scores=[]
    for path in range(len(y)):
        a=p[path,:h];b=y[path,1:h+1]
        if (not np.isfinite(a).all()) or np.max(np.abs(a))>1e4:scores.append(1e12);continue
        rmse=np.sqrt(np.sum((a-b)**2,axis=0)/h)
        center=np.mean(b,axis=0);std=np.sqrt(np.sum((b-center)**2,axis=0)/h)
        scores.append(float(np.mean(rmse/np.maximum(std,.05))))
    return np.asarray(scores)
def own_field(state,command,family,changed):
    q,p=state
    if family=='duffing':return np.array([p,-(1.5 if changed else 1.)*q-(.2 if changed else .4)*p-.2*q**3+.7*command[0]])
    rate=2.4 if changed else 1.5
    return np.array([1-(rate+1)*q+q*q*p+.2*command[0],rate*q-q*q*p+.2*command[1]])
def own_truth(init,u,family,changed):
    state=init.copy();values=[state]
    for start in range(0,128,8):
        block=u[start:start+8];assert np.max(np.abs(block-block[0]))==0
        sol=solve_ivp(lambda t,y:own_field(y,block[0],family,changed),(0,.4),state,t_eval=np.arange(1,9)*.05,method='DOP853',rtol=1e-10,atol=1e-12)
        assert sol.success and sol.y.shape==(2,8);values.extend(sol.y.T);state=sol.y[:,-1]
    return np.asarray(values)
def check_case(case):
    family,seed,change=case['family'],case['seed'],case['case'];tag=f'{family}_{seed}_{change}'
    seal_path=O/f'SELECTION_SEALED85_{tag}_V1.json';assert sha(seal_path)==case['selection_seal_sha256'];seal=json.loads(seal_path.read_text());assert seal['eval_generated'] is False
    raw=Path(case['array_path']);assert sha(raw)==case['array_sha256'];d=np.load(raw,allow_pickle=False)
    assert datetime.datetime.fromisoformat(seal['utc']).timestamp()<=raw.stat().st_mtime
    td=Path(case['target_data_path']);assert sha(td)==case['target_data_sha256']==seal['target_data_sha256'];t=np.load(td,allow_pickle=False)
    source_record=json.loads((O/f'SOURCE_READY85_{family}_{seed}_V1.json').read_text());assert source_record['qualified'] and sha(source_record['checkpoint'])==source_record['checkpoint_sha256']==seal['source_checkpoint_sha256']
    sp=DATA/f'{family}_{seed}'/'source_trainval_v1.npz';assert sha(sp)==source_record['source_data_sha256'];s=np.load(sp,allow_pickle=False)
    assert np.max(np.abs(s['y'][:,:-1].mean(axis=(0,1))))<1e-12 and np.max(np.abs(s['y'][:,:-1].std(axis=(0,1))-1))<1e-12
    nu=d['actions'].shape[-1];source_state=torch.load(source_record['checkpoint'],map_location='cpu',weights_only=True)
    models={'frozen':model(nu,source_state)};selection_scores={};replay_error=0.;coefficient_error=0.;score_error=0.;reference_error=0.;values=0
    for name in ['native_frozen_alignment','joint_output_error']:
        record=seal['selected_neural'][name];assert sha(record['checkpoint'])==record['checkpoint_sha256'];assert record['validation_score']==min(z['validation_score'] for z in record['validation_history'])
        fits=[json.loads(p.read_text()) for p in O.glob(f'FIT85_{family}_{seed}_{change}_{name}_*_V1.json')]
        assert len(fits)==2 and record['validation_score']==min(z['validation_score'] for z in fits)
        models[name]=model(nu,torch.load(record['checkpoint'],map_location='cpu',weights_only=True));selection_scores[name]=record['validation_score']
        if name=='native_frozen_alignment':
            for key,z in source_state['fwd_dyn'].items():assert torch.equal(z,pack(models[name])['fwd_dyn'][key])
    ordinary=json.loads((O/f'ORDINARY85_{tag}_V1.json').read_text());cp=Path(ordinary['coefficient_path']);assert sha(cp)==ordinary['coefficient_sha256'];coefs=np.load(cp,allow_pickle=False);spec={k:coefs[k] for k in ['exponents','frequency','phase']}
    xx=np.concatenate([t['y'][:,:-1].reshape(-1,2),t['u'].reshape(-1,nu)],axis=1);yy=np.diff(t['y'],axis=1).reshape(-1,2)
    for kind in ['polynomial3','rff128']:
        z=ordinary['selected'][kind];selection_scores[kind]=z['validation_score']
        assert z['validation_score']==min(v['validation_score'] for v in ordinary['candidates'] if v['method']==kind)
        features=phi(xx,kind,spec);n=features.shape[1];lam,tau=z['ridge'],z['source_prior']
        independent=np.linalg.lstsq(np.vstack([features,np.sqrt(lam)*np.eye(n),np.sqrt(tau)*np.eye(n)]),np.vstack([yy,np.zeros((n,2)),np.sqrt(tau)*coefs[kind+'_source_coef']]),rcond=None)[0]
        coefficient_error=max(coefficient_error,float(np.max(np.abs(independent-coefs[kind+'_coef']))))
        assert np.allclose(independent,coefs[kind+'_coef'],atol=1e-5,rtol=1e-5)
    assert case['best_existing_by_validation']==min(selection_scores,key=selection_scores.get)
    for name in ['frozen','native_frozen_alignment','joint_output_error','polynomial3','rff128']:
        expected=d[name+'_prediction']
        got=predict(models[name],d['initial_observation'],d['actions']) if name in models else roll(d['initial_observation'],d['actions'],coefs[name+'_coef'],name,spec)
        assert np.allclose(got,expected,atol=2e-5,rtol=2e-5,equal_nan=True)
        finite=np.isfinite(got)&np.isfinite(expected)
        if finite.any():replay_error=max(replay_error,float(np.max(np.abs(got[finite]-expected[finite]))))
        values+=expected.size
        for h in [32,128]:
            scores=metric(expected,d['reference'],h)
            rows=[z for z in case['rows'] if z['method']==name and z['kind']=='trajectory' and z['horizon']==h]
            for row,score in zip(sorted(rows,key=lambda q:q['path']),scores):score_error=max(score_error,abs(score-row['score']));assert abs(score-row['score'])<1e-10
            assert abs(float(np.median(scores))-case['summary'][name][str(h)]['median_nrmse'])<1e-10
        for j in range(8):
            a=expected[2*j+1]-expected[2*j];b=d['reference'][2*j+1,1:]-d['reference'][2*j,1:]
            bad=(not np.isfinite(a).all()) or np.max(np.abs(expected[2*j:2*j+2]))>1e4
            value=1e12 if bad else float(np.sqrt(np.mean((a-b)**2))/max(float(np.sqrt(np.mean(b*b))),.05))
            row=next(z for z in case['rows'] if z['method']==name and z['kind']=='action_response' and z['path']==j);assert abs(value-row['score'])<1e-10
            if np.isfinite(a[:32]).all():assert np.max(np.abs(a[:32]))<2e-5
    for j in range(16):
        independent=own_truth(d['physical_reference'][j,0],d['actions'][j],family,change in ['dynamics','joint'])
        reference_error=max(reference_error,float(np.max(np.abs(independent-d['physical_reference'][j]))))
        assert np.allclose(independent,d['physical_reference'][j],atol=1e-4,rtol=1e-4)
    state=(d['physical_reference']-d['mu'])/d['scale']
    if change in ['observation','joint']:
        a,b=np.cos(.5),np.sin(.5);matrix=np.array([[.8*a,-1.2*b],[.8*b,1.2*a]])
        state=(state+.15*np.tanh(state))@matrix.T+np.array([.2,-.15])
    assert np.allclose(state,d['reference'],atol=1e-12,rtol=1e-12)
    for j in range(8):assert np.max(np.abs(d['actions'][2*j,:32]-d['actions'][2*j+1,:32]))==0
    return dict(family=family,seed=seed,case=change,prediction_scalars=values,max_prediction_replay_error=replay_error,max_metric_recompute_error=score_error,max_augmented_lstsq_coefficient_error=coefficient_error,max_independent_reference_error=reference_error,selected_existing_suffices=case['selected_existing_suffices'])

def main():
    result=json.loads((O/'RESULT85_V1.json').read_text());exit_record=json.loads((O/'WORKER_EXIT85_V1.json').read_text());assert exit_record['exit_code']==0
    rows=[]
    for p in sorted(O.glob('CASE85_*_V1.json')):
        row=check_case(json.loads(p.read_text()));rows.append(row);log(dict(kind='independent_case_verified',**row))
    assert len(rows)==result['closed_cases']
    record=dict(status='PASS',utc=utc(),cases=rows,result_sha256=sha(O/'RESULT85_V1.json'),protocol_sha256=sha(C/'PROTOCOL85_V1.md'),verification_protocol_sha256=sha(C/'VERIFICATION_PROTOCOL85_V1.md'),raw_log_sha256=sha(O/'worker85_v1.log'),total_prediction_scalars=sum(z['prediction_scalars'] for z in rows),native_source_hashes=HASHES,no_model_reselected_on_eval=True,local_result_payload_files_written=0)
    write(O/'VERIFICATION85_V2.json',record);print(json.dumps({k:v for k,v in record.items() if k!='cases'},indent=2),flush=True)

if __name__=='__main__':main()
