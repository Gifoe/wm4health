from common85_v1 import *
import verify85_v2 as independent

def matrices(y,u,kind,spec,L):
    function=np.polynomial.Polynomial([0.,0.,16.,-32.,16.]);df=function.deriv();duration=L*.05
    edge=np.linspace(0,1,L+1);middle=(edge[1:]+edge[:-1])/2
    weights=np.vstack([function(edge[:-1]),4*function(middle),function(edge[1:])])*.05/6
    deriv=np.vstack([df(edge[:-1]),4*df(middle),df(edge[1:])])*.05/(6*duration)
    total=weights.sum();xs=[];ys=[]
    for j in range(len(y)):
        for start in range(0,len(u[j])-L+1,4):
            left=y[j,start:start+L];right=y[j,start+1:start+1+L];mid=(left+right)*.5;action=u[j,start:start+L]
            state=np.stack([left,mid,right]);feature=np.stack([independent.phi(np.concatenate([z,action],axis=1),kind,spec) for z in state])
            xs.append(np.einsum('ij,ijk->k',weights,feature)/total);ys.append(-np.einsum('ij,ijk->k',deriv,state)/total)
    return np.asarray(xs),np.asarray(ys)
def simulate(y0,u,w,kind,spec):
    state=y0.copy();result=[]
    def f(x,a):return independent.phi(np.concatenate([x,a],axis=1),kind,spec)@w
    with np.errstate(over='ignore',invalid='ignore'):
        for a in u.transpose(1,0,2):
            failed=(~np.isfinite(state).all(axis=1))|(np.max(np.abs(state),axis=1)>1e4);s=state.copy();s[failed]=0
            a1=f(s,a);a2=f(s+.025*a1,a);a3=f(s+.025*a2,a);a4=f(s+.05*a3,a);state=state+(.05/6)*(a1+2*a2+2*a3+a4);state[failed]=np.nan;result.append(state.copy())
    return np.stack(result,axis=1)
def normal_path_failure(x):return (not np.isfinite(x).all()) or np.max(np.abs(x))>1e4
SOURCE_CACHE={}
def source_coefficient(data,path,kind,spec,L):
    key=(str(path),kind,L)
    if key not in SOURCE_CACHE:
        a,b=matrices(data['y'],data['u'],kind,spec,L);n=a.shape[1]
        SOURCE_CACHE[key]=np.linalg.lstsq(np.concatenate([a,np.sqrt(1e-6)*np.eye(n)],axis=0),np.concatenate([b,np.zeros((n,2))],axis=0),rcond=None)[0]
    return SOURCE_CACHE[key]
def verify_stage(stage,factorial):
    prefix='ATTRIBUTION' if factorial else 'WEAK';folder=O/stage;assert json.loads((folder/'WORKER_EXIT85_V1.json').read_text())['exit_code']==0
    checked=[];total=0
    for cp in sorted(folder.glob(f'CASE85_{prefix}_*_V1.json')):
        case=json.loads(cp.read_text());seed,change=case['seed'],case['case'];tag=f'brusselator_{seed}_{change}'
        sealpath=folder/f'SELECTION_SEALED85_{prefix}_{tag}_V1.json';assert sha(sealpath)==case['selection_seal_sha256'];seal=json.loads(sealpath.read_text());assert seal['eval_generated'] is False
        path=Path(case['array_path']);assert sha(path)==case['array_sha256'];d=np.load(path,allow_pickle=False);assert datetime.datetime.fromisoformat(seal['utc']).timestamp()<=path.stat().st_mtime
        coefpath=Path(seal['coefficient_path']);assert sha(coefpath)==seal['coefficient_sha256'];coefs=np.load(coefpath,allow_pickle=False);spec={k:coefs[k] for k in ['exponents','frequency','phase']}
        trainpath=Path(seal['calibration_path']);assert sha(trainpath)==seal['calibration_sha256'];train=np.load(trainpath,allow_pickle=False)
        sp=Path(seal['source_path']) if factorial else Path(CONFIG['data_root'])/'cycle85'/'anchored_screen_v1'/f'brusselator_{seed}'/'source_trainval_v1.npz'
        assert sha(sp)==(seal['source_sha256'] if factorial else seal['source_data_sha256']);sd=np.load(sp,allow_pickle=False)
        if factorial:
            assert train['short_y'].shape==(4,65,2) and train['long_y'].shape==(2,129,2)
            assert train['short_val_y'].shape==(2,65,2) and train['long_val_y'].shape==(1,129,2)
            for key,n in [('y',65),('u',64)]:assert np.array_equal(train['long_'+key][:,:n],train['short_'+key][:2])
            for key,n in [('val_y',65),('val_u',64)]:assert np.array_equal(train['long_'+key][:,:n],train['short_'+key][:1])
        maxcoef=0.;maxreplay=0.;scores_checked=0;failure_paths=0
        for label,selected in seal['selected'].items():
            if factorial:
                candidate=[r for r in seal['candidates'] if r['allocation']==selected['allocation'] and r['model_family']==selected['model_family']];key='val32' if label.endswith('_val32') else 'val_long';allocation=selected['allocation']+'_'
            else:candidate=seal['candidates'];key='val32' if label=='weak_val32' else 'val64';allocation=''
            assert selected==min(candidate,key=lambda z:z[key])
            kind,L=selected['kind'],selected['window'];a,b=matrices(train[allocation+'y'],train[allocation+'u'],kind,spec,L);cs=source_coefficient(sd,sp,kind,spec,L);n=a.shape[1];lam=selected['ridge'];tau=selected['source_prior']
            if factorial and selected['model_family']=='affine_residual':
                mask=np.sum(spec['exponents'],axis=1)<=1 if kind=='polynomial3' else np.arange(n)<5;z=a[:,mask]
                delta=np.linalg.lstsq(np.concatenate([z,np.sqrt(lam)*np.eye(int(mask.sum()))]),np.concatenate([b-a@cs,np.zeros((int(mask.sum()),2))]),rcond=None)[0];w=cs.copy();w[mask]+=delta
            else:w=np.linalg.lstsq(np.concatenate([a,np.sqrt(lam)*np.eye(n),np.sqrt(tau)*np.eye(n)]),np.concatenate([b,np.zeros((n,2)),np.sqrt(tau)*cs]),rcond=None)[0]
            actual=coefs[label+'_coef'];maxcoef=max(maxcoef,float(np.max(np.abs(w-actual))));assert np.allclose(w,actual,atol=1e-5,rtol=1e-5)
            expected=d[label+'_prediction'];replayed=simulate(d['initial_observation'],d['actions'],actual,kind,spec)
            for j in range(16):
                eb,rb=normal_path_failure(expected[j]),normal_path_failure(replayed[j]);assert eb==rb
                if eb:failure_paths+=1
                else:assert np.allclose(replayed[j],expected[j],atol=2e-5,rtol=2e-5);maxreplay=max(maxreplay,float(np.max(np.abs(replayed[j]-expected[j]))))
            total+=expected.size
            for h in [32,128]:
                values=independent.metric(expected,d['reference'],h)
                for row in [z for z in case['rows'] if z['method']==label and z['kind']=='trajectory' and z['horizon']==h]:assert abs(values[row['path']]-row['score'])<1e-10;scores_checked+=1
            response=[]
            for j in range(8):
                p=expected[2*j+1]-expected[2*j];t=d['reference'][2*j+1,1:]-d['reference'][2*j,1:]
                bad=normal_path_failure(expected[2*j]) or normal_path_failure(expected[2*j+1]);v=1e12 if bad else float(np.sqrt(np.mean((p-t)**2))/max(float(np.sqrt(np.mean(t*t))),.05));response.append(v)
                row=next(z for z in case['rows'] if z['method']==label and z['kind']=='action_response' and z['path']==j);assert abs(v-row['score'])<1e-10;scores_checked+=1
            sm=case['summary'][label];assert abs(float(np.median(independent.metric(expected,d['reference'],128)))-sm['median128'])<1e-10 and abs(float(np.median(response))-sm['median_response'])<1e-10
        error=0.
        for j in range(16):
            x=independent.own_truth(d['physical_reference'][j,0],d['actions'][j],'brusselator',change in ['dynamics','joint']);error=max(error,float(np.max(np.abs(x-d['physical_reference'][j]))));assert np.allclose(x,d['physical_reference'][j],atol=1e-4,rtol=1e-4)
        if factorial and seed in SEEDS:
            previous=json.loads((O/'weak_controls_v1'/f'CASE85_WEAK_{tag}_V1.json').read_text());prior=np.load(previous['array_path'],allow_pickle=False)
            for key in ['reference','actions','initial_observation']:assert np.array_equal(d[key],prior[key])
            for a,b in [('short_full_field_val32','weak_val32'),('short_full_field_val_long','weak_val64')]:assert np.array_equal(d[a+'_prediction'],prior[b+'_prediction'],equal_nan=True)
        record=dict(seed=seed,case=change,max_augmented_coefficient_difference=maxcoef,max_successful_replay_difference=maxreplay,failed_paths_independently_reproduced=failure_paths,metric_rows_checked=scores_checked,max_independent_reference_difference=error);checked.append(record);log(dict(kind='ordinary_control_verified',stage=stage,**record))
    return dict(cases=len(checked),prediction_scalars=total,checks=checked)
def main():
    a=verify_stage('weak_controls_v1',False);b=verify_stage('attribution_v1',True);assert a['cases']==8 and b['cases']==12
    record=dict(status='PASS',utc=utc(),weak=a,attribution=b,verification_protocol_sha256=sha(C/'CONTROL_VERIFICATION85_V1.md'),weak_result_sha256=sha(O/'weak_controls_v1'/'RESULT85_WEAK_V1.json'),attribution_result_sha256=sha(O/'attribution_v1'/'RESULT85_ATTRIBUTION_V1.json'),same_budget_prefixes_verified=True,all_selectors_validation_only=True,new_method_validated=False,local_result_payload_files_written=0)
    write(O/'CONTROL_VERIFICATION85_V1.json',record);print(json.dumps({k:v for k,v in record.items() if k not in ['weak','attribution']},indent=2),flush=True)

if __name__=='__main__':main()
