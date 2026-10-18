from common85_v1 import *
import verify_controls85_v1 as control
import verify85_v2 as independent

def cal(x,model,inverse=False):
    shape=x.shape;xx=x.reshape(-1,2)
    if model['kind']=='oracle_family':
        p=np.asarray(model['p']);assert -.5<=p[6]<=.5;m=p[:4].reshape(2,2)
        if inverse:
            rhs=np.linalg.solve(m,(xx-p[4:6]).T).T;result=rhs.copy()
            for _ in range(20):result-=(result+p[6]*np.tanh(result)-rhs)/(1+p[6]/np.cosh(np.clip(result,-350,350))**2)
        else:result=np.einsum('ij,kj->ki',m,xx+p[6]*np.tanh(xx))+p[4:6]
    else:
        part=model['inverse' if inverse else 'forward'];result=np.c_[np.ones(len(xx)),xx]@np.array(part['w'])
        if model['kind']=='rbf_residual':
            centers=np.array(part['centers']);scale=np.array(part['scale']);dist=np.sum((xx[:,None,:]/scale-centers[None,:,:]/scale)**2,axis=2);result+=np.exp(-dist/(2*model['bw']**2))@np.array(part['coef'])
    return result.reshape(shape)
def check_ordinary(sensor,s,y):
    for name,x,z in [('forward',s,y),('inverse',y,s)]:
        part=sensor[name];w=np.linalg.lstsq(np.c_[np.ones(len(x)),x],z,rcond=None)[0];assert np.allclose(w,part['w'],atol=1e-12,rtol=1e-12)
        if sensor['kind']=='rbf_residual':
            scale=np.maximum(np.std(x,axis=0),.05);assert np.array_equal(x,part['centers']) and np.array_equal(scale,part['scale'])
            k=np.exp(-np.sum(((x[:,None]-x[None])/scale)**2,axis=2)/(2*sensor['bw']**2));coef=np.linalg.lstsq(k+.01*np.eye(len(x)),z-np.c_[np.ones(len(x)),x]@w,rcond=None)[0];assert np.allclose(coef,part['coef'],atol=1e-10,rtol=1e-10)
def replay(initial,u,w,spec,sensor):return cal(control.simulate(cal(initial,sensor,True),u,w,'polynomial3',spec),sensor)
def main():
    folders=[O/'composed_control_v1',O/'composed_control_v2'];assert json.loads((folders[0]/'WORKER_EXIT85_V1.json').read_text())['exit_code']==1;assert json.loads((folders[1]/'WORKER_EXIT85_V1.json').read_text())['exit_code']==0
    errorlog=(folders[0]/'worker85_v1.log').read_text();assert 'FileExistsError' in errorlog and 'brusselator_2026853_source_v1.npz' in errorlog
    checks=[];count=0;rows_checked=0
    for folder in folders:
        cases=sorted(folder.glob('CASE85_COMPOSED_*_V1.json'));assert len(cases)==8
        for cp in cases:
            r=json.loads(cp.read_text());tag=f"brusselator_{r['seed']}_{r['case']}";sp=folder/f'SELECTION_SEALED85_COMPOSED_{tag}_V1.json';assert sha(sp)==r['selection_seal_sha256'];seal=json.loads(sp.read_text());assert seal['eval_generated'] is False
            for name in ['source','calibration','coefficient']:assert sha(seal[name+'_path'])==seal[name+'_sha256']
            assert sha(r['array_path'])==r['array_sha256'] and datetime.datetime.fromisoformat(seal['utc']).timestamp()<=Path(r['array_path']).stat().st_mtime
            sd=np.load(seal['source_path'],allow_pickle=False);d=np.load(r['array_path'],allow_pickle=False);train=np.load(seal['calibration_path'],allow_pickle=False);coeff=np.load(seal['coefficient_path'],allow_pickle=False);spec={k:coeff[k] for k in ['exponents','frequency','phase']};assert train['y'].shape==(2,129,2) and train['val_y'].shape==(1,129,2)
            for sensor in seal['sensors']:
                if sensor['kind']!='oracle_family':check_ordinary(sensor,train['anchor_source'],train['anchor_target'])
                else:
                    grid=np.random.default_rng(85).normal(size=(50,2))*3;assert np.allclose(cal(cal(grid,sensor),sensor,True),grid,atol=1e-10,rtol=1e-10)
            coeferr=0.;replayerr=0.;valerr=0.;trutherr=0.;failed=0
            for label,selected in seal['selected'].items():
                group=[z for z in seal['candidates'] if (z['sensor_kind']=='oracle_family')==label.startswith('oracle_')];criterion='val32' if label.endswith('val32') else 'val128';assert selected==min(group,key=lambda z:z[criterion])
                sensor=seal['sensors'][selected['sensor_index']];L=selected['source_window'];cs=control.source_coefficient(sd,seal['source_path'],'polynomial3',spec,L);source_pred=control.simulate(sd['val_y'][:,0],sd['val_u'],cs,'polynomial3',spec)
                assert np.median(independent.metric(source_pred,sd['val_y'],32))<=.15 and np.median(independent.metric(source_pred,sd['val_y'],128))<=.35
                w=cs.copy();lam=selected['ridge']
                if lam is not None:
                    a,b=control.matrices(cal(train['y'],sensor,True),train['u'],'polynomial3',spec,L);mask=np.sum(spec['exponents'],axis=1)<=1;z=a[:,mask];delta=np.linalg.lstsq(np.vstack([z,np.sqrt(lam)*np.eye(int(mask.sum()))]),np.vstack([b-a@cs,np.zeros((int(mask.sum()),2))]),rcond=None)[0];w[mask]+=delta
                actual=coeff[label+'_coef'];coeferr=max(coeferr,float(np.max(np.abs(w-actual))));assert np.allclose(w,actual,atol=1e-5,rtol=1e-5)
                val=replay(train['val_y'][:,0],train['val_u'],actual,spec,sensor)
                for h in [32,128]:
                    v=finite_score(float(np.mean(independent.metric(val,train['val_y'],h))));valerr=max(valerr,abs(v-selected['val'+str(h)]));assert np.isclose(v,selected['val'+str(h)],atol=1e-7,rtol=1e-7)
                expected=d[label+'_prediction'];p=replay(d['initial_observation'],d['actions'],actual,spec,sensor);count+=expected.size
                for i in range(16):
                    eb,rb=control.normal_path_failure(expected[i]),control.normal_path_failure(p[i]);assert eb==rb
                    if eb:failed+=1
                    else:assert np.allclose(p[i],expected[i],atol=2e-5,rtol=2e-5);replayerr=max(replayerr,float(np.max(np.abs(p[i]-expected[i]))))
                for h in [32,128]:
                    values=independent.metric(expected,d['reference'],h)
                    for row in [z for z in r['rows'] if z['method']==label and z['kind']=='trajectory' and z['horizon']==h]:assert abs(row['score']-values[row['path']])<1e-10;rows_checked+=1
                response=[]
                for j in range(8):
                    bad=control.normal_path_failure(expected[2*j]) or control.normal_path_failure(expected[2*j+1]);delta=expected[2*j+1]-expected[2*j];truth=d['reference'][2*j+1,1:]-d['reference'][2*j,1:];value=1e12 if bad else float(np.sqrt(np.mean((delta-truth)**2))/max(float(np.sqrt(np.mean(truth**2))),.05));response.append(value);row=next(z for z in r['rows'] if z['method']==label and z['kind']=='action_response' and z['path']==j);assert abs(row['score']-value)<1e-10;rows_checked+=1
                sm=r['summary'][label];h128=independent.metric(expected,d['reference'],128);assert abs(np.median(h128)-sm['median128'])<1e-10 and abs(np.median(response)-sm['median_response'])<1e-10
                assert sm['suffices']==(sm['failed128']==0 and sm['failed_response']==0 and sm['median128']<=.2 and sm['median_response']<=.3)
            for j in range(16):
                x=independent.own_truth(d['physical_reference'][j,0],d['actions'][j],'brusselator',r['case'] in ['dynamics','joint']);trutherr=max(trutherr,float(np.max(np.abs(x-d['physical_reference'][j]))));assert np.allclose(x,d['physical_reference'][j],atol=1e-4,rtol=1e-4)
            prior=O/'attribution_v1'/f'CASE85_ATTRIBUTION_{tag}_V1.json'
            if prior.exists():
                pr=json.loads(prior.read_text());pd=np.load(pr['array_path'],allow_pickle=False)
                for key in ['reference','actions','initial_observation']:assert np.array_equal(d[key],pd[key])
                ps=O/'attribution_v1'/f'SELECTION_SEALED85_ATTRIBUTION_{tag}_V1.json';pt=np.load(json.loads(ps.read_text())['calibration_path'],allow_pickle=False)
                for key in train.files:assert np.array_equal(train[key],pt['long_'+key])
            record=dict(seed=r['seed'],case=r['case'],max_coefficient_difference=coeferr,max_successful_replay_difference=replayerr,max_val_score_difference=valerr,max_independent_truth_difference=trutherr,failed_paths_reproduced=failed);checks.append(record);log(dict(kind='composed_case_verified',**record))
    assert len(checks)==16 and count==262144 and rows_checked==2560
    out=dict(status='PASS',utc=utc(),cases=16,prediction_scalars=count,metric_rows=rows_checked,checks=checks,verification_protocol_sha256=sha(C/'COMPOSED_VERIFICATION85_V1.md'),case_file_sha256={str(p.relative_to(O)):sha(p) for f in folders for p in sorted(f.glob('CASE85_COMPOSED_*_V1.json'))},ordinary_sensor_fits_and_residuals_verified=True,oracle_privileged=True,initial_loader_failure_retained=True,all_selectors_validation_only=True,new_method_validated=False,local_result_payload_files_written=0)
    write(O/'COMPOSED_VERIFICATION85_V1.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['checks','case_file_sha256']},indent=2),flush=True)
if __name__=='__main__':main()
