from common86_v1 import *
from verify_paired86_v1 import network,ordinary
from scipy.linalg import expm

def main():
    r=json.loads((O/'JOINT_DEPENDENCE86_V1.json').read_text());p=json.loads((O/'paired_v1'/'RESULT86_PAIRED_V1.json').read_text());assert sha(O/'paired_v1'/'RESULT86_PAIRED_V1.json')==r['frozen']['result_sha256'] and sha(C/'JOINT_DEPENDENCE86_V1.md')==r['frozen']['protocol_sha256'];assert sha(r['array_path'])==r['array_sha256'];d=np.load(r['array_path'],allow_pickle=False);assert len(np.unique(d['mechanism_ids']))==1024 and np.array_equal(np.sort(d['permutation']),np.arange(1024)) and np.array_equal(d['crossed_initial'],d['retained_initial'][d['permutation']]);assert np.array_equal(np.sort(d['crossed_initial'],axis=0),np.sort(d['retained_initial'],axis=0))
    maxphysical=0.;maxreplay=0.;total=0
    for population in ['retained','crossed']:
        initial=d[population+'_initial'];truth=d[population+'_reference'];history=d[population+'_history']
        for j in range(1024):
            for t in [0,7,8,23]:
                actual=expm(t*.05*d['a'][j])@initial[j]
                if t>=8:maxphysical=max(maxphysical,float(np.max(np.abs(actual-truth[j,t-8]))))
                else:assert np.allclose((actual+d['noise'][j,t]).astype(np.float32),history[j,t],atol=1e-7,rtol=1e-7)
        for fit in p['fits']:
            label=f"{fit['seed']}_{fit['architecture']}_{fit['arm']}";state=torch.load(fit['selected']['checkpoint_path'],map_location='cpu',weights_only=False)['model'];prediction=network(history,state,fit['architecture']);expected=d[population+'_'+label+'_prediction'];maxreplay=max(maxreplay,float(np.max(np.abs(prediction-expected))));assert np.allclose(prediction,expected,atol=1e-4,rtol=1e-4);total+=expected.size
        for selected in p['linear']:
            label=f"linear_{selected['seed']}";prediction=ordinary(history,selected['selected']['ridge']);assert np.allclose(prediction,d[population+'_'+label+'_prediction'],atol=1e-9,rtol=1e-9);total+=prediction.size
        assert np.array_equal(np.repeat(history[:,-1,None],16,axis=1),d[population+'_persistence_prediction']);total+=d[population+'_persistence_prediction'].size
        for row in [x for x in r['results'] if x['population']==population]:
            prefix=population+'_'+row['label'];prediction=d[prefix+'_prediction'];mse=np.mean((prediction-truth)**2,axis=(1,2));nrmse=np.sqrt(mse)/np.maximum(np.sqrt(np.mean(truth**2,axis=(1,2))),.1);failed=(~np.isfinite(prediction).all(axis=(1,2)))|(np.max(np.abs(prediction),axis=(1,2))>1e4);mse[failed]=1e12;nrmse[failed]=1e12;assert np.array_equal(failed,d[prefix+'_failed']) and np.allclose(mse,d[prefix+'_mse'],atol=1e-12,rtol=1e-12) and np.allclose(nrmse,d[prefix+'_nrmse'],atol=1e-12,rtol=1e-12);assert np.isclose(mse.mean(),row['mean_mse'],atol=1e-12,rtol=1e-12) and np.isclose(nrmse.mean(),row['mean_nrmse'],atol=1e-12,rtol=1e-12) and np.isclose(np.median(nrmse),row['median_nrmse'],atol=1e-12,rtol=1e-12) and int(failed.sum())==row['failed']
    assert maxphysical<1e-10 and total==1441792
    out=dict(status='PASS',utc=utc(),result_sha256=sha(O/'JOINT_DEPENDENCE86_V1.json'),prediction_scalars=total,max_scipy_physical_difference=maxphysical,max_numpy_prediction_difference=maxreplay,mechanism_and_initial_marginals_verified=True,shared_noise_verified=True,frozen_models_and_hyperparameters_verified=True,model_selection_changed=False,new_method_validated=False);write(O/'JOINT_VERIFICATION86_V1.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
