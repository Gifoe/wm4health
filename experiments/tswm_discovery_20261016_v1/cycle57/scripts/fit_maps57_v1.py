"""Train-state mean diagnostics only; no heldout/validation array opens."""
from eval_utils57 import *

def main():
    source_check();seal=read(OUT/'MODEL_SEAL57_V1.json')
    assert seal['status']=='ALL_32_TERMINAL_MODELS_SEALED_HELDOUT_UNOPENED'
    assert seal['heldout_opened'] is False
    assert not (OUT/'HELDOUT_OPEN_RECEIPT57_V1.json').exists()
    p=paths();folder=p['artifact_root']/'maps_v1';folder.mkdir(parents=True,exist_ok=False)
    with np.load(p['data_root']/'train_inputs_v1.npz') as archive:inputs={k:archive[k] for k in archive.files}
    with np.load(p['data_root']/'train_truth_v1.npz') as archive:z=archive['z']
    records=[]
    for item in seal['models']:
        run=OUT/'runs'/item['tag']/'result.json'
        assert sha(run)==item['result_sha256'];record=read(run)
        model,dtype=load_model(record)
        means=infer(model,dtype,inputs[record['sensor']],inputs['actions'])
        X=np.c_[means.reshape(-1,2),np.ones(means.shape[0]*means.shape[1])]
        Y=z.reshape(-1,2);W=np.linalg.lstsq(X,Y,rcond=None)[0]
        pred=X@W;error=((pred-Y)**2).mean();r2=1-((pred-Y)**2).sum()/((Y-Y.mean(0))**2).sum()
        dest=folder/(item['tag']+'.npz')
        with dest.open('xb') as stream:np.savez(stream,W=W)
        records.append({'tag':item['tag'],'map':str(dest),'map_sha256':sha(dest),'checkpoint_sha256':record['checkpoint_sha256'],'train_smoothed_state_mean_mse':float(error),'train_smoothed_state_mean_r2':float(r2),'linear_map_condition':float(np.linalg.cond(W[:2])),'map_role':'privileged affine train-mean diagnostic only; not a gauge certificate or posterior covariance map'})
        del model;torch.cuda.empty_cache()
    assert len(records)==32
    output={'utc':utc(),'status':'ALL_32_TRAIN_ONLY_MEAN_MAPS_SEALED_HELDOUT_UNOPENED','models_sha256':sha(OUT/'MODEL_SEAL57_V1.json'),'maps':records,'script_sha256':sha(__file__),'eval_utils_sha256':sha(ROOT/'scripts/eval_utils57.py'),'decision_rule_sha256':sha(ROOT/'DECISION_RULE_V1.md'),'train_inputs_sha256':sha(p['data_root']/'train_inputs_v1.npz'),'train_truth_sha256':sha(p['data_root']/'train_truth_v1.npz'),'heldout_opened':False,'validation_opened':False,'qualification_roll_max_error':qualify_roll()}
    dump(OUT/'MAP_SEAL57_V1.json',output)
    print(json.dumps({'status':output['status'],'maps':len(records)}),flush=True)

if __name__=='__main__':main()
