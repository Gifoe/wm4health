"""Native ablation prefix replay, common-future checks and independent event arithmetic."""
import runtime_profile
import hashlib,importlib.metadata,json,pathlib,sys,time
import numpy as np
import torch
from safetensors.torch import load_file
from verify_event_audit import counts
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent/'cycle18';FOLDER=ROOT/'outputs/attribution'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    start=time.perf_counter();torch.set_num_threads(1)
    receipt=FOLDER/'independent_verification.json';assert not receipt.exists()
    result=json.loads((FOLDER/'results.json').read_text());source=BASE/'upstream/DynaMix-python'
    sys.path.insert(0,str(source/'src'))
    from dynamix.model.dynamix import DynaMix
    from dynamix.model.forecaster import DynaMixForecaster
    cfg=json.loads((BASE/'outputs/external/DynaMix/config_3d-alrnn-v1.0.json').read_text())['architecture']
    weight=BASE/'outputs/external/DynaMix/dynamix-3d-alrnn-v1.0.safetensors'
    assert sha(weight)==result['checkpoint_sha256']
    replay_errors={};checked=0;max_mase_error=0.;common_truth=None
    for row in result['rows']:
        p=FOLDER/row['array_file'];assert sha(p)==row['array_sha256'];saved=np.load(p,allow_pickle=False)
        pred=saved['prediction'];truth=saved['truth'];context=saved['reference_context']
        assert pred.shape==truth.shape==(10000,3) and np.isfinite(pred).all()
        data=np.load(source/'notebooks/test_data'/(row['system']+'.npy'),allow_pickle=False)
        origin=row['origin'];assert np.array_equal(truth,data[origin:origin+10000].astype('float32'))
        if origin==2000 and row['system']=='chua':
            if common_truth is None:common_truth=truth.copy()
            assert np.array_equal(common_truth,truth) and np.array_equal(context,data[972:2000].astype('float32'))
        x=pred[:,0].astype('float64')
        for mode,threshold_data in [('context_thresholds',context[:,0]),('own_rank_diagnostic',x)]:
            lo,hi=np.quantile(threshold_data,[.5,.9])
            for dense in [False,True]:
                num,den=counts(x,lo,hi,dense);expected=row['event_summaries'][mode+('_all_offsets' if dense else '')]
                actual=num.sum(0)/den.sum()
                assert np.array_equal(actual,np.asarray(expected['first_passage_CDF']));checked+=1
        numerator=np.mean(np.abs(pred[:10].astype('float64')-truth[:10]),axis=0)
        denominator=np.mean(np.abs(np.diff(truth.astype('float64'),axis=0)),axis=0)
        mase=float(np.mean(numerator/(denominator+1e-8)))
        error=abs(mase-row['global_metrics']['MASE_true_denominator']);assert error<2e-6;max_mase_error=max(max_mase_error,error)
        # Three deterministic zero-noise arms and one seed for each common-origin arm.
        if row['condition']=='zero_attention_noise' or row['seed']==2021:
            model=DynaMix(**cfg).eval();model.load_state_dict(load_file(str(weight)),strict=True)
            if row['condition']=='zero_attention_noise':
                with torch.no_grad():model.gating_network.sigma.zero_()
                context_tensor=torch.tensor(data[:origin],dtype=torch.float32);standardize=True;inverse=None
            elif row['condition']=='short_native':
                context_tensor=torch.tensor(data[972:2000],dtype=torch.float32);standardize=True;inverse=None
            elif row['condition']=='long_native_coordinates':
                context_tensor=torch.tensor(data[:2000],dtype=torch.float32);standardize=True;inverse=None
            else:
                short=torch.tensor(data[972:2000],dtype=torch.float32);mean=short.mean(0);std=short.std(0).clamp(min=1e-6)
                context_tensor=(torch.tensor(data[:2000],dtype=torch.float32)-mean)/std;standardize=False;inverse=(mean,std)
            torch.manual_seed(row['seed']);replay=DynaMixForecaster(model).forecast(context_tensor,64,standardize=standardize)
            if inverse is not None:replay=replay*inverse[1]+inverse[0]
            e=float(np.max(np.abs(replay.numpy()-pred[:64])));assert e==0
            replay_errors[row['system']+'_'+row['condition']]=e
    assert sha(weight)==result['checkpoint_sha256']
    info={'pass':True,'arrays_hash_verified':len(result['rows']),'independent_event_cohorts_exact':checked,
        'native_64_step_prefix_replay_errors':replay_errors,'same_origin_same_future_verified':True,
        'max_true_denominator_MASE_arithmetic_error':max_mase_error,'checkpoint_unchanged':True,
        'script_sha256':sha(pathlib.Path(__file__)),
        'environment':{'python':sys.version,'packages':{k:importlib.metadata.version(k) for k in
            ['numpy','scipy','torch','safetensors','scikit-learn','statsmodels','huggingface-hub','matplotlib']}},
        'elapsed_seconds':time.perf_counter()-start,'objective_achieved':False}
    receipt.write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info),flush=True)
if __name__=='__main__':main()
