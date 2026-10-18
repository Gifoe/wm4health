from common88_v1 import *
import numpy as np
from plan_data88_v1 import generate

def main():
    manifest=read(O/'PLAN_DATA88_V1.json');train=generate(2026881,32,8);val=generate(2026882,8,4)
    center=train['observations'].mean(axis=(0,1),dtype=np.float64);scale=np.maximum(train['observations'].std(axis=(0,1),dtype=np.float64),.05)
    replay=[];theta_sets=[]
    for split,regen in [('train',train),('val',val)]:
        row=next(r for r in manifest['files'] if r['split']==split);assert sha(row['path'])==row['sha256']
        regen['observations']=((regen['observations']-center)/scale).astype(np.float32)
        with np.load(row['path']) as z:
            differences={k:float(abs(z[k].astype(np.float64)-v.astype(np.float64)).max()) for k,v in regen.items()}
            assert all(v==0 for v in differences.values()),differences
            theta_sets.append({v.tobytes() for v in z['parameters']})
        replay.append(dict(split=split,arrays=len(differences),maximum_abs_difference=max(differences.values())))
    assert not theta_sets[0].intersection(theta_sets[1])
    result=read(O/'PAIRED_PLAN_RESULTS88_V1.json');assert sha(result['payload_path'])==result['payload_sha256']
    errors=[]
    with np.load(result['payload_path']) as z:
        truth=z['truth'];family=z['family']
        for row in result['rows']:
            choose=family==row['family'];base=z[row['arm']+'_base'][choose];edited=z[f"{row['arm']}_cut{row['cut']}_sign{row['sign']}"][choose];target=z[f"truth_cut{row['cut']}_sign{row['sign']}"][choose]
            delta=edited-base;response=target-truth[choose];early=row['early_steps']
            assert np.array_equal(target[:,:early],truth[choose,:early])
            calc=dict(early_max=float(abs(delta[:,:early]).max()),early_rms=float(np.sqrt(np.mean(delta[:,:early].astype(np.float64)**2))),post_response_error_rms=float(np.sqrt(np.mean((delta[:,early:]-response[:,early:]).astype(np.float64)**2))),factual_mse=float(np.mean((base-truth[choose]).astype(np.float64)**2)),edited_mse=float(np.mean((edited-target).astype(np.float64)**2)))
            errors.extend(abs(v-row[k]) for k,v in calc.items())
    assert max(errors)==0,max(errors)
    stock=read(O/'PLAN_FIT88_V1_stock.json');lag=read(O/'PLAN_FIT88_V1_lagged.json');assert stock['parameters']==lag['parameters'] and stock['initial_sha256']==lag['initial_sha256']
    for arm in ['stock','lagged','acrkn']:
        fit=read(O/f'PLAN_FIT88_V1_{arm}.json');assert fit['updates']==1000 and fit['allow_tf32']==False and fit['matmul_precision']=='highest';assert sha(fit['selected_path'])==fit['selected_sha256'] and sha(fit['final_path'])==fit['final_sha256']
    pinned=read(O/'SOURCE_MANIFEST88_V1.json')
    for row in pinned['files']:assert sha(Path(pinned['source_root'])/row['path'])==row['sha256']
    receipt=dict(utc=utc(),status='DATA_REGENERATION_AND_REPORTED_PAIRED_METRICS_REPLAYED',data_replay=replay,train_val_shared_exact_parameter_vectors=0,metric_scalars_checked=len(errors),metric_max_abs_difference=max(errors),native_source_files_unchanged=len(pinned['files']),matched_stock_lagged_initialization_and_capacity=True,three_fits_updates=1000,independent_evaluation_claim=False,seed_replication_claim=False)
    write(O/'VERIFICATION88_V1.json',receipt);log(dict(stage='plan_verification_closed',**{k:v for k,v in receipt.items() if k!='utc'}))

if __name__=='__main__':main()
