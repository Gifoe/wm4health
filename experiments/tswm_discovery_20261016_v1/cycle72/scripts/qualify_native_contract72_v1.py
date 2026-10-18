"""Execute the pinned preprocessor/loader on constructed, non-benchmark sentinels."""
import sys,pathlib,types
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v1 import *
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
selection=read(ROOT/'SOURCE_SELECTION72_V1.json');upstream=PROJECT/'upstream'/'epi-cf-benchmark'
for spec in selection['files']:assert sha(upstream/spec['path'])==spec['sha256']
sys.path.insert(0,str(upstream/'src'))
from epicf_benchmark import preprocessing as pre
from epicf_benchmark.data import FluDataset
from epicf_benchmark.kde import collect_x,flatten_features

base=storage('artifact')/'native_contract_v1';base.mkdir(exist_ok=False)
nunits=3;ntime=4;n=nunits*ntime;t=np.arange(n,dtype=float)
df=pd.DataFrame({'population_size':1000+100*np.repeat(np.arange(nunits),ntime),
    'treatment':(t.astype(int)%2),'0-F':10+t,'0-CF':100+t})
for k,c in enumerate(pre.AGE_COLUMNS):df[c]=10.+k+0.1*t
for k,c in enumerate(pre.MOBILITY_COLUMNS):df[c]=-5+k+t
for k,c in enumerate(pre.SINGLE_SERM_FACTUAL):df[c]=1000.+100*k+t
for k,c in enumerate(pre.SINGLE_SERM_COUNTERFACTUAL):df[c]=5000.+100*k+t
raw=base/'constructed_single.csv';df.to_csv(raw,index=False)
pre.preprocess_single(raw,base/'single_a')
changed=df.copy()
for c in pre.SINGLE_SERM_COUNTERFACTUAL:changed[c]+=13377
changed.to_csv(base/'constructed_single_changed.csv',index=False)
pre.preprocess_single(base/'constructed_single_changed.csv',base/'single_b')
train_equal={name:sha(base/'single_a'/name)==sha(base/'single_b'/name) for name in ('y_train.csv','x_train.csv','a_train.csv','outcome_stats.json')}
assert all(train_equal.values())
x_a=pd.read_csv(base/'single_a'/'x_test.csv');x_b=pd.read_csv(base/'single_b'/'x_test.csv')
assert list(x_a.columns[-4:])==['S','E','R','M']
assert np.array_equal(x_a.iloc[:,-4:].to_numpy(),df[pre.SINGLE_SERM_COUNTERFACTUAL].to_numpy())
assert np.array_equal(x_a.iloc[:,:-4].to_numpy(),x_b.iloc[:,:-4].to_numpy())
assert np.array_equal(x_b.iloc[:,-4:].to_numpy()-x_a.iloc[:,-4:].to_numpy(),np.full((n,4),13377.))
assert sha(base/'single_a'/'y_test.csv')==sha(base/'single_b'/'y_test.csv')
def dataset(folder,units,policy):
    return FluDataset(folder/'y_test.csv',folder/'a_test.csv',folder/'x_test.csv',policy_dim=policy,sequence_length=ntime,num_covariates=17,num_units=units)
ds_a=dataset(base/'single_a',nunits,1);ds_b=dataset(base/'single_b',nunits,1)
inputs_a=collect_x(ds_a,2);inputs_b=collect_x(ds_b,2)
assert inputs_a.shape==(nunits,ntime,18) and np.array_equal(inputs_b[:,:,-4:]-inputs_a[:,:,-4:],np.full((nunits,ntime,4),13377.))
flat_a=flatten_features(inputs_a);flat_b=flatten_features(inputs_b)
assert flat_a.shape==(nunits,ntime*18) and int(np.sum(flat_a!=flat_b))==nunits*ntime*4

multi=df[['population_size',*pre.AGE_COLUMNS,*pre.MOBILITY_COLUMNS]].copy()
multi['school_treatment']=np.arange(n)%2;multi['occ_treatment']=(np.arange(n)//2)%2
for k,suffix in enumerate(['F-F',*pre.MULTI_COUNTERFACTUALS]):
    multi[suffix]=20.+k*100+t
    for j,prefix in enumerate(['susceptible','exposed','recovered','dead']):multi[prefix+'_'+suffix]=2000.+10000*k+100*j+t
multi.to_csv(base/'constructed_multi.csv',index=False)
pre.preprocess_multi(base/'constructed_multi.csv',base/'multi')
shape_error=None
try:dataset(base/'multi',nunits,2)
except ValueError as e:shape_error=dict(type=type(e).__name__,message=str(e))
assert shape_error is not None
ds_m=dataset(base/'multi',3*nunits,2);assert len(ds_m)==3*nunits and ds_m.inputs[0].shape==(ntime,19)
expected_policy=np.concatenate([multi[['school_treatment','occ_treatment']].assign(school_treatment=1-multi['school_treatment']),
    multi[['school_treatment','occ_treatment']].assign(occ_treatment=1-multi['occ_treatment']),
    1-multi[['school_treatment','occ_treatment']]],axis=0)
assert np.array_equal(ds_m.policy.reshape(-1,2),expected_policy)
obj=dict(utc=utc(),status='PINNED_NATIVE_INPUT_CONTRACT_EXECUTED_ON_CONSTRUCTED_SENTINELS',
    revision=selection['revision'],native_functions=['preprocess_single','preprocess_multi','FluDataset','collect_x','flatten_features'],
    single_counterfactual_states_directly_enter_loader=True,number_changed_test_input_values=int(np.sum(inputs_a!=inputs_b)),
    train_exports_unchanged_under_counterfactual_mediator_change=train_equal,counterfactual_target_export_unchanged=True,
    native_kde_flattened_input_contains_whole_supplied_sequence=True,
    multi_counterfactual_scenarios=3,multi_test_units_multiplier=3,
    multi_default_training_unit_count_cannot_reshape_stacked_test=True,constructed_shape_error=shape_error,
    explicit_threefold_test_unit_count_succeeds=True,benchmark_numeric_targets_accessed=False,
    performance_evidence=False,new_method=False,accepted_final_directions=0,
    artifacts=dict(storage='artifact_root/cycle72/native_contract_v1',synthetic_only=True),
    runtime=dict(torch=torch.__version__,numpy=np.__version__,cuda_available=torch.cuda.is_available(),numeric_threads=torch.get_num_threads(),affinity=AFFINITY))
dump(OUT/'NATIVE_CONTRACT_QUALIFICATION72_V1.json',obj)
print(json.dumps({k:obj[k] for k in ['status','single_counterfactual_states_directly_enter_loader','train_exports_unchanged_under_counterfactual_mediator_change','multi_test_units_multiplier','benchmark_numeric_targets_accessed','performance_evidence']}),flush=True)
