from common88_v1 import *
import numpy as np,torch
from gru_plan88_v1 import OrdinaryGRU
from evaluate_plan88_v1 import predict,group_metrics

def main():
    fit=read(O/'GRU_FIT88_V1.json');ref=read(O/'PAIRED_PLAN_RESULTS88_V1.json');assert sha(ref['payload_path'])==ref['payload_sha256'];assert sha(fit['selected_path'])==fit['selected_sha256']
    with np.load(ref['payload_path']) as z:arrays={k:z[k] for k in z.files if k in ['truth','observations','actions','family','system_id'] or k.startswith('truth_cut')}
    model=OrdinaryGRU().cuda();model.load_state_dict(torch.load(fit['selected_path'],map_location='cpu',weights_only=False)['model']);model.eval();base,var=predict(model,arrays['observations'],arrays['actions']);repeat,_=predict(model,arrays['observations'],arrays['actions']);masked,maskedvar=predict(model,arrays['observations'],arrays['actions'],True)
    replay=float(abs(base-repeat).max());masked_difference=float(abs(base-masked).max())
    # GRU uses an explicit validity bit, not a hard measurement-update mask. Assess
    # masked-input perturbations descriptively; inference always supplies zeros.
    assert replay<=1e-6
    rows=[]
    for cut in [80,88]:
        for sign in [-1,1]:
            a=arrays['actions'].copy();a[:,cut:,0]=sign*.8;p,_=predict(model,arrays['observations'],a);target=arrays[f'truth_cut{cut}_sign{sign}']
            rows.extend(dict(arm='gru',**r) for r in group_metrics(base,p,arrays['truth'],target,arrays['family'],cut,sign));arrays[f'gru_cut{cut}_sign{sign}']=p
    factual=[]
    for f in [0,1]:
        choose=arrays['family']==f;mse=float(np.mean((base[choose]-arrays['truth'][choose]).astype(np.float64)**2));original=fit['selected']['families'][f]['mse'];assert abs(mse-original)<1e-6
        factual.append(dict(family=f,mse=mse,persistence_ratio=mse/ref['reference'][f]['persistence_mse'],train_mean_ratio=mse/ref['reference'][f]['train_mean_mse'],untrained_ratio=mse/fit['validation_trace'][0]['families'][f]['mse']))
    arrays['gru_base']=base;arrays['gru_variance']=var;path=Path(STORAGE['artifact_root'])/'cycle88/gru_predictions88_v1.npz'
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    write(O/'GRU_RESULTS88_V1.json',dict(utc=utc(),status='ORDINARY_GRU_PAIRED_TEST_COMPLETE',parameters=fit['parameters'],selected_step=fit['selected']['step'],factual=factual,rows=rows,exact_repeat_max=replay,masked_input_perturbation_max=masked_difference,masked_inference_values_always_zero=True,payload_path=str(path),payload_sha256=sha(path),new_method=False,held_out_eval_accessed=False))
    log(dict(stage='ordinary_gru_paired_closed',factual=factual,early_max=max(r['early_max'] for r in rows),masked_input_perturbation_max=masked_difference))

if __name__=='__main__':main()
