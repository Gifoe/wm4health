from common88_v1 import *
import numpy as np,torch
from plan_models88_v1 import make_model,qualify_sources
from plan_data88_v1 import observe,advance

torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
Q=65;T=96;START=48

def predict(model,obs,act,perturb_masked=False):
    chunks=[];variances=[]
    with torch.no_grad():
        for b in range(0,len(obs),16):
            x=obs[b:b+16,:T].copy();x[:,Q:]=7 if perturb_masked else 0
            mask=np.ones((len(x),T,1),bool);mask[:,Q:]=False
            x,a,m=[torch.as_tensor(z,device='cuda') for z in [x,act[b:b+16],mask]]
            mean,var=model(x,a,m)[:2]
            assert torch.isfinite(mean).all() and torch.isfinite(var).all() and (var>0).all()
            chunks.append(mean[:,Q-1:].cpu().numpy());variances.append(var[:,Q-1:].cpu().numpy())
    return np.concatenate(chunks),np.concatenate(variances)

def group_metrics(base,pred,true,edited_true,family,cut,sign):
    early=cut-(Q-1);rows=[]
    for f in [0,1]:
        choose=family==f;delta=pred[choose]-base[choose];response=edited_true[choose]-true[choose]
        rows.append(dict(family=f,cut=cut,sign=sign,early_steps=early,early_rms=float(np.sqrt(np.mean(delta[:,:early].astype(np.float64)**2))),early_max=float(abs(delta[:,:early]).max()),true_early_max=float(abs(response[:,:early]).max()),post_true_response_rms=float(np.sqrt(np.mean(response[:,early:].astype(np.float64)**2))),post_model_response_rms=float(np.sqrt(np.mean(delta[:,early:].astype(np.float64)**2))),post_response_error_rms=float(np.sqrt(np.mean((delta[:,early:]-response[:,early:]).astype(np.float64)**2))),factual_mse=float(np.mean((base[choose]-true[choose]).astype(np.float64)**2)),edited_mse=float(np.mean((pred[choose]-edited_true[choose]).astype(np.float64)**2))))
    return rows

def main():
    qualify_sources();manifest=read(O/'PLAN_DATA88_V1.json');row=next(r for r in manifest['files'] if r['split']=='val');assert sha(row['path'])==row['sha256']
    with np.load(row['path']) as z:data={k:z[k] for k in z.files}
    observations=data['observations'][:,START:START+T+1];act=data['actions'][:,START:START+T].copy();truth=observations[:,Q:];family=data['family'];theta=data['parameters'];center=data['center'];scale=data['scale'];noise=data['observation_noise'];states=data['states']
    persistence=np.repeat(observations[:,Q-1:Q],T-Q+1,axis=1)
    zero=np.zeros_like(truth);reference=[]
    for f in [0,1]:
        choose=family==f;reference.append(dict(family=f,persistence_mse=float(np.mean((persistence[choose]-truth[choose]).astype(np.float64)**2)),train_mean_mse=float(np.mean(truth[choose].astype(np.float64)**2))))
    branches={};continuation_errors=[]
    # Source state used only to construct paired simulator truth, never by a learner.
    for cut in [80,88]:
        for sign in [-1,1]:
            a=act.copy();a[:,cut:,0]=sign*.8;target=truth.copy();normal_replay=truth.copy()
            for f in [0,1]:
                ids=np.flatnonzero(family==f);state=states[ids,START+cut].copy();normal_state=state.copy()
                for t in range(cut,T):
                    state=advance(state,a[ids,t].astype(np.float64),theta[ids],f)
                    normal_state=advance(normal_state,act[ids,t].astype(np.float64),theta[ids],f)
                    sensed=(observe(state)+noise[ids,START+t+1]).astype(np.float32)
                    normal_sensed=(observe(normal_state)+noise[ids,START+t+1]).astype(np.float32)
                    target[ids,t-(Q-1)]=((sensed-center)/scale).astype(np.float32)
                    normal_replay[ids,t-(Q-1)]=((normal_sensed-center)/scale).astype(np.float32)
            error=float(abs(normal_replay-truth).max());continuation_errors.append(dict(cut=cut,sign=sign,max_standardized_error=error))
            assert error<=1e-5,('factual-continuation-replay',error)
            assert np.array_equal(target[:,:cut-(Q-1)],truth[:,:cut-(Q-1)])
            branches[(cut,sign)]=(a,target)
    raw=Path(STORAGE['artifact_root'])/'cycle88';raw.mkdir(parents=True,exist_ok=True);arrays=dict(truth=truth,observations=observations,actions=act,family=family,system_id=data['system_id'])
    all_rows=[];fits=[];replays=[]
    for arm in ['stock','lagged','acrkn']:
        fit=read(O/f'PLAN_FIT88_V1_{arm}.json');assert sha(fit['selected_path'])==fit['selected_sha256'];model=make_model(arm);saved=torch.load(fit['selected_path'],map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);model.eval()
        base,var=predict(model,observations,act);again,_=predict(model,observations,act);masked,masked_var=predict(model,observations,act,True)
        replay=float(abs(base-again).max());mask_delta=float(abs(base-masked).max());mask_v=float(abs(var-masked_var).max());assert replay==0;assert mask_delta<=1e-6 and mask_v<=1e-6
        replays.append(dict(arm=arm,exact_repeat_max=replay,masked_observation_mean_max=mask_delta,masked_observation_variance_max=mask_v))
        factual=[]
        for f in [0,1]:
            choose=family==f;mse=float(np.mean((base[choose]-truth[choose]).astype(np.float64)**2));original=fit['selected']['families'][f]['mse'];assert abs(mse-original)<1e-6
            init=fit['validation_trace'][0]['families'][f]['mse'];ref=reference[f]
            factual.append(dict(family=f,mse=mse,persistence_ratio=mse/ref['persistence_mse'],train_mean_ratio=mse/ref['train_mean_mse'],untrained_ratio=mse/init))
        fits.append(dict(arm=arm,selected_step=saved['step'],parameters=fit['parameters'],initial_sha256=fit['initial_sha256'],source_competent_original_gate=fit['source_competent'],factual=factual))
        arrays[arm+'_base']=base;arrays[arm+'_variance']=var
        for (cut,sign),(a,target) in branches.items():
            edited,_=predict(model,observations,a);r=group_metrics(base,edited,truth,target,family,cut,sign)
            all_rows.extend(dict(arm=arm,**v) for v in r);arrays[f'{arm}_cut{cut}_sign{sign}']=edited
        del model;torch.cuda.empty_cache()
    assert fits[0]['initial_sha256']==fits[1]['initial_sha256']
    for (cut,sign),(_,target) in branches.items():arrays[f'truth_cut{cut}_sign{sign}']=target
    payload=raw/'paired_plan_predictions88_v1.npz'
    with payload.open('xb') as f:np.savez_compressed(f,**arrays)
    ordinary=[]
    for f in [0,1]:
        stock=fits[0]['factual'][f];lag=fits[1]['factual'][f]
        original_criteria=lag['persistence_ratio']<.9 and lag['mse']<=1.05*stock['mse'] and all(r['early_max']<=1e-6 for r in all_rows if r['arm']=='lagged' and r['family']==f)
        ordinary.append(dict(family=f,lagged_over_stock_mse=lag['mse']/stock['mse'],ordinary_fix_passes_original_rejection_gate=original_criteria))
    result=dict(utc=utc(),status='DEVELOPMENTAL_PAIRED_PLAN_TEST_COMPLETE',fits=fits,reference=reference,rows=all_rows,ordinary_controls=ordinary,replays=replays,factual_continuation_replay=continuation_errors,source_commit=read(O/'SOURCE_MANIFEST88_V1.json')['commit'],source_files_unchanged=True,payload_path=str(payload),payload_sha256=sha(payload),protocol_sha256=sha(C/'PLAN_CAUSALITY_PROTOCOL88_V1.md'),developmental_val_used_for_selection=True,independent_test_pool_accessed=False,optimizer_seeds=1,new_method_claim_accepted=False)
    qualify_sources();write(O/'PAIRED_PLAN_RESULTS88_V1.json',result);log(dict(stage='paired_plan_test_closed',fits=fits,ordinary_controls=ordinary,rows=len(all_rows)))

if __name__=='__main__':main()
