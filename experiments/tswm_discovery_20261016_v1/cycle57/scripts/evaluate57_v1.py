"""One heldout opening after all models/maps are sealed; fixed planned metrics."""
from eval_utils57 import *
from scipy.stats import t as student_t
from math57 import raw_rho

def summarize(v):
    v=np.asarray(v);se=v.std(ddof=1)/np.sqrt(len(v));q=student_t.ppf(.975,len(v)-1)
    return {'mean':float(v.mean()),'sequence_mc_se':float(se),'sequence_interval_95':[float(v.mean()-q*se),float(v.mean()+q*se)],'sequences':len(v),'interval_scope':'conditional on this fitted model and fixed simulator data law; not seed uncertainty'}

def main():
    pins=source_check();maps=read(OUT/'MAP_SEAL57_V1.json');models=read(OUT/'MODEL_SEAL57_V1.json')
    assert maps['status']=='ALL_32_TRAIN_ONLY_MEAN_MAPS_SEALED_HELDOUT_UNOPENED'
    assert maps['models_sha256']==sha(OUT/'MODEL_SEAL57_V1.json')
    assert maps['eval_utils_sha256']==sha(ROOT/'scripts/eval_utils57.py')
    assert maps['decision_rule_sha256']==sha(ROOT/'DECISION_RULE_V1.md')
    assert maps['heldout_opened'] is False and models['heldout_opened'] is False
    for item in models['models']:
        assert sha(item['checkpoint'])==item['checkpoint_sha256']
        assert sha(OUT/'runs'/item['tag']/'result.json')==item['result_sha256']
    for item in maps['maps']:assert sha(item['map'])==item['map_sha256']
    receipt={'utc':utc(),'status':'FIRST_HELDOUT_SCORING_OPEN_AFTER_ALL_32_MODEL_AND_MAP_SEALS','model_seal_sha256':sha(OUT/'MODEL_SEAL57_V1.json'),'map_seal_sha256':sha(OUT/'MAP_SEAL57_V1.json'),'decision_rule_sha256':sha(ROOT/'DECISION_RULE_V1.md'),'script_sha256':sha(__file__),'eval_utils_sha256':sha(ROOT/'scripts/eval_utils57.py'),'roll_qualification_max_error':qualify_roll(),'training_complete':True,'score_based_selection':False}
    dump(OUT/'HELDOUT_OPEN_RECEIPT57_V1.json',receipt)
    p=paths();artifact=p['artifact_root']/'heldout_v1';artifact.mkdir(parents=True,exist_ok=False)
    with np.load(p['data_root']/'heldout_inputs_v1.npz') as archive:inputs={k:archive[k] for k in archive.files}
    with np.load(p['data_root']/'heldout_truth_v1.npz') as archive:
        states=archive['z'];true_rho=float(archive['rho']);true_angles=archive['angles'];r=float(archive['observation_variance'])
    N=len(states);prefix=12;horizons=[1,5,10,20]
    future=inputs['actions'][:,prefix-1:prefix-1+20]
    plans={'factual':future,'zero':np.zeros_like(future),'one':np.ones_like(future),'alternating':np.broadcast_to(np.arange(20)%2,future.shape)}
    records=[];oracle_cache={}
    for sensor in SENSORS:
        observed=inputs[sensor][:,:prefix].astype('float64')
        # The analytic oracle sees only the same prefix and known simulator parameters.
        y=np.sinh(observed) if sensor=='asinh' else observed
        theta=torch.tensor([raw_rho(true_rho),*true_angles],dtype=torch.float64,device='cuda')
        with torch.no_grad():
            q=posterior(theta,torch.full((2,),1/r,dtype=torch.float64,device='cuda'),torch.tensor(y,device='cuda'),torch.tensor(inputs['actions'][:,:prefix-1],device='cuda'),jitter=0.)
            mean=q['m'][:,-1].cpu().numpy()
        oracle_cache[sensor]={name:roll(mean,true_rho,true_angles,actions)[:,horizons] for name,actions in plans.items()}
        oracle_cache[sensor]['filtered_mean']=mean
    for item in models['models']:
        record=read(OUT/'runs'/item['tag']/'result.json');sensor=record['sensor']
        maprec=next(x for x in maps['maps'] if x['tag']==item['tag'])
        with np.load(maprec['map']) as archive:W=archive['W']
        model,dtype=load_model(record)
        filtered=infer(model,dtype,inputs[sensor][:,:prefix],inputs['actions'][:,:prefix-1])[:,-1]
        rho,angles=parameter_values(model)
        predictions={name:roll(filtered,rho,angles,actions)[:,horizons]@W[:2]+W[2] for name,actions in plans.items()}
        filtered_physical=filtered@W[:2]+W[2];true_z=states[:,prefix-1]
        r2=1-((filtered_physical-true_z)**2).sum()/((true_z-true_z.mean(0))**2).sum()
        conditional=np.stack([((predictions[name]-oracle_cache[sensor][name])**2).mean(-1) for name in plans])
        effect=np.stack([((predictions[name]-predictions['factual']-oracle_cache[sensor][name]+oracle_cache[sensor]['factual'])**2).mean(-1) for name in ['zero','one','alternating']])
        factual=((predictions['factual']-states[:,np.array(horizons)+prefix-1])**2).mean(-1)
        primary=effect[:,:,-1].mean(0)
        assert np.isfinite(primary).all() and np.isfinite(conditional).all()
        dest=artifact/(item['tag']+'.npz')
        with dest.open('xb') as stream:np.savez(stream,primary=primary,conditional=conditional,effect=effect,factual=factual,filtered=filtered_physical,oracle_filtered=oracle_cache[sensor]['filtered_mean'],predictions=np.stack([predictions[k] for k in plans]),oracle=np.stack([oracle_cache[sensor][k] for k in plans]))
        result={'tag':item['tag'],'sensor':sensor,'method':record['method'],'seed':record['seed'],'stochastic_fit':record['stochastic_fit'],'rho':rho,'absolute_rho_error':abs(rho-true_rho),'filtered_state_r2':float(r2),'filtered_conditional_mean_mse':float(((filtered_physical-oracle_cache[sensor]['filtered_mean'])**2).mean()),'train_map_r2':maprec['train_smoothed_state_mean_r2'],'primary_h20_plan_effect_mse':summarize(primary),'horizons':horizons,'plan_order':list(plans),'conditional_mean_mse':{name:[float(x) for x in conditional[i].mean(0)] for i,name in enumerate(plans)},'changed_plan_effect_mse':{name:[float(x) for x in effect[i].mean(0)] for i,name in enumerate(['zero','one','alternating'])},'factual_realized_state_mse':[float(x) for x in factual.mean(0)],'arrays':str(dest),'arrays_sha256':sha(dest),'checkpoint_sha256':record['checkpoint_sha256'],'map_sha256':maprec['map_sha256'],'training_seconds':record['training_seconds'],'peak_training_cuda_bytes':record['peak_training_cuda_bytes'],'parameters':record['description']['parameters'],'dtype':record['dtype'],'normalizer_tracking':record['normalizer_tracking']}
        records.append(result)
        print(json.dumps({'tag':item['tag'],'status':'HELDOUT_SCORED','primary':result['primary_h20_plan_effect_mse']['mean']}),flush=True)
        del model;torch.cuda.empty_cache()
    assert source_check()==pins and len(records)==32
    result={'utc':utc(),'status':'ALL_32_MODELS_SCORED_FIXED_HELDOUT','records':records,'primary_endpoint':'h20 conditional-mean changed-plan effect MSE, average of zero/one/alternating plans','data_seed':5700,'heldout_sequences':N,'heldout_inputs_sha256':sha(p['data_root']/'heldout_inputs_v1.npz'),'heldout_truth_sha256':sha(p['data_root']/'heldout_truth_v1.npz'),'open_receipt_sha256':sha(OUT/'HELDOUT_OPEN_RECEIPT57_V1.json'),'script_sha256':sha(__file__),'eval_utils_sha256':sha(ROOT/'scripts/eval_utils57.py'),'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'scope':'Synthetic controlled linear latent dynamics with paired linear/nonlinear sensing. Privileged train-state mean readout, no covariance calibration or real-data evidence.'}
    dump(OUT/'EVALUATION57_V1.json',result)
    print(json.dumps({'status':result['status'],'models':32,'accepted_final_directions':0}),flush=True)

if __name__=='__main__':main()
