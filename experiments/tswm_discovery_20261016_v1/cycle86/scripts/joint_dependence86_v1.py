from common86_v1 import *
from train86_v1 import Predictor,flow,scores,linear_predict
from sample86_v1 import population

def main():
    stage=O/'paired_v1';result=json.loads((stage/'RESULT86_PAIRED_V1.json').read_text());assert json.loads((stage/'WORKER_EXIT86_V1.json').read_text())['exit_code']==0
    seed=2026865;rates,angles,a,initial=population(seed);full=flow(a,initial[:,0],np.linspace(0,4,81));ids=np.flatnonzero(np.max(np.abs(full),axis=(1,2))<3.)[:1024];assert len(ids)==1024;a=a[ids];rates=rates[ids];retained=initial[ids,0];permutation=np.random.default_rng(np.random.SeedSequence([seed,9000])).permutation(1024);crossed=retained[permutation];noise=np.random.default_rng(np.random.SeedSequence([seed,9001])).normal(0,.01,(1024,24,2));saved=dict(mechanism_ids=ids,a=a,rates=rates,retained_initial=retained,crossed_initial=crossed,permutation=permutation,noise=noise);results=[]
    frozen=dict(result_sha256=sha(stage/'RESULT86_PAIRED_V1.json'),selection_seal_sha256=sha(stage/'SELECTION_SEALED86_V1.json'),protocol_sha256=sha(C/'JOINT_DEPENDENCE86_V1.md'));write(O/'JOINT_DEPENDENCE_FROZEN86_V1.json',dict(utc=utc(),**frozen,model_selection_changed=False))
    for name,init in [('retained',retained),('crossed',crossed)]:
        clean=flow(a,init,np.arange(24)*.05);assert np.isfinite(clean).all();noisy=(clean+noise).astype(np.float32);history=noisy[:,:8];truth=clean[:,8:];saved[name+'_history']=history;saved[name+'_reference']=truth
        predictions=[]
        for fit in result['fits']:
            model=Predictor(fit['architecture']);model.load_state_dict(torch.load(fit['selected']['checkpoint_path'],map_location='cpu',weights_only=False)['model']);model.eval()
            with torch.no_grad():p=model(torch.from_numpy(history)).numpy().astype(np.float64)
            predictions.append((f"{fit['seed']}_{fit['architecture']}_{fit['arm']}",p))
        predictions.append(('persistence',np.repeat(history[:,-1,None],16,axis=1).astype(np.float64)))
        for r in result['linear']:predictions.append((f"linear_{r['seed']}",linear_predict(history.astype(np.float64),r['selected']['ridge'])[0]))
        for label,prediction in predictions:
            saved[name+'_'+label+'_prediction']=prediction;m,n,f=scores(prediction,truth);saved[name+'_'+label+'_mse']=m;saved[name+'_'+label+'_nrmse']=n;saved[name+'_'+label+'_failed']=f;results.append(dict(population=name,label=label,mean_mse=float(m.mean()),mean_nrmse=float(n.mean()),median_nrmse=float(np.median(n)),failed=int(f.sum())))
    path=RAW/'joint_dependence_2026865_v1.npz';save(path,**saved);record=dict(status='JOINT_DEPENDENCE_CONTROL_COMPLETE',utc=utc(),results=results,array_path=str(path),array_sha256=sha(path),frozen=frozen,mechanism_marginals_identical=True,initial_state_marginals_identical=True,model_selection_changed=False,new_method_validated=False,accepted_research_directions=0);write(O/'JOINT_DEPENDENCE86_V1.json',record);print(json.dumps({k:v for k,v in record.items() if k!='results'},indent=2),flush=True)
if __name__=='__main__':main()
