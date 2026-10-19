"""Independent SciPy mixture/likelihood arithmetic plus native checkpoint replay."""
import runtime_profile
import hashlib,json,math,pathlib,sys,time
import numpy as np
from scipy.special import logsumexp,ndtr
ROOT=pathlib.Path(__file__).resolve().parents[1];PREV=ROOT.parent/'cycle23'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}
def mix(mu,var,w,y):
    mu=mu.astype('float64');var=np.asarray(var,dtype='float64');w=np.asarray(w,dtype='float64')
    if w.ndim==1:w=np.broadcast_to(w,mu.shape[:-1])
    logpdf=-.5*(np.log(2*np.pi*var)+(y[:,None,:]-mu)**2/var).sum(-1)
    with np.errstate(divide='ignore'):nll=-logsumexp(np.log(w)+logpdf,axis=-1)
    mean=(mu*w[...,None]).sum(-2);second=((mu**2+var)*w[...,None]).sum(-2)
    pit=(ndtr((y[:,None,:]-mu)/np.sqrt(var))*w[...,None]).sum(-2)
    return dict(nll=nll,mean=mean,second=second,pit=pit,error=mean-y)
def aggregate(s):
    inside=(s['pit']>=.025)&(s['pit']<=.975)
    return dict(nll=float(s['nll'].mean()),rmse=float(np.sqrt((s['error']**2).mean())),channel_rmse=np.sqrt((s['error']**2).mean(0)),
        marginal_coverage95=float(inside.mean()),channel_coverage95=inside.mean(0),predictive_variance=float((s['second']-s['mean']**2).mean()))
def drift(p1,p2,a,b):
    variance=np.maximum(p2-p1**2,1e-15);result={}
    for key,prior,post,norm in [('first',p1,a,np.sqrt(variance)),('second',p2,b,np.maximum(abs(p2),1e-12))]:
        delta=post.mean(1)-prior;se=post.std(1,ddof=1)/np.sqrt(post.shape[1]);rel=delta/norm;corr=(delta**2-se**2)/norm**2
        vv=np.sort(abs(rel[:,0]));rng=np.random.default_rng(20262490);ids=rng.integers(0,len(vv),size=(2000,len(vv)))
        result[key]=dict(median_abs_relative=vv[(len(vv)-1)//2],mean_signed_relative=rel[:,0].mean(),mean_noise_corrected_squared_relative=corr[:,0].mean(),
            fraction_beyond3_mcse=float(np.mean(abs(delta[:,0])>3*se[:,0]+1e-8)),max_abs_relative=abs(rel[:,0]).max(),
            context_bootstrap95_noise_corrected_squared_relative=np.quantile(corr[:,0][ids].mean(1),[.025,.975]))
    return result
def alpha_post(data,n,kind,probe,u,k=8193):
    aa=np.linspace(-1,1,k);logw=np.zeros((64,k));logw[:,[0,-1]]-=np.log(2)
    A=np.array([[.8,.2],[-.2,.8]],dtype='float64')
    for t in range(n):
        ac=data['actions'][t,:,0];g=np.maximum(np.float32(0),np.float32(1)-abs(ac)) if kind==1 else np.maximum(np.float32(0),abs(ac)-np.float32(1))
        residual=data['states'][t+1,:,0].astype('float64')-(data['states'][t].astype('float64')@A.T)[:,0]
        logw-=.5*((residual[:,None]-g[:,None]*aa)/.01)**2
    g=max(0,1-abs(u)) if kind==1 else max(0,abs(u)-1)
    logw-=.5*((probe[:,0,None]-g*aa)/.01)**2
    return aa,np.exp(logw-logsumexp(logw,axis=-1,keepdims=True))
def main():
    start=time.perf_counter();out=ROOT/'outputs/integration_repaired';result=json.loads((out/'results.json').read_text())
    original=ROOT/'outputs/native_belief';assert sha(original/'results.json')==result['original_results_sha256']
    assert sha(ROOT/'PROTOCOL.md')==result['protocol_sha256'];assert sha(ROOT/'NUMERICAL_NOTES.md')==result['numerical_notes_sha256']
    assert sha(ROOT/'INTEGRATION_REPAIR.md')==result['integration_repair_protocol_sha256']
    assert len(result['cells'])==48 and len(result['checkpoints'])==6
    files=0
    for folder,key in [(original,'base_files'),(original,'raw_files'),(out,'integration_repair_files')]:
        for row in result[key]:assert sha(folder/row['path'])==row['sha256'];files+=1
    errors=[];score_errors=[];posterior_errors=[];scalar_count=0;deltas=[];max_recursion=0.
    data_manifest=json.loads((ROOT/'outputs/data/manifest.json').read_text());assert sha(ROOT/'outputs/data/manifest.json')==result['data_manifest_sha256']
    data={}
    for row in data_manifest['files']:
        f=ROOT/'outputs/data'/row['path'];assert sha(f)==row['sha256'];a=arrays(f);data[row['kind']]=a
        g=np.maximum(0,1-np.abs(a['actions'])) if row['kind']==1 else np.maximum(0,np.abs(a['actions'])-1)
        yy=a['states'][:-1].astype('float64')@np.array([[.8,-.2],[.2,.8]])+np.concatenate((g*a['alpha'][None],np.zeros_like(g)),-1)+a['noise']
        max_recursion=max(max_recursion,float(abs(yy-a['states'][1:]).max()))
    assert max_recursion<1e-6
    for cell in result['cells']:
        b=arrays(original/cell['base_file']);r=arrays(original/cell['raw_file']);nr=arrays(out/cell['raw_file']);w=b['grid_weight'];y=b['actual_query'];kind=cell['kind'];u=cell['probe_u']
        base_sc=mix(b['query_grid_mean'],b['query_variance'],w,y)
        native=mix(r['real_post_query_grid_mean'],r['real_post_query_variance'],w,y)
        logpdf=-.5*(np.log(2*np.pi*r['probe_variance'])+(r['real_probe'][:,None,:]-r['probe_grid_mean'])**2/r['probe_variance']).sum(-1)
        lw=np.log(w)[None]+logpdf;rw=np.exp(lw-logsumexp(lw,axis=-1,keepdims=True));posterior_errors.append(float(abs(rw-r['real_reweighted_weights']).max()))
        bayes=mix(b['query_grid_mean'],b['query_variance'],rw,y)
        aa,aw=alpha_post(data[kind],cell['context_size'],kind,r['real_probe'],u);posterior_errors.append(float(abs(aw-r['real_alpha_weights']).max()))
        qg=1. if kind==1 else .5;amu=np.stack((qg*aa,np.zeros_like(aa)),-1)[None].repeat(64,axis=0)
        oracle=mix(amu,np.array([.0001,.0001]),aw,y)
        scores=dict(no_update=base_sc,native=native,reweighted=bayes,physical_oracle=oracle)
        for name,sc in scores.items():
            for key,value in sc.items():score_errors.append(float(abs(value-r[f'{name}_{key}']).max()))
            for key,value in aggregate(sc).items():errors.append(float(abs(np.asarray(value)-cell['actual_query'][name][key]).max()));scalar_count+=np.asarray(value).size
        for name,a1,a2 in [('native',nr['native_post_first'],nr['native_post_second']),('reweighted',r['reweighted_post_first'],r['reweighted_post_second'])]:
            dd=drift(b['prior_first'],b['prior_second'],a1,a2)
            for moment,values in dd.items():
                for key,value in values.items():errors.append(float(abs(np.asarray(value)-cell[name+'_coherence'][moment][key]).max()));scalar_count+=np.asarray(value).size
        # Independent mixture conditioning on fixed synthetic observations,
        # checking all64contexts and the first4 probe draws per condition.
        qmu=b['query_grid_mean'].astype('float64');qv=b['query_variance']
        for j in range(4):
            py=r['synthetic_probes'][:,j].astype('float64');ll=-.5*(np.log(2*np.pi*r['probe_variance'])+(py[:,None,:]-r['probe_grid_mean'])**2/r['probe_variance']).sum(-1)
            lw=np.log(w)[None]+ll;ww=np.exp(lw-logsumexp(lw,axis=-1,keepdims=True))
            pp=(ww[...,None]*qmu).sum(1);ss=(ww[...,None]*(qmu*qmu+qv)).sum(1)
            posterior_errors.append(float(abs(pp-r['reweighted_post_first'][:,j]).max()));posterior_errors.append(float(abs(ss-r['reweighted_post_second'][:,j]).max()))
        delta=native['nll']-base_sc['nll'];rng=np.random.default_rng(20262501);ids=rng.integers(0,64,size=(2000,64))
        deltas.append(dict(kind=kind,seed=cell['checkpoint_seed'],n=cell['context_size'],probe_u=u,informative=cell['physically_informative'],native_minus_no_update_nll=float(delta.mean()),context_bootstrap95=np.quantile(delta[ids].mean(1),[.025,.975]).tolist()))
    assert max(errors)<2e-8,max(errors);assert max(score_errors)<2e-8,max(score_errors);assert max(posterior_errors)<2e-8,max(posterior_errors)
    # Native independent input assembly and forward replay on both probe types
    # at empty/full-supported contexts for all six unchanged checkpoints.
    sys.path.insert(0,str(PREV/'scripts'));from native_context_runtime import load,verify_source
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source_replay=[]
    with torch.inference_mode():
        for cr in result['checkpoints']:
            path=PREV/'outputs/native_context_weights'/cr['selected_path'];assert sha(path)==cr['checkpoint_sha256'];env,model,encoder=load(cr['config'],path)
            for cell in [c for c in result['cells'] if c['kind']==cr['kind'] and c['checkpoint_seed']==cr['seed'] and c['context_size'] in [0,9]]:
                b=arrays(original/cell['base_file']);r=arrays(original/cell['raw_file']);a=data[cr['kind']];n=cell['context_size']
                xx=np.concatenate((a['states'][:n],np.zeros((1,64,2),dtype='float32')))
                ac=np.concatenate((a['actions'][:n],np.full((1,64,1),cell['probe_u'],dtype='float32')))
                yn=np.concatenate((a['states'][1:n+1],r['real_probe'].astype('float32')[None]))
                q=encoder.forward_tensor(torch.tensor(xx,device='cuda:0'),torch.tensor(ac,device='cuda:0'),torch.tensor(yn,device='cuda:0'))
                e=max(abs(q.loc[:,0].cpu().numpy()-r['real_post_loc']).max(),abs(q.scale[:,0].cpu().numpy()-r['real_post_scale']).max())
                ids=[1000,3000,4096,5000,7000];bet=torch.tensor(b['grid_beta'][:,ids].reshape(-1,1),dtype=torch.float32,device='cuda:0')
                x=torch.zeros(len(bet),2,device='cuda:0');uc=torch.full((len(bet),1),0. if cr['kind']==1 else 1.5,device='cuda:0')
                dd=model(x,uc,bet);pred=dd.mean.cpu().numpy().reshape(64,5,2)
                e=max(e,abs(pred-b['query_grid_mean'][:,ids]).max());assert e<5e-6,e;source_replay.append(float(e))
            del model,encoder;torch.cuda.empty_cache()
    gates={}
    for key in ['native_uniform513_to1025_max_first_in_prior_sd','native_uniform513_vs8193_prior_max_first_in_prior_sd','native_uniform513_vs8193_real_post_max_first_in_prior_sd',
        'reweighted_grid_refinement_max_first_in_prior_sd','reweighted_grid_refinement_real_nll_max_error','native_grid_refinement_real_nll_max_error','physical_alpha_grid_refinement_real_nll_max_error','reweighted_7_to10_sigma_tail_real_nll_max_error','native_7_to10_sigma_tail_real_nll_max_error']:
        maximum=max(c['numerics'][key] for c in result['cells']);gates[key]=dict(maximum=maximum,threshold=.01,pass_gate=maximum<.01)
    receipt=dict(status='pass' if all(g['pass_gate'] for g in gates.values()) else 'numerical_gate_failed',computational_verification_pass=True,
        result_sha256=sha(out/'results.json'),verifier_sha256=sha(pathlib.Path(__file__)),verified_array_files=files,reported_scalars_replayed=scalar_count,
        aggregate_max_error=max(errors),raw_score_max_error=max(score_errors),independent_posterior_max_error=max(posterior_errors),independent_physical_recurrence_max_error=max_recursion,
        native_replay_cases=len(source_replay),native_replay_max_error=max(source_replay),source_revision=verify_source(),numerical_gates=gates,
        paired_context_nll_differences=deltas,elapsed_seconds=time.perf_counter()-start)
    (out/'independent_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items() if k not in ['paired_context_nll_differences','numerical_gates']}),flush=True)
if __name__=='__main__':main()
