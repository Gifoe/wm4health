"""Independent scores, importance ratios/moments and frozen-source replay."""
import runtime_profile
import json,math,time,pathlib,sys,hashlib
import numpy as np
from scipy.special import logsumexp
from verify_belief_results import arrays,mix,aggregate,sha
ROOT=pathlib.Path(__file__).resolve().parents[1];PREV=ROOT.parent/'cycle23'
def main():
    start=time.perf_counter();base=ROOT/'outputs/native_belief';original=json.loads((base/'results.json').read_text())
    filt=ROOT/'outputs/ordinary_filter_controls';fr=json.loads((filt/'results.json').read_text())
    ekf=ROOT/'outputs/ekf_density_repaired';er=json.loads((ekf/'results.json').read_text())
    particle=ROOT/'outputs/particle_moment_control';pr=json.loads((particle/'results.json').read_text())
    assert len(fr['cells'])==len(er['cells'])==48 and len(pr['cells'])==144
    file_count=0
    for folder,r in [(filt,fr),(ekf,er),(particle,pr)]:
        for f in r['files']:assert sha(folder/f['path'])==f['sha256'];file_count+=1
    errs=[];prob_err=[];score_err=[];count=0;gates=[];paired=[]
    for c in fr['cells']:
        oc=next(x for x in original['cells'] if (x['kind'],x['checkpoint_seed'],x['context_size'],x['probe_u'])==(c['kind'],c['seed'],c['n'],c['probe_u']))
        b=arrays(base/oc['base_file']);a=arrays(filt/c['raw_file']);y=b['actual_query'];w=b['grid_weight']
        ms=mix(a['map_ggn_laplace_grid_mean'],a['map_ggn_laplace_variance'],w,y)
        for key,v in ms.items():score_err.append(float(abs(v-a['map_ggn_laplace_'+key]).max()))
        for key,v in aggregate(ms).items():errs.append(float(abs(np.asarray(v)-c['actual_query']['map_ggn_laplace'][key]).max()));count+=np.asarray(v).size
        ec=next(x for x in er['cells'] if (x['kind'],x['seed'],x['n'],x['probe_u'])==(c['kind'],c['seed'],c['n'],c['probe_u']))
        ea=arrays(ekf/ec['raw_file']);assert np.array_equal(ea['ekf_loc'],a['ekf_loc']) and np.array_equal(ea['ekf_scale'],a['ekf_scale'])
        es=mix(ea['component_mean_first8'],ea['component_variance'],ea['grid_weights'],ea['actual_query'][:8])
        for key,v in es.items():score_err.append(float(abs(v-ea[key][:8]).max()))
        for key,v in aggregate({k:ea[k] for k in ['nll','mean','second','pit','error']}).items():errs.append(float(abs(np.asarray(v)-ec['actual_query'][key]).max()));count+=np.asarray(v).size
        gates.append(dict(arm='ekf_repaired',kind=c['kind'],seed=c['seed'],n=c['n'],probe_u=c['probe_u'],grid_error=ec['refined_nll_max_error'],tail_error=ec['tail35_to37_nll_max_error']))
        gates.append(dict(arm='map_ggn_laplace',kind=c['kind'],seed=c['seed'],n=c['n'],probe_u=c['probe_u'],grid_error=c['actual_query']['map_ggn_laplace']['grid_refinement_nll_max_error'],tail_error=c['actual_query']['map_ggn_laplace']['tail7_to10_nll_max_error']))
    for c in pr['cells']:
        b=arrays(base/c['base_file']);a=arrays(particle/c['raw_file']);old=arrays(base/c['original_raw_file']);beta=a['beta'];loc=b['loc'].astype('float64');scale=b['scale'].astype('float64')
        def ln(m,s):return -.5*((beta-m[:,None])/s[:,None])**2-np.log(s[:,None])-.5*np.log(2*np.pi)
        lp=ln(loc,scale);lq=np.logaddexp(lp,ln(a['proposal_loc'],a['proposal_scale']))-np.log(2)
        ll=-.5*(np.log(2*np.pi*a['probe_variance'])+(old['real_probe'][:,None,:]-a['probe_component_mean'])**2/a['probe_variance']).sum(-1)
        logs=lp+ll-lq
        for k,v in [('log_prior',lp),('log_mixture_proposal',lq),('unnormalized_logweights',logs)]:prob_err.append(float(abs(v-a[k]).max()))
        for size in [256,1024,4096]:
            ids=np.r_[np.arange(size//2),np.arange(2048,2048+size//2)];l=logs[:,ids];ww=np.exp(l-logsumexp(l,axis=-1,keepdims=True))
            bm=(ww*beta[:,ids]).sum(-1);bv=(ww*(beta[:,ids]-bm[:,None])**2).sum(-1)
            prob_err.extend([float(abs(ww-a[f'c{size}_weights']).max()),float(abs(bm-a[f'c{size}_latent_loc']).max()),float(abs(bv-a[f'c{size}_latent_variance']).max())])
            sc=mix(a[f'c{size}_component_mean_first8'],a[f'c{size}_component_noise'],b['grid_weight'],b['actual_query'][:8])
            for key,v in sc.items():score_err.append(float(abs(v-a[f'c{size}_{key}'][:8]).max()))
            full={key:a[f'c{size}_{key}'] for key in ['nll','mean','second','pit','error']}
            for key,v in aggregate(full).items():errs.append(float(abs(np.asarray(v)-c['actual_query'][str(size)][key]).max()));count+=np.asarray(v).size
            ess=1/(ww*ww).sum(-1);errs.extend([abs(ess.mean()-c['actual_query'][str(size)]['ess_mean']),abs(ess.min()-c['actual_query'][str(size)]['ess_min'])])
            gates.append(dict(arm=f'particle{size}',kind=c['kind'],seed=c['seed'],n=c['n'],probe_u=c['probe_u'],rep=c['importance_rep'],grid_error=c['actual_query'][str(size)]['grid_refinement_nll_max_error'],tail_error=c['actual_query'][str(size)]['tail7_to10_nll_max_error']))
            if c['n']==0 and not c['informative']:
                paired.append(dict(kind=c['kind'],seed=c['seed'],particle_count=size,rep=c['importance_rep'],mean_nll_delta=float((full['nll']-old['no_update_nll']).mean())))
    assert max(errs)<2e-7,max(errs);assert max(score_err)<2e-7,max(score_err);assert max(prob_err)<2e-7,max(prob_err)
    # Validate original-source Jacobian update and replay particle query mapping.
    sys.path.insert(0,str(PREV/'scripts'));from native_context_runtime import load,verify_source
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    replay=[];jac_errors=[]
    for cr in original['checkpoints']:
        f=PREV/'outputs/native_context_weights'/cr['selected_path'];assert sha(f)==cr['checkpoint_sha256'];env,model,encoder=load(cr['config'],f)
        for p in model.parameters():p.requires_grad_(False)
        for c in [x for x in fr['cells'] if (x['kind'],x['seed'])==(cr['kind'],cr['seed']) and x['n'] in [0,9]]:
            oc=next(x for x in original['cells'] if (x['kind'],x['checkpoint_seed'],x['context_size'],x['probe_u'])==(c['kind'],c['seed'],c['n'],c['probe_u']))
            b=arrays(base/oc['base_file']);r=arrays(base/oc['raw_file']);a=arrays(filt/c['raw_file'])
            xx=torch.zeros(64,2,device='cuda:0');uu=torch.full((64,1),c['probe_u'],device='cuda:0');bb=torch.tensor(b['loc'],dtype=torch.float32,device='cuda:0',requires_grad=True)
            dist=model(xx,uu,bb[:,None]);mu=dist.mean;J=[]
            for dim in range(2):J.append(torch.autograd.grad(mu[:,dim].sum(),bb,retain_graph=dim==0)[0].detach().cpu().numpy())
            J=np.stack(J,-1).astype('float64');R=dist.variance[0].detach().cpu().numpy().astype('float64');h=mu.detach().cpu().numpy().astype('float64')
            vv=1/(1/b['scale'].astype('float64')**2+(J*J/R).sum(-1));mm=b['loc']+vv*(J*(r['real_probe']-h)/R).sum(-1)
            jac_errors.extend([float(abs(mm-a['ekf_loc']).max()),float(abs(np.sqrt(vv)-a['ekf_scale']).max())])
            pc=next(x for x in pr['cells'] if (x['kind'],x['seed'],x['n'],x['probe_u'],x['importance_rep'])==(c['kind'],c['seed'],c['n'],c['probe_u'],0));pa=arrays(particle/pc['raw_file'])
            with torch.inference_mode():
                z=np.linspace(-7,7,8193);beta=pa['c1024_latent_loc'][:8,None]+np.sqrt(pa['c1024_latent_variance'][:8,None])*z
                vals=[]
                for k in range(0,beta.size,16384):
                    xb=torch.zeros(min(16384,beta.size-k),2,device='cuda:0');ub=torch.full((len(xb),1),0. if cr['kind']==1 else 1.5,device='cuda:0');bt=torch.tensor(beta.reshape(-1,1)[k:k+16384],dtype=torch.float32,device='cuda:0');vals.append(model(xb,ub,bt).mean.cpu().numpy())
                pred=np.concatenate(vals).reshape(8,8193,2);replay.append(float(abs(pred-pa['c1024_component_mean_first8']).max()))
        assert sha(f)==cr['checkpoint_sha256'];del model,encoder;torch.cuda.empty_cache()
    assert max(jac_errors)<5e-6,max(jac_errors);assert max(replay)<5e-6,max(replay)
    receipt=dict(status='pass' if all(x['grid_error']<.01 and x['tail_error']<.01 for x in gates) else 'numerical_gate_failed',computational_verification_pass=True,
        verifier_sha256=sha(pathlib.Path(__file__)),filter_results_sha256=sha(filt/'results.json'),repaired_ekf_results_sha256=sha(ekf/'results.json'),particle_results_sha256=sha(particle/'results.json'),verified_files=file_count,
        reported_scalars_replayed=count,aggregate_max_error=max(errs),score_max_error=max(score_err),importance_algebra_max_error=max(prob_err),ekf_jacobian_update_replay_max_error=max(jac_errors),native_particle_query_replay_max_error=max(replay),native_replay_cases=len(replay),
        numerical_gate_rows=gates,particle_cold_noinfo_nll_deltas=paired,elapsed_seconds=time.perf_counter()-start,source_revision=verify_source(),
        scope='All raw files hash-checked; full MAP scores and all aggregates; first8 dense component scores per repaired EKF and particle row; independent importance weights/moments all64;24 native Jacobian/query cases. Post-hoc controls, no fresh scientific confirmation.')
    (ROOT/'outputs/ordinary_controls_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items() if k not in ['numerical_gate_rows','particle_cold_noinfo_nll_deltas']}),flush=True)
if __name__=='__main__':main()
