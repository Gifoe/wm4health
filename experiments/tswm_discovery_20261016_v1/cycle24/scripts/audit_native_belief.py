"""Frozen six-checkpoint physical belief-update audit on held-out systems."""
import runtime_profile
import json,math,pathlib,time,sys,hashlib
import numpy as np
import torch
from belief_core import *
OUT=ROOT/'outputs/native_belief'
N=64;M=256
def numpy(x):return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else x
def save(path,**arrays):np.savez_compressed(path,**{k:numpy(v) for k,v in arrays.items()})
def bayes_moments(qmu,qvar,pmu,pvar,w,probes,chunk=16):
    first=[];second=[]
    for i in range(0,probes.shape[1],chunk):
        y=probes[:,i:i+chunk].double()
        lp=likelihood(pmu[:,None],pvar,y)+torch.log(w)[None,None,:]
        post=torch.softmax(lp,-1)
        a=torch.einsum('nmk,nkd->nmd',post,qmu)
        b=torch.einsum('nmk,nkd->nmd',post,qmu*qmu+qvar)
        first.append(a);second.append(b)
    return torch.cat(first,1),torch.cat(second,1)
def max_relative(a,b,denom):return float(((a-b).abs()/denom).max())
def main():
    started=time.perf_counter();assert not OUT.exists();OUT.mkdir(parents=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats()
    data_manifest=json.loads((ROOT/'outputs/data/manifest.json').read_text())
    assert data_manifest['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    wm=json.loads((PREV/'outputs/native_context_weights/manifest.json').read_text())
    selected=[r for r in wm['models'] if r['config']['env_name'] in ['action_squash_1','action_squash_2']]
    selected.sort(key=lambda r:(r['config']['env_name'],r['config']['seed']));assert len(selected)==6
    z,w=normal_grid();ghz,ghw=hermite(128);cells=[];base_files=[];raw_files=[];load_receipts=[]
    for record in selected:
        cfg=record['config'];kind=int(cfg['env_name'][-1]);seed=cfg['seed'];query_u=0. if kind==1 else 1.5
        checkpoint=PREV/'outputs/native_context_weights'/record['path'];assert sha(checkpoint)==record['sha256']
        assert cfg['context_dim']==1 and cfg['max_ctx_card']==10 and not cfg['transition_model_kwargs']['hidden_dependent_noise']
        env,model,encoder=load(cfg,checkpoint);load_receipts.append(dict(kind=kind,seed=seed,checkpoint_sha256=sha(checkpoint),selected_path=record['path'],config=cfg))
        row=next(r for r in data_manifest['files'] if r['kind']==kind);data_path=ROOT/'outputs/data'/row['path'];assert sha(data_path)==row['sha256']
        with np.load(data_path) as f:data={k:f[k] for k in f.files}
        states=tensor(data['states']);actions=tensor(data['actions']);actual_query=tensor(data['query']).double()
        for n in [0,2,5,9]:
            loc,scale=encode(encoder,states,actions,n);loc=loc[:,0];scale=scale[:,0]
            qmu,qvar,beta=normal_components(model,loc,scale,query_u,z)
            prior_first,prior_second=moments(qmu,qvar,w)
            prior_score=score(qmu,qvar,w,actual_query)
            gh1,gh2=physical_moments(model,loc,scale,query_u,128)
            gh64a,gh64b=physical_moments(model,loc,scale,query_u,64)
            prior_sd=torch.sqrt((prior_second-prior_first**2).clamp_min(1e-15))
            alpha,aw=alpha_weights(states,actions,n,kind);oracle_prior=alpha_score(alpha,aw,query_u,kind,actual_query)
            base=OUT/f'base_k{kind}_s{seed}_n{n}.npz'
            save(base,loc=loc,scale=scale,grid_z=z,grid_weight=w,grid_beta=beta,query_grid_mean=qmu.float(),query_variance=qvar,
                prior_first=prior_first,prior_second=prior_second,gh128_first=gh1,gh128_second=gh2,gh64_first=gh64a,gh64_second=gh64b,
                actual_query=actual_query,alpha_grid=alpha,alpha_weight=aw)
            base_files.append(dict(path=base.name,sha256=sha(base)))
            for probe_index,probe_u in enumerate([0.,1.5]):
                cell_started=time.perf_counter()
                pmu,pvar,_=normal_components(model,loc,scale,probe_u,z)
                gen=torch.Generator(device=DEVICE).manual_seed(202624200+kind*10000+seed*1000+n*10+probe_index)
                latent=loc[:,None]+scale[:,None]*torch.randn(N,M,device=DEVICE,generator=gen)
                generated_mu,generated_var=decode(model,latent,probe_u)
                probes=generated_mu+torch.sqrt(generated_var.float())*torch.randn(N,M,2,device=DEVICE,generator=gen)
                ploc,pscale=append_evidence(encoder,states,actions,n,probes,probe_u)
                native_first,native_second=physical_moments(model,ploc,pscale,query_u,128)
                bayes_first,bayes_second=bayes_moments(qmu,qvar,pmu,pvar,w,probes)
                real_probe=tensor(data['probe'][probe_index]).double()
                rloc,rscale=append_evidence(encoder,states,actions,n,real_probe.float(),probe_u);rloc=rloc[:,0];rscale=rscale[:,0]
                rqmu,rqvar,_=normal_components(model,rloc,rscale,query_u,z)
                native_score=score(rqmu,rqvar,w,actual_query)
                rw=posterior_weights(torch.log(w)[None],pmu,pvar,real_probe);bayes_score=score(qmu,qvar,rw,actual_query)
                aa,aaw=alpha_weights(states,actions,n,kind,probe=real_probe,probe_u=probe_u)
                oracle_score=alpha_score(aa,aaw,query_u,kind,actual_query)
                numerics={
                    'prior_gh128_vs_dense_max_first_in_prior_sd':max_relative(gh1,prior_first,prior_sd),
                    'prior_gh64_vs128_max_first_in_prior_sd':max_relative(gh64a,gh1,prior_sd),
                    'prior_gh128_vs_dense_max_second':float((gh2-prior_second).abs().max())}
                # Native broadcasting is a new execution path; compare original
                # forward_tensor on the exact same selected evidence.
                errors=[]
                for j in range(2):
                    xx=torch.cat((states[:n],torch.zeros(1,N,2,device=DEVICE)),0)
                    uu=torch.cat((actions[:n],torch.full((1,N,1),probe_u,device=DEVICE)),0)
                    yy=torch.cat((states[1:n+1],probes[:,j][None]),0)
                    direct=encoder.forward_tensor(xx,uu,yy)
                    errors.append(max(float((direct.loc[:,0]-ploc[:,j]).abs().max()),float((direct.scale[:,0]-pscale[:,j]).abs().max())))
                assert max(errors)<5e-6,errors;numerics['append_broadcast_vs_native_tensor_max_error']=max(errors)
                # Predetermined fixed subset and resolution refinement. It is
                # numerical verification, never selection of a scientific arm.
                f64,s64=physical_moments(model,ploc[:8],pscale[:8],query_u,64)
                numerics['synthetic_post_gh64_vs128_max_first_in_prior_sd']=max_relative(f64,native_first[:8],prior_sd[:8,None,:])
                numerics['synthetic_post_gh64_vs128_max_second']=float((s64-native_second[:8]).abs().max())
                zr,wr=normal_grid(16385);qmr,qvr,_=normal_components(model,loc[:8],scale[:8],query_u,zr)
                pmr,pvr,_=normal_components(model,loc[:8],scale[:8],probe_u,zr)
                b1r,b2r=bayes_moments(qmr,qvr,pmr,pvr,wr,probes[:8])
                numerics['reweighted_grid_refinement_max_first_in_prior_sd']=max_relative(b1r,bayes_first[:8],prior_sd[:8,None,:])
                numerics['reweighted_grid_refinement_max_second']=float((b2r-bayes_second[:8]).abs().max())
                rwref=posterior_weights(torch.log(wr)[None],pmr,pvr,real_probe[:8]);brscore=score(qmr,qvr,rwref,actual_query[:8])
                numerics['reweighted_grid_refinement_real_nll_max_error']=float((brscore['nll']-bayes_score['nll'][:8]).abs().max())
                qmreal,qvreal,_=normal_components(model,rloc[:8],rscale[:8],query_u,zr);rnscore=score(qmreal,qvreal,wr,actual_query[:8])
                numerics['native_grid_refinement_real_nll_max_error']=float((rnscore['nll']-native_score['nll'][:8]).abs().max())
                ar,arw=alpha_weights(states[:,:8],actions[:,:8],n,kind,k=16385,probe=real_probe[:8],probe_u=probe_u)
                arscore=alpha_score(ar,arw,query_u,kind,actual_query[:8])
                numerics['physical_alpha_grid_refinement_real_nll_max_error']=float((arscore['nll']-oracle_score['nll'][:8]).abs().max())
                # Tail-range sensitivity checks were added before data generation
                # in NUMERICAL_NOTES.md; retain both results if the gate fails.
                zt,wt=normal_grid(16385,10.)
                tq,tv,_=normal_components(model,loc[:8],scale[:8],query_u,zt);tp,tpv,_=normal_components(model,loc[:8],scale[:8],probe_u,zt)
                tw=posterior_weights(torch.log(wt)[None],tp,tpv,real_probe[:8]);tscore=score(tq,tv,tw,actual_query[:8])
                tn,tvn,_=normal_components(model,rloc[:8],rscale[:8],query_u,zt);tnscore=score(tn,tvn,wt,actual_query[:8])
                numerics['reweighted_7_to10_sigma_tail_real_nll_max_error']=float((tscore['nll']-brscore['nll']).abs().max())
                numerics['native_7_to10_sigma_tail_real_nll_max_error']=float((tnscore['nll']-rnscore['nll']).abs().max())
                raw=OUT/f'cell_k{kind}_s{seed}_n{n}_p{probe_index}.npz'
                arrays=dict(synthetic_latent=latent,synthetic_probe_mean=generated_mu,synthetic_probe_variance=generated_var,synthetic_probes=probes,
                    post_loc=ploc,post_scale=pscale,native_post_first=native_first,native_post_second=native_second,
                    reweighted_post_first=bayes_first,reweighted_post_second=bayes_second,probe_grid_mean=pmu.float(),probe_variance=pvar,
                    real_probe=real_probe,real_post_loc=rloc,real_post_scale=rscale,real_post_query_grid_mean=rqmu.float(),real_post_query_variance=rqvar,
                    real_reweighted_weights=rw,real_alpha_weights=aaw,
                    refine_native_first64=f64,refine_native_second64=s64,refine_reweighted_first=b1r,refine_reweighted_second=b2r,
                    refined_reweighted_nll=brscore['nll'],refined_native_nll=rnscore['nll'],refined_alpha_nll=arscore['nll'],tail10_reweighted_nll=tscore['nll'],tail10_native_nll=tnscore['nll'])
                scores={'no_update':prior_score,'native':native_score,'reweighted':bayes_score,'physical_oracle':oracle_score}
                for name,sc in scores.items():
                    for key,value in sc.items():arrays[f'{name}_{key}']=value
                save(raw,**arrays);raw_files.append(dict(path=raw.name,sha256=sha(raw)))
                cell=dict(kind=kind,checkpoint_seed=seed,context_size=n,probe_u=probe_u,
                    physically_informative=(probe_u==0 if kind==1 else probe_u==1.5),
                    native_coherence=summarize_drift(prior_first,prior_second,native_first,native_second),
                    reweighted_coherence=summarize_drift(prior_first,prior_second,bayes_first,bayes_second),
                    actual_query={name:summary_scores(sc) for name,sc in scores.items()},
                    physical_prior_score=summary_scores(oracle_prior),numerics=numerics,elapsed_seconds=time.perf_counter()-cell_started,
                    raw_file=raw.name,base_file=base.name)
                cells.append(cell)
                with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(cell)+'\n')
                print(json.dumps(dict(completed_cells=len(cells),expected_cells=48,kind=kind,seed=seed,n=n,probe_u=probe_u,elapsed_seconds=time.perf_counter()-started)),flush=True)
        assert sha(checkpoint)==record['sha256'];del model,encoder;torch.cuda.empty_cache()
    result=dict(protocol_sha256=sha(ROOT/'PROTOCOL.md'),numerical_notes_sha256=sha(ROOT/'NUMERICAL_NOTES.md'),script_sha256=sha(pathlib.Path(__file__)),core_sha256=sha(ROOT/'scripts/belief_core.py'),
        source_revision=verify_source(),data_manifest_sha256=sha(ROOT/'outputs/data/manifest.json'),checkpoints=load_receipts,
        cells=cells,base_files=base_files,raw_files=raw_files,systems_per_variant=N,synthetic_probes_per_context=M,
        trained_models=0,native_checkpoint_count=6,elapsed_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        accepted_novel_directions=0,objective_achieved=False,limitations=['New capability audit, not paper leaderboard reproduction.','Two actuation variants of one toy plant; no multi-real-system generalization.','Same64 physical systems reused across three checkpoint seeds.','Ordinary Bayes reweighting and known physical posterior are existing controls, not proposed methods.','Integration diagnostics must be assessed before any scientific conclusion.'],
        environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,device=torch.cuda.get_device_name(0),tf32=False))
    (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['cells','base_files','raw_files','checkpoints']}),flush=True)
if __name__=='__main__':
    with torch.inference_mode():main()
