import runtime_profile
import json,pathlib,time,math,sys
from pendulum_core import *
OUT=ROOT/'outputs/belief_audit';assert not OUT.exists();OUT.mkdir(parents=True)
gate=json.loads((ROOT/'outputs/interface/results.json').read_text());assert gate['status']=='pass'
manifest=json.loads((ROOT/'outputs/data/manifest.json').read_text());assert sha(ROOT/'outputs/data/pendulum.npz')==manifest['sha256']
data=np.load(ROOT/'outputs/data/pendulum.npz');states=tensor(data['states']);actions=tensor(data['actions']);px=tensor(data['probe_state']);truth=tensor(data['query'][...,2]).double();N=len(truth)
records=gate['models'];rows=[];began=time.perf_counter();torch.cuda.reset_peak_memory_stats()
def save(name,method,n,p,rep,mu,v,mu2,v2,noise,noise2,extra=None):
    s=scores(mu,v,mu2,v2,truth,noise,noise2);arrays={k:t.detach().cpu().numpy() for k,t in s.items()};arrays.update(mu=mu.float().cpu().numpy(),variance_components=v.float().cpu().numpy(),mu2=mu2.float().cpu().numpy(),variance_components2=v2.float().cpu().numpy())
    if extra:arrays.update({k:t.detach().cpu().numpy() for k,t in extra.items()})
    f=OUT/f'{name}_n{n}_p{p}_r{rep}_{method}.npz';assert not f.exists();np.savez_compressed(f,**arrays)
    row=dict(model=name,method=method,n=n,probe_index=p,mc_seed=rep,path=f.name,sha256=sha(f),**summary(s));rows.append(row)
    (OUT/'progress.json').write_text(json.dumps(dict(rows=len(rows),last=row,elapsed_seconds=time.perf_counter()-began),indent=2))
    return row
with torch.inference_mode():
    for r in records:
        cfg=r['config'];name=('positive' if cfg['context_encoder_kwargs']['positive_weights'] else 'unconstrained')+f"_s{cfg['seed']}"
        checkpoint=PREV/'outputs/native_context_weights'/r['weight_path'];assert sha(checkpoint)==r['weight_sha256'];env,model,encoder=load(cfg,checkpoint)
        for n in [0,8,32]:
            loc,scale=encode(encoder,states,actions,n)
            for rep in [20262591,20262592,20262593]:
                gen=torch.Generator(device=DEVICE).manual_seed(rep)
                z=torch.randn(N,M,16,device=DEVICE,dtype=torch.float64,generator=gen);z2=torch.randn(N,M,16,device=DEVICE,dtype=torch.float64,generator=gen)
                noise=torch.randn(N,M,4,device=DEVICE,dtype=torch.float64,generator=gen);noise2=torch.randn(N,M,4,device=DEVICE,dtype=torch.float64,generator=gen)
                iz=torch.randn(N,4096,16,device=DEVICE,dtype=torch.float64,generator=gen)
                unif=torch.rand(N,M,device=DEVICE,dtype=torch.float64,generator=gen);unif2=torch.rand(N,M,device=DEVICE,dtype=torch.float64,generator=gen)
                mu,v=query_decode(model,loc[:,None]+scale[:,None]*z,data['query_theta']);mu2,v2=query_decode(model,loc[:,None]+scale[:,None]*z2,data['query_theta'])
                save(name,'no_update',n,-1,rep,mu,v,mu2,v2,noise,noise2,dict(latent_loc=loc,latent_scale=scale))
                for p,u in enumerate([0.,1.5]):
                    py=tensor(data['probe'][p]);pl,ps=append(encoder,states,actions,n,px,u,py)
                    mu,v=query_decode(model,pl[:,None]+ps[:,None]*z,data['query_theta']);mu2,v2=query_decode(model,pl[:,None]+ps[:,None]*z2,data['query_theta'])
                    save(name,'native',n,p,rep,mu,v,mu2,v2,noise,noise2,dict(latent_loc=pl,latent_scale=ps))
                    sl,sc=slr(model,loc,scale,px,u,py,iz)
                    mu,v=query_decode(model,gaussian_samples(sl,sc,z),data['query_theta']);mu2,v2=query_decode(model,gaussian_samples(sl,sc,z2),data['query_theta'])
                    save(name,'slr',n,p,rep,mu,v,mu2,v2,noise,noise2,dict(latent_loc=sl,latent_covariance=sc))
                    for count in [1024,4096]:
                        beta,w,il,ic,ess=importance(model,loc,scale,pl,ps,px,u,py,count,iz)
                        for kind in ['particles','diagonal','full']:
                            if kind=='particles':
                                cdf=w.cumsum(1);cdf[:,-1]=1;ids=torch.searchsorted(cdf.contiguous(),unif.contiguous());ids2=torch.searchsorted(cdf.contiguous(),unif2.contiguous());ba=beta.gather(1,ids[...,None].expand(-1,-1,16));bb=beta.gather(1,ids2[...,None].expand(-1,-1,16))
                            else:
                                cv=ic if kind=='full' else torch.diagonal(ic,dim1=-2,dim2=-1).clamp_min(0);ba=gaussian_samples(il,cv,z);bb=gaussian_samples(il,cv,z2)
                            mu,v=query_decode(model,ba,data['query_theta']);mu2,v2=query_decode(model,bb,data['query_theta'])
                            save(name,f'is{count}_{kind}',n,p,rep,mu,v,mu2,v2,noise,noise2,dict(latent_loc=il,latent_covariance=ic,ess=ess,min_covariance_eigenvalue=torch.linalg.eigvalsh(ic).min(-1).values))
                        # Keep proposals/weights once, not six redundant copies.
                        f=OUT/f'{name}_n{n}_p{p}_r{rep}_proposal{count}.npz';np.savez_compressed(f,beta=beta.float().cpu().numpy(),weights=w.cpu().numpy(),ess=ess.cpu().numpy())
        assert sha(checkpoint)==r['weight_sha256'];del env,model,encoder;torch.cuda.empty_cache();print(json.dumps(dict(completed_model=name,rows=len(rows),elapsed_seconds=time.perf_counter()-began)),flush=True)
result=dict(status='completed',source_revision=verify_source(),protocol_sha256=sha(ROOT/'PROTOCOL.md'),execution_details_sha256=sha(ROOT/'EXECUTION_DETAILS.md'),data_sha256=manifest['sha256'],script_sha256=sha(pathlib.Path(__file__)),core_sha256=sha(ROOT/'scripts/pendulum_core.py'),systems=N,mc_samples=M,mc_seeds=[20262591,20262592,20262593],new_training=False,elapsed_seconds=time.perf_counter()-began,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),rows=rows)
(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
