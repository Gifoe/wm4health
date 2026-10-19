"""Native model replay and independent mixture-weight/SLR algebra."""
import runtime_profile
import hashlib,json,pathlib,sys,math,time
import numpy as np
from scipy.special import logsumexp
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];PREV=ROOT.parent/'cycle23';sys.path.insert(0,str(PREV/'scripts'))
from native_context_runtime import load,verify_source
OUT=ROOT/'outputs/belief_audit';r=json.loads((OUT/'results.json').read_text());gate=json.loads((ROOT/'outputs/interface/results.json').read_text());data=np.load(ROOT/'outputs/data/pendulum.npz')
N=64;M=1024;device='cuda:0';start=time.perf_counter();mx=dict(encoder=0.,native_prediction=0.,proposal=0.,weights=0.,ess=0.,moments=0.,slr=0.);counts=dict(encoder=0,native_prediction=0,proposal=0,slr=0)
def tensor(a):return torch.as_tensor(a,device=device,dtype=torch.float32)
def normal(b,m,s):return (-.5*((b-m[:,None])/s[:,None])**2-np.log(s[:,None])-.5*np.log(2*np.pi)).sum(-1)
def raw(model,beta,x,u):
    shp=beta.shape[:-1];flat=torch.as_tensor(beta,device=device,dtype=torch.float32).reshape(-1,16);xs=tensor(x).expand(*shp,3).reshape(-1,3);us=torch.full((len(flat),1),u,device=device);a=[];b=[]
    for k in range(0,len(flat),8192):
        dist=model(xs[k:k+8192].contiguous(),us[k:k+8192],flat[k:k+8192].contiguous());a.append(dist.mean.cpu().numpy());b.append(dist.variance.cpu().numpy())
    return np.concatenate(a).reshape(*shp,3).astype('float64'),np.concatenate(b).reshape(*shp,3).astype('float64')
with torch.inference_mode():
 for rec in gate['models']:
    cfg=rec['config'];name=('positive' if cfg['context_encoder_kwargs']['positive_weights'] else 'unconstrained')+f"_s{cfg['seed']}";env,model,encoder=load(cfg,PREV/'outputs/native_context_weights'/rec['weight_path']);states=tensor(data['states']);actions=tensor(data['actions'])
    gen=torch.Generator(device=device).manual_seed(20262591);z=torch.randn(N,M,16,device=device,dtype=torch.float64,generator=gen).cpu().numpy()
    torch.randn(N,M,16,device=device,dtype=torch.float64,generator=gen)
    for _ in range(2):torch.randn(N,M,4,device=device,dtype=torch.float64,generator=gen)
    iz=torch.randn(N,4096,16,device=device,dtype=torch.float64,generator=gen).cpu().numpy()
    for n in [0,8,32]:
        prior=np.load(OUT/f'{name}_n{n}_p-1_r20262591_no_update.npz');loc=prior['latent_loc'];scale=prior['latent_scale']
        q=encoder.empty_set_context() if not n else encoder.forward_tensor(states[:n],actions[:n],states[1:n+1]);qm=np.broadcast_to(q.loc.cpu().numpy(),loc.shape);qs=np.broadcast_to(q.scale.cpu().numpy(),scale.shape)
        mx['encoder']=max(mx['encoder'],float(abs(qm-loc).max()),float(abs(qs-scale).max()));counts['encoder']+=1
        for p,u in enumerate([0.,1.5]):
            posterior=np.load(OUT/f'{name}_n{n}_p{p}_r20262591_native.npz');pl=posterior['latent_loc'];ps=posterior['latent_scale'];px=tensor(data['probe_state']).expand(1,N,3);pa=torch.full((1,N,1),u,device=device);py=tensor(data['probe'][p])[None]
            if n:q=encoder.forward_tensor(torch.cat((states[:n],px)),torch.cat((actions[:n],pa)),torch.cat((states[1:n+1],py)))
            else:q=encoder.forward_tensor(px,pa,py)
            mx['encoder']=max(mx['encoder'],float(abs(q.loc.cpu().numpy()-pl).max()),float(abs(q.scale.cpu().numpy()-ps).max()));counts['encoder']+=1
            # Distinct direct decoder batches at all four query initial states.
            for qidx,theta in enumerate(data['query_theta']):
                b=(pl[:,None]+ps[:,None]*z)[:,:8];mu,v=raw(model,b,[math.cos(theta),math.sin(theta),0.],1.5)
                mx['native_prediction']=max(mx['native_prediction'],float(abs(mu[...,2]-posterior['mu'][:,:8,qidx]).max()),float(abs(v[...,2]-posterior['variance_components'][:,:8,qidx]).max()));counts['native_prediction']+=1
            # Statistical linear regression checked using independent NumPy solves.
            beta=loc[:,None]+scale[:,None]*iz;mu,v=raw(model,beta,data['probe_state'],u);db=beta-beta.mean(1,keepdims=True);dm=mu-mu.mean(1,keepdims=True)
            cb=np.einsum('nmi,nmj->nij',db,db)/4096;cross=np.einsum('nmi,nmj->nij',db,dm)/4096;H=np.linalg.solve(cb,cross).transpose(0,2,1);res=dm-np.einsum('nsi,nmi->nms',H,db)
            omega=np.einsum('nmi,nmj->nij',res,res)/4096
            for j in range(3):omega[:,j,j]+=v[...,j].mean(1)
            P=np.zeros_like(cb)
            for j in range(16):P[:,j,j]=scale[:,j]**2
            PH=P@H.transpose(0,2,1);S=H@PH+omega;K=np.linalg.solve(S,PH.transpose(0,2,1)).transpose(0,2,1)
            sl=loc+np.einsum('nbs,ns->nb',K,data['probe'][p].astype('float32').astype('float64')-mu.mean(1));sc=P-K@H@P;sc=(sc+sc.transpose(0,2,1))/2
            sr=np.load(OUT/f'{name}_n{n}_p{p}_r20262591_slr.npz');mx['slr']=max(mx['slr'],float(abs(sl-sr['latent_loc']).max()),float(abs(sc-sr['latent_covariance']).max()));counts['slr']+=1
            for budget in [1024,4096]:
                half=budget//2;b=np.concatenate((loc[:,None]+scale[:,None]*iz[:,:half],pl[:,None]+ps[:,None]*iz[:,half:budget]),1);prop=np.load(OUT/f'{name}_n{n}_p{p}_r20262591_proposal{budget}.npz')
                mx['proposal']=max(mx['proposal'],float(abs(b.astype('float32')-prop['beta']).max()));counts['proposal']+=1
                pm,pv=raw(model,b,data['probe_state'],u);lp=normal(b,loc,scale);lproposal=np.logaddexp(lp,normal(b,pl,ps))-math.log(2);ll=-.5*(np.log(2*np.pi*pv)+(data['probe'][p].astype('float32')[:,None]-pm)**2/pv).sum(-1);lw=lp+ll-lproposal;w=np.exp(lw-logsumexp(lw,axis=1,keepdims=True));ess=1/(w*w).sum(1)
                wm=(w[...,None]*b).sum(1);dd=b-wm[:,None];wc=np.einsum('nm,nmi,nmj->nij',w,dd,dd);wr=np.load(OUT/f'{name}_n{n}_p{p}_r20262591_is{budget}_particles.npz')
                mx['weights']=max(mx['weights'],float(abs(w-prop['weights']).max()));mx['ess']=max(mx['ess'],float(abs(ess-prop['ess']).max()));mx['moments']=max(mx['moments'],float(abs(wm-wr['latent_loc']).max()),float(abs(wc-wr['latent_covariance']).max()))
    del env,model,encoder;torch.cuda.empty_cache()
assert mx['encoder']==mx['proposal']==0,mx
assert mx['native_prediction']<3e-6 and mx['weights']<1e-6 and mx['ess']<1e-3 and mx['moments']<1e-5 and mx['slr']<1e-5,mx
result=dict(status='pass',source_revision=verify_source(),max_errors=mx,checks=counts,scope='All six checkpoints and all three context lengths, both probes, first MC seed; all prediction scores independently checked in separate receipt.',verifier_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),raw_results_sha256=hashlib.sha256((OUT/'results.json').read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-start)
(OUT/'independent_inference_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
