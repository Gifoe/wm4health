"""Generic feasible adversary: an INNER range, never an outer certificate.

If feasible fitted-kernel/coupling witnesses already force a wide range, no
sound algorithm on that SAME uncertainty set can produce a narrower interval.
This tests whether the proposed algorithmic gap is actually set uncertainty.
"""
import hashlib,json,pathlib,time
import numpy as np
from scipy.stats import chi2
import torch
from confidence_tube import mean,simulate,family,regression_fit,tubes,shared_parameter_tube
from run_confidence_screen import TRUE,SIGMA
ROOT=pathlib.Path(__file__).resolve().parents[1]
torch.set_num_threads(4)

def torch_family(w,f):
    # batch shape [2 sides, queries, restarts]; f [queries,time,2].
    A0=w[...,:2,:].transpose(-1,-2);A1=(w[...,:2,:]+w[...,3:5,:]).transpose(-1,-2)
    b0=w[...,-1,:];b1=w[...,2,:]+b0
    d=f[None,:,None,0,:].expand(w.shape[:-2]+(2,));c=torch.zeros_like(d);s=torch.zeros_like(d)
    for t in range(f.shape[1]-1):
        xf=f[None,:,None,t,:]
        residual=f[None,:,None,t+1,:]-(A0@xf[...,None]).squeeze(-1)-b0
        d=(A1@d[...,None]).squeeze(-1)+b1
        c=(A1@c[...,None]).squeeze(-1)+residual
        jr=torch.stack([-residual[...,1],residual[...,0]],dim=-1)
        s=(A1@s[...,None]).squeeze(-1)+jr
    center=d[...,0]-f[None,:,None,-1,0];rad=torch.sqrt(c[...,0]**2+s[...,0]**2+1e-30)
    return center-rad,center+rad

def main():
    pp=ROOT/'outputs/feasible_extrema_plan.json';plan=json.loads(pp.read_text())
    start=time.perf_counter();records=[];witness_dir=ROOT/'outputs/feasible_witnesses';witness_dir.mkdir(exist_ok=True)
    for rep in range(plan['independent_fits']):
        rg=np.random.default_rng(plan['training_seed_base']+rep);n=plan['training_n'];H=plan['horizon'];Q=plan['queries_per_fit']
        x=rg.normal(size=(n,2));a=rg.uniform(-1,1,(n,1));y=mean(TRUE,x,a)+rg.normal(scale=SIGMA,size=(n,2))
        w,P=regression_fit(x,a,y);lam,U=np.linalg.eigh(P);S=(U*np.sqrt(lam))@U.T
        radius=SIGMA*np.sqrt(chi2.ppf(1-plan['alpha'],w.size))
        qg=np.random.default_rng(plan['query_seed_base']+rep)
        f=simulate(TRUE,qg.normal(scale=.4,size=(Q,2)),qg.normal(scale=SIGMA,size=(Q,H,2)),0.)
        basic=tubes(w,P,SIGMA,f,plan['alpha'],plan['angle_grid'])
        shared=shared_parameter_tube(w,P,SIGMA,f,plan['alpha'],plan['angle_grid'])
        lo,hi=family(w,f);outerlo=lo-basic['paired_radius'];outerhi=hi+basic['paired_radius']
        torch.manual_seed(plan['optimizer_seed']+rep)
        G=torch.randn((2,Q,plan['random_restarts'],6,2),dtype=torch.float64)
        G*=radius/G.norm(dim=(-2,-1),keepdim=True);G[:,:,0]=0.;G.requires_grad_(True)
        opt=torch.optim.Adam([G],lr=plan['learning_rate']);fw=torch.tensor(f,dtype=torch.float64)
        Sw=torch.tensor(S,dtype=torch.float64);ww=torch.tensor(w,dtype=torch.float64)
        bestlo=torch.tensor(lo,dtype=torch.float64);besthi=torch.tensor(hi,dtype=torch.float64)
        bestG=torch.zeros((2,Q,6,2),dtype=torch.float64)
        curve=[]
        for step in range(plan['steps']+1):
            weights=ww+Sw@G;l,h=torch_family(weights,fw)
            with torch.no_grad():
                lv,li=l[0].min(-1);hv,hi_idx=h[1].max(-1)
                improve_l=lv<bestlo;improve_h=hv>besthi
                bestG[0,improve_l]=G[0,torch.arange(Q)[improve_l],li[improve_l]]
                bestG[1,improve_h]=G[1,torch.arange(Q)[improve_h],hi_idx[improve_h]]
                bestlo=torch.minimum(bestlo,lv);besthi=torch.maximum(besthi,hv)
            if step%50==0:curve.append({'step':step,'inner_mean_width':float((besthi-bestlo).mean())})
            if step==plan['steps']:break
            loss=l[0].sum()-h[1].sum();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            with torch.no_grad():G.mul_(torch.clamp(radius/G.norm(dim=(-2,-1),keepdim=True),max=1.))
        blo=bestlo.numpy();bhi=besthi.numpy();width=bhi-blo
        saved_g=bestG.numpy();assert np.linalg.norm(saved_g,axis=(-2,-1)).max()<=radius+1e-12
        # Independent numpy replay of EVERY saved feasible endpoint model.
        for qi in range(Q):
            wl=w+S@saved_g[0,qi];wh=w+S@saved_g[1,qi]
            ll,_=family(wl,f[qi:qi+1]);_,hh=family(wh,f[qi:qi+1])
            assert abs(ll[0]-blo[qi])<1e-9 and abs(hh[0]-bhi[qi])<1e-9
        np.savez_compressed(witness_dir/f'rep{rep}.npz',w_hat=w,P=P,factual=f,whitened_parameter_witnesses=saved_g,lo=blo,hi=bhi)
        # A feasible inner range must lie within both sound outer bounds.
        assert np.all(blo>=outerlo-1e-9) and np.all(bhi<=outerhi+1e-9)
        assert np.all(blo>=shared['lo']-1e-9) and np.all(bhi<=shared['hi']+1e-9)
        maxreduction=1-width/(outerhi-outerlo)
        row={'rep':rep,'training_n':n,'horizon':H,'queries':Q,
             'feasible_inner_mean_width':float(width.mean()),'basic_outer_mean_width':float((outerhi-outerlo).mean()),
             'shared_outer_mean_width':float((shared['hi']-shared['lo']).mean()),
             'maximum_possible_mean_width_reduction_vs_basic':float(1-width.mean()/(outerhi-outerlo).mean()),
             'max_parameter_norm':float(G.detach().norm(dim=(-2,-1)).max()),'confidence_radius':float(radius),
             'optimization_curve':curve}
        records.append(row);print(json.dumps(row),flush=True)
    inner=np.mean([r['feasible_inner_mean_width'] for r in records]);basic=np.mean([r['basic_outer_mean_width'] for r in records])
    shared=np.mean([r['shared_outer_mean_width'] for r in records])
    result={'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'records':records,
            'mean_inner_width':float(inner),'mean_basic_outer_width':float(basic),'mean_shared_outer_width':float(shared),
            'maximum_possible_width_reduction_vs_basic':float(1-inner/basic),
            'interpretation':'Feasible lower bound on required width. No optimization convergence or outer-coverage claim. A larger inner range strengthens this falsification.',
            'runtime_seconds':time.perf_counter()-start}
    (ROOT/'outputs/feasible_extrema.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FEASIBLE_EXTREMA_COMPLETE',result['runtime_seconds'],flush=True)
if __name__=='__main__':main()
