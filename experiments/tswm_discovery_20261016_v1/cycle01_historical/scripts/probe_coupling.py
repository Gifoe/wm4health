"""Likelihood-preserving counterfactual witnesses for continuous dynamics.

No novelty is claimed for the classical nonidentification principle or for
Procrustes. The question is whether a cheap continuous-model witness is useful.
All factual future states are available: this is retrospective counterfactual
analysis, explicitly not prospective forecasting with future state access.
"""
import hashlib,json,pathlib,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
PLAN=ROOT/'outputs/coupling_probe_plan.json'
def rotation(theta):return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
A=np.array([[.93,.035],[-.02,.86]])
D=np.array([[-.015,.012],[.008,.02]])
B=np.array([.018,-.012]);BIAS=np.array([.002,-.001]);SIGMA=.12
def features(x,a):return np.column_stack([x,a,x*a,np.ones(len(x))])
TRUE=np.row_stack([A.T,B,D.T,BIAS])
def mean(x,a,w=TRUE):return features(x,a)@w
def simulate(x0,actions,eps,w=TRUE,theta=0.,chol=None):
    chol=np.eye(2)*SIGMA if chol is None else chol
    n,H=actions.shape;states=np.empty((n,H+1,2));states[:,0]=x0
    for t in range(H):
        ang=actions[:,t]*theta;c=np.cos(ang);s=np.sin(ang)
        e=eps[:,t];rot=np.column_stack([c*e[:,0]-s*e[:,1],s*e[:,0]+c*e[:,1]])
        states[:,t+1]=mean(states[:,t],actions[:,t,None],w)+rot@chol.T
    return states
def certificate(x0,factual,eps,w,chol):
    """Exact global-angle terminal range via three linear recurrences."""
    n,H,_=eps.shape;det=x0.copy();c=np.zeros_like(x0);s=np.zeros_like(x0)
    ones=np.ones((n,1));mat=w[:2].T+w[3:5].T
    J=np.array([[0.,-1.],[1.,0.]])
    for t in range(H):
        det=mean(det,ones,w)
        c=c@mat.T+eps[:,t]@chol.T
        s=s@mat.T+eps[:,t]@J.T@chol.T
    center=det[:,0]-factual[:,-1,0]
    rad=np.sqrt(c[:,0]**2+s[:,0]**2)
    return center-rad,center+rad,center,c[:,0],s[:,0]
def fit(n,seed):
    r=np.random.default_rng(seed);x=r.normal(size=(n,2));a=r.uniform(-1,1,(n,1))
    y=mean(x,a)+r.normal(scale=SIGMA,size=x.shape)
    z=features(x,a);w=np.linalg.lstsq(z,y,rcond=None)[0]
    resid=y-z@w;cov=resid.T@resid/len(x);chol=np.linalg.cholesky(cov)
    return w,chol
def run():
    plan=json.loads(PLAN.read_text());start=time.perf_counter();records=[]
    r=np.random.default_rng(981016);maxH=max(plan['horizons']);N=plan['discovery_episodes']
    x0=r.normal(scale=.4,size=(N,2));eps=r.normal(size=(N,maxH,2))
    zero=np.zeros((N,maxH));factual=simulate(x0,zero,eps)
    arrays={};estimates={}
    for train_n in plan['training_sizes']:
      estimates[train_n]=[]
      for seed in plan['training_seeds']:
        w,chol=fit(train_n,seed)
        # Abduction from observed factual path under the fitted model.
        residual=factual[:,1:]-mean(factual[:,:-1].reshape(-1,2),zero.reshape(-1,1),w).reshape(N,maxH,2)
        abduct=np.linalg.solve(chol,residual.reshape(-1,2).T).T.reshape(N,maxH,2)
        estimates[train_n].append((w,chol,abduct))
        for H in plan['horizons']:
            xx=factual[:,:H+1];ee=abduct[:,:H]
            lo,hi,center,c,s=certificate(x0,xx,ee,w,chol)
            cf0=simulate(x0,np.ones((N,H)),ee,w,0.,chol)
            delta0=cf0[:,-1,0]-xx[:,-1,0]
            assert np.max(np.abs(delta0-(center+c)))<1e-10
            max_nll=0.;effects=[]
            # Equivalent kernels have identical whitened innovation norms under
            # all gauges. This is a pointwise algebraic, not Monte Carlo, check.
            flat=ee.reshape(-1,2);base_norm=(flat*flat).sum(-1)
            for angle in plan['gauge_angles']:
                R=rotation(angle);assert np.max(np.abs(R.T@R-np.eye(2)))<1e-12
                alt_norm=((flat@R)**2).sum(-1)
                max_nll=max(max_nll,float(np.max(np.abs(alt_norm-base_norm))*.5))
                cf=simulate(x0,np.ones((N,H)),ee,w,angle,chol)
                delta=cf[:,-1,0]-xx[:,-1,0]
                assert np.all(delta>=lo-1e-10) and np.all(delta<=hi+1e-10)
                effects.append(delta)
            effects=np.stack(effects)
            row={'train_n':train_n,'seed':seed,'horizon':H,'n':N,'kernel_nll_max_abs_difference':max_nll,
                 'coefficient_rmse':float(np.sqrt(np.mean((w-TRUE)**2))),
                 'counterfactual_family_width_mean':float((hi-lo).mean()),
                 'sign_ambiguous_fraction':float(((lo<0)&(hi>0)).mean()),
                 'gauge_sign_disagreement_fraction':float(np.any(np.sign(effects)!=np.sign(effects[0]),axis=0).mean()),
                 'max_mean_abs_change_from_common_seed':float(np.abs(effects-effects[:1]).mean(axis=1).max()),
                 'cf0_mean_effect':float(delta0.mean())}
            records.append(row)
            if train_n==max(plan['training_sizes']):
                arrays[f's{seed}_H{H}_lo']=lo;arrays[f's{seed}_H{H}_hi']=hi;arrays[f's{seed}_H{H}_effects']=effects
    # Ordinary fitted-parameter disagreement is distinct from coupling ambiguity.
    H=maxH;nmax=max(plan['training_sizes']);preds=[]
    for w,chol,abduct in estimates[nmax]:
        cf=simulate(x0,np.ones((N,H)),abduct,w,0.,chol)
        preds.append(cf[:,-1,0]-factual[:,-1,0])
    param_range=np.ptp(preds,axis=0)
    # Additional paired simulator transitions identify a rotation by the standard
    # orthogonal Procrustes estimator. They are extra information, not free labels.
    r=np.random.default_rng(131016);x=r.normal(size=(16,2));e=r.normal(size=(16,2));theta=.7
    y0=mean(x,np.zeros((16,1)))+SIGMA*e
    y1=mean(x,np.ones((16,1)))+SIGMA*e@rotation(theta).T
    e0=(y0-mean(x,np.zeros((16,1))))/SIGMA;e1=(y1-mean(x,np.ones((16,1))))/SIGMA
    U,S,Vt=np.linalg.svd(e0.T@e1);Rhat=U@Vt
    recovered=np.arctan2(Rhat.T[1,0],Rhat.T[0,0])
    assert abs(recovered-theta)<1e-10
    np.savez_compressed(ROOT/'outputs/coupling_probe_arrays.npz',**arrays)
    result={'plan_sha256':hashlib.sha256(PLAN.read_bytes()).hexdigest(),'scope':'new Gaussian controlled-system discovery; exact ranges only within rotation family',
            'not_a_novelty_claim':'Nonidentification and robust counterfactual bounds predate this work.',
            'records':records,'parameter_ensemble_cf_range_mean':float(param_range.mean()),
            'paired_anchor_control':{'n':16,'true_angle':theta,'recovered_angle':float(recovered),'estimator':'known orthogonal Procrustes; paired simulator outcomes are additional unavailable-in-general data'},
            'runtime_seconds':time.perf_counter()-start}
    (ROOT/'outputs/coupling_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'last':records[-1],'parameter_ensemble_range':float(param_range.mean()),'paired_control':result['paired_anchor_control'],'seconds':result['runtime_seconds']},indent=2))
if __name__=='__main__':run()
