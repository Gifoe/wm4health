"""Family-specific confidence tubes for a controlled affine Gaussian system.

This is a minimal prototype of ordinary regression confidence sets propagated
through retrospective dynamics, not a general neural or all-SCM certificate.
The noise scale is KNOWN, states fully observed, and the coupling family is one
global SO(2) rotation for action 1 versus action 0. No latent confounding.
"""
import numpy as np
from scipy.stats import chi2

def features(x,a):
    a=np.broadcast_to(np.asarray(a),x.shape[:-1]+(1,))
    return np.concatenate([x,a,x*a,np.ones_like(a)],axis=-1)

def mean(w,x,a):return features(x,a)@w

def simulate(w,x0,eps,action,angle=0.):
    x=x0.copy();out=[x.copy()]
    c,s=np.cos(angle*action),np.sin(angle*action)
    R=np.array([[c,-s],[s,c]])
    for t in range(eps.shape[1]):
        x=mean(w,x,action)+eps[:,t]@R.T;out.append(x.copy())
    return np.stack(out,axis=1)

def family(w,factual):
    """Exact endpoint interval for the stated constant-angle family."""
    residual=factual[:,1:]-mean(w,factual[:,:-1],0.)
    d=factual[:,0].copy();c=np.zeros_like(d);s=np.zeros_like(d)
    A1=w[:2].T+w[3:5].T;J=np.array([[0.,-1.],[1.,0.]])
    for t in range(residual.shape[1]):
        d=mean(w,d,1.);c=c@A1.T+residual[:,t];s=s@A1.T+residual[:,t]@J.T
    center=d[:,0]-factual[:,-1,0];rad=np.hypot(c[:,0],s[:,0])
    return center-rad,center+rad

def regression_fit(x,a,y):
    z=features(x,a);P=np.linalg.inv(z.T@z);w=np.linalg.lstsq(z,y,rcond=None)[0]
    return w,P

def tubes(w,P,sigma,factual,alpha=.05,angles=64):
    """Uniform over continuous theta, episodes, and all W in ONE confidence set.

    If ||P^(-1/2)(W-W_hat)||_F <= r=sigma*sqrt(chi2_12(.95)),
    each predicted family endpoint is within `paired_radius` of its true
    counterpart. Grid maximization includes an analytic interpolation remainder.
    The cancelled term bounds DeltaW^T phi_cf - R DeltaW^T phi_f jointly;
    `separate_radius` uses the looser sum of two norms. Both are valid.
    """
    lam,U=np.linalg.eigh(P);S=(U*np.sqrt(lam))@U.T
    r=sigma*np.sqrt(chi2.ppf(1-alpha,w.size))
    M=np.vstack([np.eye(2),np.zeros((1,2)),np.eye(2),np.zeros((1,2))])
    A1=w[:2].T+w[3:5].T
    rho=np.linalg.norm(A1,2)+r*np.linalg.norm(S@M,2)
    grid=np.arange(angles)*(2*np.pi/angles);co=np.cos(grid);si=np.sin(grid)
    delta=np.pi/angles
    d=factual[:,0].copy();c=np.zeros_like(d);s=np.zeros_like(d)
    paired=np.zeros(len(d));separate=np.zeros(len(d));fixed=np.zeros(len(d))
    residual=factual[:,1:]-mean(w,factual[:,:-1],0.)
    J=np.array([[0.,-1.],[1.,0.]])
    for t in range(residual.shape[1]):
        vd=features(d,1.)@S
        vc=features(c,1.)@S-features(np.zeros_like(c),1.)@S
        vs=features(s,1.)@S-features(np.zeros_like(s),1.)@S
        vf=features(factual[:,t],0.)@S
        v=vd[:,None,:]+vc[:,None,:]*co[None,:,None]+vs[:,None,:]*si[None,:,None]
        vnorm=np.linalg.norm(v,axis=-1);fnorm=np.linalg.norm(vf,axis=-1)
        sq=vnorm**2+fnorm[:,None]**2-2*co[None,:]*np.sum(v*vf[:,None,:],axis=-1)
        kgrid=np.sqrt(np.maximum(sq,0)).max(axis=1)
        derivative=np.linalg.norm(vc,axis=-1)+np.linalg.norm(vs,axis=-1)
        certified_k=kgrid+(derivative+fnorm)*delta
        sep_k=vnorm.max(axis=1)+derivative*delta+fnorm
        paired=rho*paired+r*certified_k
        separate=rho*separate+r*sep_k
        vfixed=vd+vc-vf
        fixed=rho*fixed+r*np.linalg.norm(vfixed,axis=-1)
        d=mean(w,d,1.);c=c@A1.T+residual[:,t];s=s@A1.T+residual[:,t]@J.T
    return {'paired_radius':paired,'separate_radius':separate,'fixed_common_noise_radius':fixed,'rho_bound':rho,'r':r}

def shared_parameter_tube(w,P,sigma,factual,alpha=.05,angles=64):
    """First-order shared-parameter sensitivity with a sound quadratic remainder.

    This adapts standard Taylor/validated-reachability reasoning. It does not
    claim to invent parameter sharing, Taylor models, or partial identification.
    The same whitened coefficient vector appears at every step, so the endpoint
    sensitivity is summed BEFORE its support function is evaluated. The
    second-order remainder restores soundness; omitting it is an ablation only.
    """
    lam,U=np.linalg.eigh(P);S=(U*np.sqrt(lam))@U.T
    r=sigma*np.sqrt(chi2.ppf(1-alpha,w.size))
    M=np.vstack([np.eye(2),np.zeros((1,2)),np.eye(2),np.zeros((1,2))])
    A1=w[:2].T+w[3:5].T
    q=r*np.linalg.norm(S@M,2);rho=np.linalg.norm(A1,2)+q
    grid=np.arange(angles)*(2*np.pi/angles);co=np.cos(grid);si=np.sin(grid);delta=np.pi/angles
    d=factual[:,0].copy();c=np.zeros_like(d);s=np.zeros_like(d)
    Jd=np.zeros((len(d),2,w.size));Jc=np.zeros_like(Jd);Js=np.zeros_like(Jd)
    remainder=np.zeros(len(d));I=np.eye(2);J=np.array([[0.,-1.],[1.,0.]])
    residual=factual[:,1:]-mean(w,factual[:,:-1],0.)
    def K(v,R):return np.einsum('np,ij->nipj',v,R).reshape(len(d),2,w.size)
    def matnorm(a):return np.linalg.norm(a,2,axis=(-2,-1))
    for t in range(residual.shape[1]):
        # Uniform certified norm of the CURRENT derivative, needed by the
        # quadratic remainder recurrence before advancing J to the next step.
        JJ=Jd[:,None]+Jc[:,None]*co[None,:,None,None]+Js[:,None]*si[None,:,None,None]
        jbound=matnorm(JJ).max(axis=1)+(matnorm(Jc)+matnorm(Js))*delta
        remainder=rho*remainder+q*r*jbound
        vd=features(d,1.)@S
        vc=(features(c,1.)-features(np.zeros_like(c),1.))@S
        vs=(features(s,1.)-features(np.zeros_like(s),1.))@S
        vf=features(factual[:,t],0.)@S
        Jd=np.einsum('ij,njk->nik',A1,Jd)+K(vd,I)
        Jc=np.einsum('ij,njk->nik',A1,Jc)+K(vc-vf,I)
        Js=np.einsum('ij,njk->nik',A1,Js)+K(vs,I)-K(vf,J)
        d=mean(w,d,1.);c=c@A1.T+residual[:,t];s=s@A1.T+residual[:,t]@J.T
    center=d[:,0]-factual[:,-1,0]
    values=center[:,None]+c[:,0,None]*co[None,:]+s[:,0,None]*si[None,:]
    readout=Jd[:,None,0]+Jc[:,None,0]*co[None,:,None]+Js[:,None,0]*si[None,:,None]
    support=r*np.linalg.norm(readout,axis=-1)
    interp=(np.hypot(c[:,0],s[:,0])+r*(np.linalg.norm(Jc[:,0],axis=-1)+np.linalg.norm(Js[:,0],axis=-1)))*delta
    linear_lo=(values-support).min(axis=1)-interp
    linear_hi=(values+support).max(axis=1)+interp
    return {'lo':linear_lo-remainder,'hi':linear_hi+remainder,
            'linear_only_lo':linear_lo,'linear_only_hi':linear_hi,
            'quadratic_remainder':remainder,'rho_bound':rho,'r':r}
