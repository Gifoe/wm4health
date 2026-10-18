"""Independent action-interface qualification; established conjugacy, no novelty claim."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json, hashlib, time, datetime
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[2]
CYCLE=ROOT/'cycle84'
OUT=CYCLE/'outputs'
STORE=json.loads((ROOT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
RAW=Path(STORE['artifact_root'])/'cycle84'/'qualification_v1'
SEEDS=range(2026841,2026845)
DT=.05
N=128
M=np.array([[1.1,.35],[-.25,.9]],dtype=np.float64)
MINV=np.linalg.inv(M)

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj,indent=2,allow_nan=False).encode()
    with path.open('xb') as f:f.write(raw)
    return hashlib.sha256(raw).hexdigest()

def warp(x,kind):
    x=np.asarray(x,dtype=np.float64)
    if kind=='linear':return x@M.T
    if kind=='triangular':
        w=x.copy();w[...,0]+=.3*np.tanh(w[...,1]);return w@M.T
    assert kind=='actuation_preserving'
    w=x.copy();w[...,0]+=.25*np.tanh(w[...,0]);return w

def inverse(y,kind):
    y=np.asarray(y,dtype=np.float64)
    if kind=='linear':return y@MINV.T
    if kind=='triangular':
        w=y@MINV.T;w[...,0]-=.3*np.tanh(w[...,1]);return w
    assert kind=='actuation_preserving'
    w=y.copy();q=w[...,0].copy()
    for _ in range(12):q-=(q+.25*np.tanh(q)-y[...,0])/(1+.25/np.cosh(q)**2)
    w[...,0]=q;return w

def jacobian(x,kind):
    if kind=='linear':return M.copy()
    if kind=='triangular':return M@np.array([[1,.3/np.cosh(x[1])**2],[0,1]])
    return np.diag([1+.25/np.cosh(x[0])**2,1.])

def field(x,u,system):
    q,v=x
    if system=='linear':return np.array([-.3*q+.6*v,-.2*q-.4*v+u])
    assert system=='duffing'
    return np.array([v,-.8*q-.35*v-.2*q**3+u])

def rk(x,u,system):
    a=field(x,u,system);b=field(x+.5*DT*a,u,system)
    c=field(x+.5*DT*b,u,system);d=field(x+DT*c,u,system)
    return x+DT*(a+2*b+2*c+d)/6

def actions(seed,plan,count=N):
    r=np.random.default_rng(np.random.SeedSequence([seed,plan,84]))
    return np.repeat(r.uniform(-.5,.5,size=(count+7)//8),8)[:count]

def pair(system,kind,u,clamp=False):
    x=np.zeros(2);z=warp(x,kind);a=[warp(x,kind)];b=[z.copy()]
    for t,ut in enumerate(u):
        if clamp and t==64:
            x[0]=.8;z[0]=.8
            a[-1]=warp(x,kind);b[-1]=z.copy()
        x=rk(x,float(ut),system)
        z=warp(rk(inverse(z,kind),float(ut),system),kind)
        a.append(warp(x,kind));b.append(z.copy())
    return np.asarray(a),np.asarray(b)

def independent_pair(system,kind,u):
    x=np.zeros(2);z=warp(x,kind);aa=[warp(x,kind)];bb=[z.copy()]
    for ut in u:
        a=solve_ivp(lambda t,w:field(w,float(ut),system),(0,DT),x,method='DOP853',rtol=1e-10,atol=1e-12)
        b=solve_ivp(lambda t,w:jacobian(inverse(w,kind),kind)@field(inverse(w,kind),float(ut),system),(0,DT),z,method='DOP853',rtol=1e-10,atol=1e-12)
        assert a.success and b.success
        x=a.y[:,-1];z=b.y[:,-1];aa.append(warp(x,kind));bb.append(z.copy())
    return np.asarray(aa),np.asarray(bb)

def main():
    started=time.perf_counter();RAW.mkdir(parents=True,exist_ok=False);OUT.mkdir(exist_ok=True)
    protocol=CYCLE/'QUALIFICATION_PROTOCOL84_V1.md'
    rows=[];saved={};ix=0
    maxeq=0.;maxode=0.;maxjac=0.;maxinverse=0.;maxols=0.;maxB=0.
    maps=['linear','triangular','actuation_preserving']
    for kind in maps:
        pts=np.random.default_rng(20268400).normal(size=(100,2))
        inv=float(np.max(np.abs(inverse(warp(pts,kind),kind)-pts)))
        maxinverse=max(maxinverse,inv)
        for p in pts:
            eps=1e-5
            numeric=np.column_stack([(warp(p+eps*np.eye(2)[j],kind)-warp(p-eps*np.eye(2)[j],kind))/(2*eps) for j in range(2)])
            maxjac=max(maxjac,float(np.max(np.abs(numeric-jacobian(p,kind)))))
            if kind=='actuation_preserving':maxB=max(maxB,float(np.max(np.abs(jacobian(p,kind)@np.array([0.,1.])-np.array([0.,1.])))))
    for system in ['linear','duffing']:
        for kind in maps:
            for seed in SEEDS:
                for plan in range(8):
                    u=actions(seed,plan);a,b=pair(system,kind,u)
                    error=float(np.max(np.abs(a-b)));maxeq=max(maxeq,error)
                    key='pair_'+str(ix);saved[key+'_u']=u;saved[key+'_observation_world']=a;saved[key+'_dynamics_world']=b
                    rows.append(dict(kind='common_action_equivalence',system=system,map=kind,seed=seed,plan=plan,array_key=key,max_abs_error=error))
                    ix+=1
                    if plan<4:
                        c,d=pair(system,kind,u,True)
                        key='named_'+str(ix);saved[key+'_observation_world']=c;saved[key+'_dynamics_world']=d
                        other=float(np.sqrt(np.mean((c[65:,1]-d[65:,1])**2)))
                        rows.append(dict(kind='different_named_state_interventions',system=system,map=kind,seed=seed,plan=plan,array_key=key,max_abs_difference=float(np.max(np.abs(c-d))),unclamped_coordinate_rmse=other,not_a_common_action_failure=True))
                    if seed==2026841 and plan<2:
                        c,d=independent_pair(system,kind,u);err=float(np.max(np.abs(c-d)));maxode=max(maxode,err)
                        key='ode_'+str(ix);saved[key+'_observation_world']=c;saved[key+'_dynamics_world']=d
                        rows.append(dict(kind='independent_continuous_fields',system=system,map=kind,seed=seed,plan=plan,array_key=key,max_abs_error=err))
    # Ordinary affine dynamics fit in the observation coordinates. No target physical-state pairs.
    for seed in SEEDS:
        for budget in [64,256]:
            u=actions(seed+100000,0,budget);x=np.zeros(2);ys=[warp(x,'linear')]
            for ut in u:x=rk(x,float(ut),'linear');ys.append(warp(x,'linear'))
            y=np.asarray(ys);design=np.column_stack([y[:-1],u,np.ones(budget)])
            coef,_,rank,singular=np.linalg.lstsq(design,y[1:],rcond=None);assert rank==4
            k=f'ols_{seed}_{budget}';saved[k+'_train_y']=y;saved[k+'_train_u']=u;saved[k+'_coef']=coef
            for plan in range(8):
                utest=actions(seed+200000,plan);truth,_=pair('linear','linear',utest);pred=[np.zeros(2)]
                for ut in utest:pred.append(np.r_[pred[-1],ut,1.]@coef)
                pred=np.asarray(pred);err=float(np.max(np.abs(pred-truth)));maxols=max(maxols,err)
                key=k+'_'+str(plan);saved[key+'_prediction']=pred;saved[key+'_reference']=truth
                rows.append(dict(kind='ordinary_linear_output_fit',seed=seed,budget=budget,plan=plan,array_key=key,coefficient_key=k,design_rank=int(rank),smallest_singular=float(singular[-1]),max_abs_error=err))
    for val in saved.values():assert np.isfinite(val).all()
    np.savez_compressed(RAW/'arrays.npz',**saved)
    array_sha=hashlib.sha256((RAW/'arrays.npz').read_bytes()).hexdigest()
    counts={kind:sum(r['kind']==kind for r in rows) for kind in sorted({r['kind'] for r in rows})}
    assert counts==dict(common_action_equivalence=192,different_named_state_interventions=96,independent_continuous_fields=12,ordinary_linear_output_fit=64)
    assert maxeq<=1e-10 and maxode<=2e-6 and maxjac<=1e-7 and maxinverse<=1e-10 and maxols<=1e-8 and maxB==0.
    result=dict(status='QUALIFICATION_PASS_NOT_A_NEW_RESEARCH_GAP',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),raw_array_path=str(RAW/'arrays.npz'),raw_array_sha256=array_sha,counts=counts,ordinary_fits=8,max_common_action_equivalence_error=maxeq,max_independent_ode_error=maxode,max_jacobian_fd_error=maxjac,max_inverse_error=maxinverse,max_ordinary_linear_rollout_error=maxols,max_known_actuation_direction_error=maxB,rows=rows,wall_seconds=time.perf_counter()-started,scope='Established state-space conjugacy qualification; no neural baseline reproduced; no adaptation method failure claimed; named internal-coordinate interventions differ in physical meaning',qualified_research_directions=0)
    sha=write(OUT/'QUALIFICATION84_V1.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True);print('RESULT_SHA256',sha,flush=True)

if __name__=='__main__':main()
