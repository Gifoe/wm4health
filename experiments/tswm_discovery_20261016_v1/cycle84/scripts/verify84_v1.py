"""Recompute saved evidence without importing the experiment implementation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json, hashlib, datetime
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'cycle84'/'outputs'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def unmap(y,kind):
    if kind=='actuation_preserving':
        return np.array([brentq(lambda q:q+.25*np.tanh(q)-y[0],y[0]-.251,y[0]+.251,xtol=1e-14),y[1]])
    z=np.linalg.solve(np.array([[1.1,.35],[-.25,.9]]),y)
    if kind=='triangular':z[0]-=.3*np.tanh(z[1])
    return z

def observe(z,kind):
    a,b=z
    if kind=='actuation_preserving':return np.array([a+.25*np.tanh(a),b])
    if kind=='triangular':a+=.3*np.tanh(b)
    return np.array([1.1*a+.35*b,-.25*a+.9*b])

def continuous(z,action,system):
    if system=='linear':return np.array([-.3*z[0]+.6*z[1],-.2*z[0]-.4*z[1]+action])
    return np.array([z[1],-.8*z[0]-.35*z[1]-.2*z[0]**3+action])

def main():
    result_path=OUT/'QUALIFICATION84_V1.json';r=json.loads(result_path.read_text())
    assert sha(ROOT/'cycle84'/'QUALIFICATION_PROTOCOL84_V1.md')==r['protocol_sha256']
    assert sha(ROOT/'cycle84'/'scripts'/'qualification84_v1.py')==r['source_sha256']
    arrays=Path(r['raw_array_path']);assert sha(arrays)==r['raw_array_sha256']
    d=np.load(arrays,allow_pickle=False)
    counts={};maxeq=0.;maxresponse=0.;maxols=0.;maxcoef=0.;maxcont=0.;maxexact=0.;minnamed=float('inf');scalar_checks=0
    index={}
    for row in r['rows']:
        kind=row['kind'];counts[kind]=counts.get(kind,0)+1;k=row['array_key']
        if kind in ['common_action_equivalence','independent_continuous_fields']:
            a=d[k+'_observation_world'];b=d[k+'_dynamics_world']
            assert a.shape==b.shape==(129,2)
            error=float(np.max(np.abs(a-b)));assert abs(error-row['max_abs_error'])<=1e-20
            if kind=='common_action_equivalence':
                maxeq=max(maxeq,error);index[row['system'],row['map'],row['seed'],row['plan']]=(a,b)
            else:
                # Independent reference ODE and scalar root inversion, rather than shared recurrence.
                actionrow=next(z for z in r['rows'] if z['kind']=='common_action_equivalence' and all(z[t]==row[t] for t in ['system','map','seed','plan']))
                u=d[actionrow['array_key']+'_u'];x=np.zeros(2);ys=[observe(x,row['map'])]
                for ut in u:
                    sol=solve_ivp(lambda t,w:continuous(w,float(ut),row['system']),(0,.05),x,method='DOP853',rtol=1e-11,atol=1e-13)
                    assert sol.success;x=sol.y[:,-1];ys.append(observe(x,row['map']))
                maxcont=max(maxcont,float(np.max(np.abs(np.asarray(ys)-a))))
                for y in a[::8]:
                    assert np.max(np.abs(observe(unmap(y,row['map']),row['map'])-y))<1e-12
            scalar_checks+=a.size+b.size
        elif kind=='ordinary_linear_output_fit':
            p=d[k+'_prediction'];y=d[k+'_reference'];err=float(np.max(np.abs(p-y)));maxols=max(maxols,err)
            assert abs(err-row['max_abs_error'])<=1e-20
            ck=row['coefficient_key'];train_y=d[ck+'_train_y'];train_u=d[ck+'_train_u']
            design=np.column_stack([train_y[:-1],train_u,np.ones(len(train_u))])
            q,t=np.linalg.qr(design,mode='reduced');coef=np.linalg.solve(t,q.T@train_y[1:])
            maxcoef=max(maxcoef,float(np.max(np.abs(coef-d[ck+'_coef']))))
            # The continuous linear source has a separate exact matrix-exponential solution.
            source=np.array([[-.3,.6,0.],[-.2,-.4,1.],[0.,0.,0.]])
            transition=expm(.05*source);state=np.zeros(2);exact=[observe(state,'linear')]
            actionrow=next(z for z in r['rows'] if z['kind']=='common_action_equivalence' and z['system']=='linear' and z['map']=='linear' and z['seed']==row['seed'] and z['plan']==row['plan'])
            # Fitted baseline test actions have a separately declared seed namespace.
            rng=np.random.default_rng(np.random.SeedSequence([row['seed']+200000,row['plan'],84]));u=np.repeat(rng.uniform(-.5,.5,16),8)
            for ut in u:state=(transition@np.r_[state,ut])[:2];exact.append(observe(state,'linear'))
            maxexact=max(maxexact,float(np.max(np.abs(np.asarray(exact)-y))))
            scalar_checks+=p.size+y.size
        else:
            assert kind=='different_named_state_interventions'
            a=d[k+'_observation_world'];b=d[k+'_dynamics_world'];err=float(np.max(np.abs(a-b)))
            other=float(np.sqrt(np.mean((a[65:,1]-b[65:,1])**2)))
            assert abs(err-row['max_abs_difference'])<=1e-20 and abs(other-row['unclamped_coordinate_rmse'])<=1e-20
            assert row['not_a_common_action_failure'] is True
            minnamed=min(minnamed,other);scalar_checks+=a.size+b.size
    for system,map_name,seed,plan in index:
        if plan==0:continue
        a,b=index[system,map_name,seed,plan];a0,b0=index[system,map_name,seed,0]
        maxresponse=max(maxresponse,float(np.max(np.abs((a-a0)-(b-b0)))))
    assert counts==r['counts'] and maxeq<=1e-10 and maxresponse<=2e-10 and maxols<=1e-8 and maxcoef<=1e-9 and maxcont<=2e-6 and maxexact<=2e-6 and minnamed>0.
    v=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),result_sha256=sha(result_path),arrays_sha256=sha(arrays),protocol_sha256=r['protocol_sha256'],counts=counts,saved_trajectory_scalar_values_checked=scalar_checks,common_action_response_pairs=168,max_common_action_error=maxeq,max_paired_action_response_error=maxresponse,max_independent_ode_reference_error=maxcont,max_exact_linear_reference_discrepancy=maxexact,max_independent_qr_coefficient_difference=maxcoef,max_ordinary_fit_rollout_error=maxols,min_different_named_intervention_recipient_rmse=minnamed,interpretation='Conjugacy qualification only; the different named-state operators are not an action-response failure',local_payload_files_written=0)
    raw=json.dumps(v,indent=2).encode()
    with (OUT/'VERIFICATION84_V1.json').open('xb') as f:f.write(raw)
    print(json.dumps(v,indent=2),flush=True)

if __name__=='__main__':main()
