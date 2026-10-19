"""Predeclared fresh confirmation of a limited continuous-CF certificate."""
import hashlib,json,pathlib,time
import numpy as np
from scipy.stats import chi2
from confidence_tube import features,mean,simulate,family,regression_fit,tubes
ROOT=pathlib.Path(__file__).resolve().parents[1]
SIGMA=.12
A=np.array([[.93,.035],[-.02,.86]])
D=np.array([[-.015,.012],[.008,.02]])
TRUE=np.vstack([A.T,np.array([.018,-.012]),D.T,np.array([.002,-.001])])

def trial(n,rep,plan):
    rng=np.random.default_rng(plan['training_seed_base']+100000*n+rep)
    x=rng.normal(size=(n,2));a=rng.uniform(-1,1,(n,1))
    y=mean(TRUE,x,a)+rng.normal(scale=SIGMA,size=(n,2))
    w,P=regression_fit(x,a,y)
    conf_stat=np.sum((w-TRUE)*(np.linalg.solve(P,w-TRUE)))/SIGMA**2
    event=conf_stat<=chi2.ppf(1-plan['alpha'],w.size)
    rg=np.random.default_rng(plan['confirmation_seed_base']+rep)
    N=plan['episodes_per_repetition'];Hmax=max(plan['horizons'])
    x0=rg.normal(scale=.4,size=(N,2));eps=rg.normal(scale=SIGMA,size=(N,Hmax,2))
    factual=simulate(TRUE,x0,eps,0.);records=[]
    for H in plan['horizons']:
        f=factual[:,:H+1];true_lo,true_hi=family(TRUE,f);hat_lo,hat_hi=family(w,f)
        tube=tubes(w,P,SIGMA,f,plan['alpha'],plan['angle_grid'])
        residual=f[:,1:]-mean(w,f[:,:-1],0.)
        point=simulate(w,x0,residual,1.,0.)[:,-1,0]-f[:,-1,0]
        truth=simulate(TRUE,x0,eps[:,:H],1.,plan['true_angle'])[:,-1,0]-f[:,-1,0]
        rr=tube['paired_radius'];rs=tube['separate_radius']
        methods={'gauge_only':(hat_lo,hat_hi),
                 'statistical_only_fixed_common_noise':(point-tube['fixed_common_noise_radius'],point+tube['fixed_common_noise_radius']),
                 'joint_paired_tube':(hat_lo-rr,hat_hi+rr),
                 'joint_separate_tube':(hat_lo-rs,hat_hi+rs),
                 'oracle_known_kernel_family':(true_lo,true_hi)}
        for method,(lo,hi) in methods.items():
            cov=(lo<=truth+1e-10)&(hi>=truth-1e-10)
            full=(lo<=true_lo+1e-10)&(hi>=true_hi-1e-10)
            rec={'training_n':n,'rep':rep,'horizon':H,'method':method,
                 'coefficient_confidence_event':bool(event),'coefficient_chi2':float(conf_stat),
                 'point_coverage':float(cov.mean()),'whole_family_coverage':float(full.mean()),
                 'all_episodes_family_covered':bool(full.all()),
                 'mean_width':float((hi-lo).mean()),'oracle_mean_width':float((true_hi-true_lo).mean()),
                 'sign_certifiable_fraction':float(((lo>0)|(hi<0)).mean()),
                 'rho_bound':float(tube['rho_bound'])}
            records.append(rec)
            if method.startswith('joint_') and event:assert full.all(),rec
    return records

def main():
    p=ROOT/'outputs/confidence_screen_plan.json';plan=json.loads(p.read_text())
    start=time.perf_counter();records=[]
    for n in plan['training_sizes']:
        for rep in range(plan['repetitions']):records.extend(trial(n,rep,plan))
        rows=[r for r in records if r['training_n']==n and r['horizon']==max(plan['horizons'])]
        print(json.dumps({'n':n,'last_horizon':[{k:r[k] for k in ['method','point_coverage','whole_family_coverage','mean_width']} for r in rows[-5:]]}),flush=True)
    summaries=[]
    for n in plan['training_sizes']:
      for H in plan['horizons']:
       for method in sorted(set(r['method'] for r in records)):
        rows=[r for r in records if r['training_n']==n and r['horizon']==H and r['method']==method]
        summaries.append({'training_n':n,'horizon':H,'method':method,**{key:float(np.mean([r[key] for r in rows])) for key in ['point_coverage','whole_family_coverage','all_episodes_family_covered','mean_width','oracle_mean_width','sign_certifiable_fraction','coefficient_confidence_event']}})
    result={'scope':'new affine Gaussian controlled-system transfer; family-specific finite-sample confidence experiment; known noise variance',
            'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'code_sha256':{f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in ['confidence_tube.py','run_confidence_screen.py']},
            'records':records,'summaries':summaries,'runtime_seconds':time.perf_counter()-start}
    (ROOT/'outputs/confidence_screen.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('CONFIDENCE_SCREEN_COMPLETE',result['runtime_seconds'],flush=True)
if __name__=='__main__':main()
