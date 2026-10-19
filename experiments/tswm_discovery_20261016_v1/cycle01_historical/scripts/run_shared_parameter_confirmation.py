"""Fresh-cohort comparison, locked before any new test outcome is observed."""
import hashlib,json,pathlib,time
import numpy as np
from scipy.stats import chi2
from confidence_tube import mean,simulate,family,regression_fit,tubes,shared_parameter_tube
from run_confidence_screen import TRUE,SIGMA
ROOT=pathlib.Path(__file__).resolve().parents[1]

def main():
    pp=ROOT/'outputs/shared_parameter_confirmation_plan.json';plan=json.loads(pp.read_text())
    start=time.perf_counter();records=[]
    for n in plan['training_sizes']:
      for rep in range(plan['repetitions']):
        rng=np.random.default_rng(plan['training_seed_base']+100000*n+rep)
        x=rng.normal(size=(n,2));a=rng.uniform(-1,1,(n,1));y=mean(TRUE,x,a)+rng.normal(scale=SIGMA,size=(n,2))
        w,P=regression_fit(x,a,y)
        conf_stat=np.sum((w-TRUE)*np.linalg.solve(P,w-TRUE))/SIGMA**2
        event=conf_stat<=chi2.ppf(1-plan['alpha'],w.size)
        rg=np.random.default_rng(plan['confirmation_seed_base']+rep)
        N=plan['episodes_per_repetition'];Hmax=max(plan['horizons'])
        x0=rg.normal(scale=.4,size=(N,2));eps=rg.normal(scale=SIGMA,size=(N,Hmax,2))
        factual=simulate(TRUE,x0,eps,0.)
        for H in plan['horizons']:
          f=factual[:,:H+1];true_lo,true_hi=family(TRUE,f);hat_lo,hat_hi=family(w,f)
          basic=tubes(w,P,SIGMA,f,plan['alpha'],plan['angle_grid'])
          shared=shared_parameter_tube(w,P,SIGMA,f,plan['alpha'],plan['angle_grid'])
          residual=f[:,1:]-mean(w,f[:,:-1],0.)
          point=simulate(w,x0,residual,1.,0.)[:,-1,0]-f[:,-1,0]
          truth=simulate(TRUE,x0,eps[:,:H],1.,plan['true_angle'])[:,-1,0]-f[:,-1,0]
          methods={'gauge_only':(hat_lo,hat_hi),
             'statistical_only_fixed_common_noise':(point-basic['fixed_common_noise_radius'],point+basic['fixed_common_noise_radius']),
             'joint_paired_tube':(hat_lo-basic['paired_radius'],hat_hi+basic['paired_radius']),
             'joint_separate_tube':(hat_lo-basic['separate_radius'],hat_hi+basic['separate_radius']),
             'joint_shared_parameter_tube':(shared['lo'],shared['hi']),
             'linearization_without_remainder_ABLATION':(shared['linear_only_lo'],shared['linear_only_hi']),
             'oracle_known_kernel_family':(true_lo,true_hi)}
          for method,(lo,hi) in methods.items():
            full=(lo<=true_lo+1e-10)&(hi>=true_hi-1e-10)
            cov=(lo<=truth+1e-10)&(hi>=truth-1e-10)
            signcert=(lo>0)|(hi<0)
            # Wrong signed claims measured against the WHOLE true admissible family.
            wrong=((lo>0)&(true_lo<=0))|((hi<0)&(true_hi>=0))
            row={'training_n':n,'rep':rep,'horizon':H,'method':method,
                 'coefficient_confidence_event':bool(event),
                 'point_coverage':float(cov.mean()),'whole_family_coverage':float(full.mean()),
                 'all_episodes_family_covered':bool(full.all()),
                 'mean_width':float((hi-lo).mean()),'oracle_mean_width':float((true_hi-true_lo).mean()),
                 'sign_certifiable_fraction':float(signcert.mean()),
                 'wrong_signed_claim_fraction':float(wrong.mean()),
                 'mean_quadratic_remainder':float(shared['quadratic_remainder'].mean())}
            records.append(row)
            if method.startswith('joint_') and event:assert full.all(),row
      print('COMPLETED_N',n,flush=True)
    summaries=[]
    keys=['point_coverage','whole_family_coverage','all_episodes_family_covered','mean_width','oracle_mean_width','sign_certifiable_fraction','wrong_signed_claim_fraction','mean_quadratic_remainder','coefficient_confidence_event']
    for n in plan['training_sizes']:
      for H in plan['horizons']:
       for method in sorted(set(r['method'] for r in records)):
        rows=[r for r in records if r['training_n']==n and r['horizon']==H and r['method']==method]
        summaries.append({'training_n':n,'horizon':H,'method':method,**{k:float(np.mean([r[k] for r in rows])) for k in keys}})
    result={'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),
            'code_sha256':{f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in ['confidence_tube.py','run_shared_parameter_confirmation.py']},
            'scope':'affine fully observed system, known isotropic noise, one global rotation coupling family; simultaneous coefficient confidence set',
            'records':records,'summaries':summaries,'runtime_seconds':time.perf_counter()-start}
    (ROOT/'outputs/shared_parameter_confirmation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SHARED_PARAMETER_CONFIRMATION_COMPLETE',result['runtime_seconds'],flush=True)
if __name__=='__main__':main()
