"""Pinned real metric + exact controlled dynamics + explicit information sets.

This is a falsification of a metric interpretation, not a new model method.
One-sided versus balanced interventions prevent a convenient gaming example
from being mistaken for a general benchmark failure.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MKL_NUM_THREADS','4')
import hashlib,json,pathlib,sys,time
import numpy as np
import torch
from scipy.special import ndtr
ROOT=pathlib.Path(__file__).resolve().parents[1];SRC=ROOT.parent/'upstream/dotime'
sys.path.insert(0,str(SRC/'src'))
from dotime.evaluation import direction_accuracy
from dotime.benchmarks import Episode
from dotime.interventions import InterventionSpec,InterventionType
from dotime.reference.pfn import episode_to_batch_interp
from dotime.baselines import VAROLSBaseline
torch.set_num_threads(4)

def draw(n,seed,H,sigma,mode,plan):
    rng=np.random.default_rng(seed)
    sign=rng.choice([-1.,1.],size=n);action=np.ones(n) if mode=='one_sided' else rng.choice([-1.,1.],size=n)
    x0=rng.normal(size=n);a=plan['state_coefficient'];b=plan['action_coefficient']
    mu=a**H*x0;unit=sign*b*(1-a**H)/(1-a);tau=unit*action
    noise=rng.normal(scale=sigma,size=(n,H))@a**np.arange(H-1,-1,-1)
    factual=mu+noise;counter=factual+tau
    return {'mu':mu,'unit':unit,'tau':tau,'factual':factual,'counter':counter}

def predict(d,beta,bias):
    # The context offset is independent of the action. Counterfactual/factual
    # model contrast is exactly beta*tau, including under balanced +/- doses.
    common=d['mu']+bias*d['unit']
    return common+beta*d['tau'],common

def score(d,beta,bias):
    cf,fa=predict(d,beta,bias)
    metric=lambda p,t:direction_accuracy(torch.tensor(p,dtype=torch.float32),torch.tensor(t,dtype=torch.float32))['accuracy']
    return {'factual_anchored_direction_accuracy':metric(cf-d['factual'],d['tau']),
            'paired_model_response_direction_accuracy':metric(cf-fa,d['tau']),
            'causal_response_nmse':float(np.mean((cf-fa-d['tau'])**2)/np.mean(d['tau']**2)),
            'interventional_level_mse':float(np.mean((cf-d['counter'])**2))}

def native_mask_contract():
    rng=np.random.default_rng(551016);T=80;onset=48
    x=torch.tensor(rng.normal(size=(T,2)),dtype=torch.float32)
    spec=InterventionSpec([0],list(range(onset,T)),InterventionType.HARD,1.)
    def episode(obs):
        return Episode(x_obs=obs,x_int=obs.clone(),intervention=spec,y_true=torch.tensor([.5]),
            query_target=torch.tensor([1]),query_time=torch.tensor([float(T-1)]),metadata={'query_time_idx':[T-1]})
    y=x.clone();y[onset+1:,1]+=10.
    ea,eb=episode(x),episode(y)
    ba=episode_to_batch_interp(ea,8,'cpu');bb=episode_to_batch_interp(eb,8,'cpu')
    input_keys=[k for k in ba if k not in ['Y_true','Y_true_norm']]
    diffs={k:float((ba[k].float()-bb[k].float()).abs().max()) for k in input_keys}
    assert max(diffs.values())==0.
    var=VAROLSBaseline();va=var.predict(ea);vb=var.predict(eb)
    return {'operation':'change only post-onset factual outcome suffix while holding prefix/onset treatment/query fixed',
            'reference_pfn_visible_batch_max_difference':max(diffs.values()),'compared_keys':input_keys,
            'native_var_prediction_difference':float((va-vb).abs().max()),
            'interpretation':'Native baseline interfaces use different factual information sets. This alone does not prove which is intended or invalidate every published result.'}

def main():
    pp=ROOT/'outputs/direction_audit_plan.json';plan=json.loads(pp.read_text());start=time.perf_counter();records=[]
    for H in plan['horizons']:
      for sigma in plan['process_noise_scales']:
       for mode in plan['action_modes']:
        for seed in plan['training_seeds']:
            val=draw(plan['validation_episodes'],seed+100000,H,sigma,mode,plan)
            test=draw(plan['confirmation_episodes'],seed+200000,H,sigma,mode,plan)
            for name,beta in [('bayes_level_oracle',1.),('same_response_common_bias',1.),('reverse_response',-1.),('intervention_blind',0.)]:
                if name=='bayes_level_oracle':bias=0.
                elif name=='same_response_common_bias':bias=2. # fixed calibration-invariance stress, not chosen on test
                else:
                    scores=[score(val,beta,c)['factual_anchored_direction_accuracy'] for c in plan['common_bias_grid']]
                    bias=plan['common_bias_grid'][int(np.argmax(scores))]
                r={'horizon':H,'process_sigma':sigma,'action_mode':mode,'seed':seed,'predictor':name,'response_slope':beta,'validation_selected_bias':bias,**score(test,beta,bias)}
                # Analytic population direction accuracy of Bayes level mean.
                a=plan['state_coefficient'];sd=sigma*np.sqrt((1-a**(2*H))/(1-a*a))
                effect=abs(plan['action_coefficient']*(1-a**H)/(1-a))
                r['bayes_level_anchor_score_population']=float(ndtr(effect/sd))
                records.append(r)
    out={'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
         'native_code_sha256':{f:hashlib.sha256((SRC/f).read_bytes()).hexdigest() for f in ['src/dotime/evaluation.py','src/dotime/reference/pfn.py','src/dotime/baselines.py']},
         'records':records,'native_mask_contract':native_mask_contract(),
         'scope':'exact new AR(1) diagnostic with native DoTime metric/batch/baseline, not official dataset scores',
         'runtime_seconds':time.perf_counter()-start}
    (ROOT/'outputs/direction_information_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out['native_mask_contract'],indent=2));print('DIRECTION_AUDIT_COMPLETE',out['runtime_seconds'])
if __name__=='__main__':main()
