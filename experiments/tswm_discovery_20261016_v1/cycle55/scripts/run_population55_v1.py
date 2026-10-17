"""Prespecified analytic expectations and bounded numerical landscape checks."""
from common55 import *
import csv,time
import numpy as np
from scipy.optimize import minimize
from population55_v1 import population

STARTS=[(.1,.5,.5),(.8,3.,1.),(-.5,1.,1.),(.2,10.,.2),(.95,.1,2.),(-.9,30.,.5)]
BOUNDS=[(-.999,.999),(np.log(1e-4),np.log(1e4)),(np.log(1e-3),np.log(100.))]
def unpack(x):return float(x[0]),float(np.exp(x[1])),float(np.exp(x[2]))

def main():
    qualification=read(OUT/'QUALIFICATION55_V1.json');assert qualification['status']=='PASS_NATIVE_FINITE_BANK_AND_INDEPENDENT_POPULATION_QUADRATURE'
    assert qualification['math_script_sha256']==sha(ROOT/'scripts/population55_v1.py')
    assert qualification['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    pins=source_check();storage=read(PROJECT/'STORAGE_LOCATION_V1.json');artifact=pathlib.Path(storage['artifact_root'])/'cycle55/population_v1';assert not artifact.exists();artifact.mkdir(parents=True)
    rng=np.random.default_rng(5502);records=[];trials=[];started=time.perf_counter()
    for aa in [.5,.95]:
        for rr in [.03,.3,3.]:
            for T in [6,24]:
                for jitter in [0.,1e-5]:
                    case=f'a{aa}_r{rr}_T{T}_j{jitter}'
                    truth=population(aa,1/rr,1.,aa,rr,T,jitter)
                    gaps=[]
                    for _ in range(32):
                        a=float(rng.uniform(-.97,.97));lam=float(np.exp(rng.uniform(-3,4)));b=float(np.exp(rng.uniform(-2,2)))
                        value=population(a,lam,b,aa,rr,T,jitter);assert value['normalizer_exists'] and value['bound_gap']>=-1e-7
                        gaps.append(value['bound_gap'])
                    derivatives=[]
                    xtrue=np.array([aa,np.log(1/rr),0.]);step=1e-5
                    for k in range(3):
                        d=np.zeros(3);d[k]=step
                        plus=population(*unpack(xtrue+d),aa,rr,T,jitter)['loss'];minus=population(*unpack(xtrue-d),aa,rr,T,jitter)['loss'];derivatives.append(float((plus-minus)/(2*step)))
                    case_trials=[]
                    for index,start in enumerate(STARTS):
                        trace=[]
                        def objective(x):
                            value=population(*unpack(x),aa,rr,T,jitter)
                            return value['loss'] if value['normalizer_exists'] else 1e100
                        def callback(x):trace.append([*unpack(x),objective(x)])
                        result=minimize(objective,np.array([start[0],np.log(start[1]),np.log(start[2])]),method='L-BFGS-B',bounds=BOUNDS,callback=callback,options={'maxiter':300,'ftol':1e-12,'gtol':1e-7,'maxls':40})
                        a,lam,b=unpack(result.x);value=population(a,lam,b,aa,rr,T,jitter)
                        assert value['loss']>=truth['true_likelihood_optimum']-1e-6,(case,index,value)
                        p=artifact/(case+'_start'+str(index)+'.csv')
                        with p.open('x',encoding='utf-8',newline='') as f:
                            w=csv.writer(f);w.writerow(['a','lambda','b','loss']);w.writerows(trace)
                        row={'case':case,'a_star':aa,'r':rr,'T':T,'reporting_jitter':jitter,'start_index':index,'initial':list(start),'a':a,'lambda':lam,'b':b,'objective':value['loss'],'proper_likelihood_objective':value['proper_likelihood_loss'],'bound_gap':value['bound_gap'],'true_population_lower_bound':truth['true_likelihood_optimum'],'objective_excess_over_true_lower_bound':value['loss']-truth['true_likelihood_optimum'],'a_error':abs(a-aa),'reference_variance':value['reference_variance'],'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),'evaluations':int(result.nfev),'trace_storage_relative_path':'cycle55/population_v1/'+p.name,'trace_sha256':sha(p)}
                        trials.append(row);case_trials.append(row)
                    best=min(case_trials,key=lambda r:r['objective'])
                    record={'case':case,'a_star':aa,'r':rr,'T':T,'reporting_jitter':jitter,'true_point':truth,'true_point_central_derivative_a_loglambda_logb':derivatives,'minimum_random_bound_gap':min(gaps),'generic_starts':6,'starts_with_objective_excess_below_1e-6':sum(r['objective_excess_over_true_lower_bound']<1e-6 for r in case_trials),'best_numerical_trial':best}
                    records.append(record);print(json.dumps({'case':case,'true_point_gap':truth['bound_gap'],'best_a':best['a'],'a_star':aa,'best_objective_excess':best['objective_excess_over_true_lower_bound'],'starts_close_to_lower_bound':record['starts_with_objective_excess_below_1e-6']}),flush=True)
    assert len(records)==24 and len(trials)==144 and source_check()==pins
    summary={'utc':utc(),'status':'COMPLETE_PRESPECIFIED_SCALAR_POPULATION_OBJECTIVE_CONTROL','cases':records,'all_144_numerical_trials':trials,'runtime_seconds':time.perf_counter()-started,'native_training_updates':0,'formal_C54_model_or_data_reads':0,'true_point_used_as_optimizer_start':False,'algorithm':'Bounded L-BFGS-B on exact expectations, six fixed generic starts; not empirical training or a global-optimum proof','qualification_sha256':sha(OUT/'QUALIFICATION55_V1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'math_script_sha256':sha(ROOT/'scripts/population55_v1.py'),'run_script_sha256':sha(__file__),'source_files_sha256':pins,'accepted_final_directions':0,'overall_goal':'ACTIVE_UNACHIEVED'}
    dump(OUT/'population55_v1.json',summary)
    with (OUT/'population55_v1.csv').open('x',encoding='utf-8',newline='') as f:
        keys=['case','a_star','r','T','reporting_jitter','start_index','a','lambda','b','objective','proper_likelihood_objective','bound_gap','objective_excess_over_true_lower_bound','a_error','success','iterations','evaluations'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(trials)
    print(json.dumps({'status':summary['status'],'cases':24,'numerical_trials':144,'runtime_seconds':summary['runtime_seconds'],'accepted_final_directions':0}),flush=True)

if __name__=='__main__':main()
