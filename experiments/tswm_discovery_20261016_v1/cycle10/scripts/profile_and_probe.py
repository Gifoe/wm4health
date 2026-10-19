"""Existing covariance fit and randomized intervention controls for the witness."""
import runtime_profile
import contextlib,json,pathlib,sys,time,traceback
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
from scipy.stats import t as student_t
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/profile_probe';OUT.mkdir(parents=True,exist_ok=True)
from lag_mask_equivalence_audit import Tee,spectral
def matrices(p,b):
 rU,rA,rY,a,c=p;B=np.array([[0,0,0],[a,0,0],[c,b,0.]])
 L=la.inv(np.eye(3)-B);D=np.diag([rU,rA,rY]);F=L@D;Q=.09*L@L.T;return F,Q
def covariances(F,Q,lags=16):
 P=la.solve_discrete_lyapunov(F,Q);C=np.eye(3)[1:];power=np.eye(3);rows=[]
 for k in range(lags+1):rows.append(C@power@P@C.T);power=power@F
 return np.asarray(rows)
def main():
 base=json.loads((ROOT/'outputs/lag_mask/results.json').read_text(encoding='utf-8'));config={'profile_fixed_b':[.6,-.4],'profile_starts':12,'profile_fit_lags':16,'profile_max_nfev':600,'profile_start_seed':1031,'profile_bounds':{'AR':[-.9,.9],'a_c':[-3.,3.]},'probe_N':[8,32,128],'probe_seeds':list(range(2024,2064)),'scope':'Ordinary covariance-matching/local search and extra-information randomized control, not a new method.'}
 (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8');profiles=[];probes=[];start=time.time()
 C=np.eye(3)[1:];S=np.asarray(base['construction']['hidden_state_transform_S']);B=np.asarray(base['construction']['B']);B2=np.asarray(base['construction']['B_alternative']);L=la.inv(np.eye(3)-B);L2=la.inv(np.eye(3)-B2)
 for matrixcase in base['matrix_checks']:
  scale=matrixcase['scale'];D=np.asarray(matrixcase['lag_original']);D2=np.asarray(matrixcase['lag_alternative']);F=L@D;Q=.09*L@L.T;target=covariances(F,Q)
  rng=np.random.default_rng(1031);starts=[np.r_[np.diag(D),1.,-1.]]+[np.r_[rng.uniform(-.75,.75,3),rng.uniform(-2.,2.,2)] for _ in range(11)]
  for b in config['profile_fixed_b']:
   fits=[]
   for x0 in starts:
    def residual(p):
     ff,qq=matrices(p,b);return ((covariances(ff,qq)-target)/.09).ravel()
    fit=least_squares(residual,x0,bounds=([-.9]*3+[-3.]*2,[.9]*3+[3.]*2),max_nfev=600,ftol=1e-11,xtol=1e-11,gtol=1e-11)
    fits.append({'parameters':fit.x.tolist(),'residual_l2':float(la.norm(fit.fun)),'nfev':fit.nfev,'success':bool(fit.success)})
   best=min(fits,key=lambda r:r['residual_l2']);ff,qq=matrices(best['parameters'],b)
   validation=max(float(la.norm(spectral(ff,qq,w)-spectral(F,Q,w))) for w in np.linspace(0,np.pi,257))
   row={'scale':scale,'fixed_b':b,'best':best,'all_starts':fits,'independent_spectral_max_error':validation,'interpretation':'A nonzero best residual is not a global impossibility certificate. A near-zero covariance/spectrum match is a numerical equivalence witness only.'};profiles.append(row);print('profile',json.dumps({k:v for k,v in row.items() if k not in ['all_starts','interpretation']}),flush=True)
  for world,(BB,DD,LL) in enumerate([(B,D,L),(B2,D2,L2)]):
   FF=LL@DD;QQ=.09*LL@LL.T;P=la.solve_discrete_lyapunov(FF,QQ)
   for n in config['probe_N']:
    for seed in config['probe_seeds']:
     rng=np.random.default_rng(seed);previous=rng.multivariate_normal(np.zeros(3),P,size=n);noise=rng.normal(size=(n,3))*.3;action=rng.choice([-1.,1.],size=n)
     # Exact source-linear one-step hard do(A); currentU precedes A and Y.
     hidden=previous@DD[0]+noise[:,0]
     outcome=BB[2,1]*action+BB[2,0]*hidden+previous@DD[2]+noise[:,2]
     design=np.column_stack([np.ones(n),action]);coef=la.lstsq(design,outcome)[0];resid=outcome-design@coef;inv=la.inv(design.T@design);se=float(np.sqrt((resid@resid)/(n-2)*inv[1,1]));crit=float(student_t.ppf(.975,n-2));low=coef[1]-crit*se;high=coef[1]+crit*se;truth=float(BB[2,1])
     probes.append({'scale':scale,'world':world,'n':n,'seed':seed,'slope':float(coef[1]),'truth':truth,'low95':float(low),'high95':float(high),'sign_correct':bool(coef[1]*truth>0),'covers_truth':bool(low<=truth<=high),'interval_excludes_zero':bool(low*high>0)})
 summary=[]
 for world in [0,1]:
  for n in config['probe_N']:
   vals=[r for r in probes if r['world']==world and r['n']==n];summary.append({'world':world,'n':n,'runs':len(vals),'sign_correct_fraction':float(np.mean([r['sign_correct'] for r in vals])),'coverage95':float(np.mean([r['covers_truth'] for r in vals])),'excludes_zero_fraction':float(np.mean([r['interval_excludes_zero'] for r in vals])),'slope_rmse':float(np.sqrt(np.mean([(r['slope']-r['truth'])**2 for r in vals])))})
 result={'profiles':profiles,'probe_runs':probes,'probe_summary':summary,'elapsed_seconds':time.time()-start,'decision':'Additional randomized information is an existing causal repair. Passive equivalence itself cannot be fixed by a stronger predictor.'}
 (OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('PROBE_SUMMARY',json.dumps(summary),flush=True);print('COMPLETE',result['elapsed_seconds'],flush=True)
if __name__=='__main__':
 with (OUT/'stdout.log').open('a',encoding='utf-8') as stdout,(OUT/'stderr.log').open('a',encoding='utf-8') as stderr,contextlib.redirect_stdout(Tee(sys.stdout,stdout)),contextlib.redirect_stderr(Tee(sys.stderr,stderr)):
  try:main()
  except BaseException:traceback.print_exc();raise
