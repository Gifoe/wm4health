"""Falsify a proposed computational gap using ordinary Gaussian/SDP controls.

Restricted global O(d) Gaussian noise-alignment family; not nonparametric bounds.
No training. All numerical SDP bounds are explicitly non-validated numerics.
"""
import ctypes,datetime,json,os,pathlib,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
if os.name=='nt':
 fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
import numpy as np
import scipy.linalg as la
import scipy.optimize as so
import cvxpy as cp
ROOT=pathlib.Path(__file__).resolve().parents[1]

def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def polar(x):u,_,v=np.linalg.svd(x);return u@v
def draw_q(rng,d):return polar(rng.normal(size=(d,d)))
def case(d,T,seed):
 rng=np.random.default_rng(seed+100*d);A=[];L=[];b=[]
 for a in range(2):
  q=draw_q(rng,d);A.append(q@np.diag(np.linspace(.65,.96,d))@q.T)
  L.append(np.diag(np.linspace(.15,.35,d))@(np.eye(d)+.1*rng.normal(size=(d,d))))
  b.append(rng.normal(size=d)*.08*(2*a-1))
 C=rng.normal(size=(max(1,d//2),d))/np.sqrt(d);V=np.eye(len(C))*.15**2
 af=(np.arange(T)//7%2).astype(int);ac=(np.arange(T)//11%2).astype(int)
 h=rng.normal(size=d);h/=np.linalg.norm(h);tau=.8
 x=rng.normal(size=d);initial=x.copy();ys=[]
 for t in range(T):
  x=A[af[t]]@x+b[af[t]]+L[af[t]]@rng.normal(size=d)
  ys.append(C@x+la.cholesky(V,lower=True)@rng.normal(size=len(C)))
 return dict(d=d,T=T,seed=seed,A=A,L=L,b=b,C=C,V=V,af=af,ac=ac,h=h,tau=tau,ys=np.array(ys),initial=initial)

def coefficients(c):
 d,T=c['d'],c['T'];k=1+d*d;f=np.zeros((T,k,d));p=c['h'].copy();constant=-c['tau']
 for t in range(T-1,-1,-1):
  af,ac=c['af'][t],c['ac'][t];w=p@c['L'][ac];constant+=p@c['b'][ac]
  if af==ac:f[t,0]=w
  else:
   for i in range(d):
    for j in range(d):f[t,1+i*d+j,j if af==0 else i]=w[i if af==0 else j]
  p=p@c['A'][ac]
 init=np.zeros((k,d));init[0]=p
 intercept=np.zeros(k);intercept[0]=constant
 return init,f,intercept

def update(m,P,H,V,y):
 PH=P@H.T;S=H@PH+V;K=la.solve(S,PH.T,assume_a='pos').T
 m=m+K@(y-H@m);P=P-K@PH.T
 return m,(P+P.T)*.5

def feature_filter(c):
 d=c['d'];init,f,intercept=coefficients(c);k=len(init);n=d+k
 m=np.r_[np.zeros(d),intercept];B0=np.vstack([np.eye(d),init]);P=B0@B0.T
 H=np.c_[c['C'],np.zeros((len(c['C']),k))]
 for t,y in enumerate(c['ys']):
  a=c['af'][t];F=la.block_diag(c['A'][a],np.eye(k));G=np.vstack([c['L'][a],f[t]])
  m=F@m+np.r_[c['b'][a],np.zeros(k)];P=F@P@F.T+G@G.T
  m,P=update(m,P,H,c['V'],y)
 return m[d:],P[d:,d:]

def feature_dense(c):
 """Independent batch Gaussian conditioning check, not the proposed fast path."""
 d,T=c['d'],c['T'];init,f,intercept=coefficients(c);n=(T+1)*d
 feature=np.c_[init,f.transpose(1,0,2).reshape(len(init),-1)]
 B=np.zeros((d,n));B[:,:d]=np.eye(d);m=np.zeros(d);rows=[];mu=[]
 for t in range(T):
  a=c['af'][t];B=c['A'][a]@B;B[:,(t+1)*d:(t+2)*d]+=c['L'][a];m=c['A'][a]@m+c['b'][a]
  rows.append(c['C']@B);mu.append(c['C']@m)
 H=np.concatenate(rows);S=H@H.T+np.kron(np.eye(T),c['V']);cross=feature@H.T
 gain=la.solve(S,cross.T,assume_a='pos').T
 return intercept+gain@(c['ys'].ravel()-np.concatenate(mu)),feature@feature.T-gain@cross.T

def twin_filter(c,R):
 d=c['d'];m=np.zeros(2*d);P=np.block([[np.eye(d),np.eye(d)],[np.eye(d),np.eye(d)]])
 H=np.c_[c['C'],np.zeros_like(c['C'])]
 for t,y in enumerate(c['ys']):
  af,ac=c['af'][t],c['ac'][t]
  M=np.eye(d) if af==ac else R if af==0 else R.T
  F=la.block_diag(c['A'][af],c['A'][ac]);G=np.vstack([c['L'][af],c['L'][ac]@M])
  m=F@m+np.r_[c['b'][af],c['b'][ac]];P=F@P@F.T+G@G.T;m,P=update(m,P,H,c['V'],y)
 qmean=c['h']@m[d:]-c['tau'];qvar=c['h']@P[d:,d:]@c['h']
 return qmean,qvar,qmean*qmean+qvar

def value(M,R):z=np.r_[1.,R.ravel()];return float(z@M@z)
def local_extreme(M,R,sign):
 d=len(R);old=value(M,R);steps=0
 for steps in range(800):
  G=-sign*2*(M[1:,1:]@R.ravel()+M[1:,0]).reshape(d,d)
  tangent=G-R@((R.T@G+G.T@R)*.5)
  norm=np.linalg.norm(tangent)
  if norm<1e-9:break
  step=min(1.,1/max(1.,norm));accepted=False
  for _ in range(25):
   candidate=polar(R+step*tangent);new=value(M,candidate)
   if sign*(old-new)>1e-4*step*norm**2:accepted=True;break
   step*=.5
  if not accepted:break
  R=candidate
  if abs(new-old)<1e-12:old=new;break
  old=new
 return old,R,steps+1

def sdp(M,d,sign):
 n=len(M);Y=cp.Variable((n,n),symmetric=True);constraints=[Y>>0,Y[0,0]==1]
 for i in range(d):
  for j in range(i,d):
   constraints.append(sum(Y[1+k*d+i,1+k*d+j] for k in range(d))==float(i==j))
   constraints.append(sum(Y[1+i*d+k,1+j*d+k] for k in range(d))==float(i==j))
 problem=cp.Problem(cp.Minimize(sign*cp.sum(cp.multiply(M,Y))),constraints)
 begin=time.perf_counter();problem.solve(solver='CLARABEL',tol_gap_abs=1e-8,tol_feas=1e-8,tol_gap_rel=1e-8,max_iter=300)
 elapsed=time.perf_counter()-begin
 assert problem.status in ['optimal','optimal_inaccurate'],problem.status
 X=(Y.value+Y.value.T)*.5;residual=max(float(np.max(np.abs(q.violation()))) for q in constraints)
 return sign*problem.value,X,{'status':problem.status,'seconds':elapsed,'max_constraint_violation':residual,'min_eigenvalue':float(np.linalg.eigvalsh(X).min()),'solver':'CLARABEL','formal_certificate':False}

def exact_o2(M):
 """Degree-two trigonometric polynomial on each component; all critical roots."""
 extrema=[]
 for parity in [-1,1]:
  def R(t):return np.array([[np.cos(t),-parity*np.sin(t)],[np.sin(t),parity*np.cos(t)]])
  theta=np.arange(8)*np.pi/4;A=np.array([[1,np.cos(t),np.sin(t),np.cos(2*t),np.sin(2*t)] for t in theta]);coef=np.linalg.lstsq(A,np.array([value(M,R(t)) for t in theta]),rcond=None)[0]
  # tan(t/2)=u. Multiplying f'(t) by (1+u^2)^2 yields a quartic.
  _,a,b,c,e=coef;poly=np.array([b+2*e,-2*a-8*c,-12*e,-2*a+8*c,-b+2*e])
  roots=np.polynomial.polynomial.polyroots(poly)
  candidates=[0.,np.pi]+[2*np.arctan(z.real) for z in roots if abs(z.imag)<1e-7]
  extrema.extend(value(M,R(t)) for t in candidates)
 return min(extrema),max(extrema)

def main():
 out=ROOT/'outputs'/'continuous_counterfactual';out.mkdir(parents=True,exist_ok=True)
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'d':[2,3,5],'T':[64,256],'seed':[61,62,63],'scope':'New controlled computation, known linear Gaussian system, partially observed factual path, fixed counterfactual open-loop plan, shared global R in O(d). Restricted-family risk bounds only.','controls':['independent dense Gaussian conditioning','direct twin-state Kalman filter','generic Shor orthogonal quadratic SDP with row and column constraints','feasible multistart polar Riemannian optimisation and SDP rounding','d2 all critical points in both determinant components'],'stop_gate':'All generic SDP feasible/relaxation gaps <=1% of max(1,abs(extremum)) stops custom solver in this regime. Gap alone does not establish a research idea.','inference':'Numerical solvers only, no formal validated certificate, no statistical uncertainty in fitted kernels, no native-world-model claim.'}
 if (out/'results.json').exists():raise RuntimeError('Completed experiment already exists')
 if (out/'plan.json').exists():
  original=json.loads((out/'plan.json').read_text());assert all(original[k]==plan[k] for k in ['d','T','seed','scope','stop_gate'])
 else:dump(out/'plan.json',plan)
 rows=[]
 for seed in plan['seed']:
  for d in plan['d']:
   for T in plan['T']:
    c=case(d,T,seed);start=time.perf_counter();mean,cov=feature_filter(c);seconds=time.perf_counter()-start
    start=time.perf_counter();md,cd=feature_dense(c);dense_sec=time.perf_counter()-start
    dense_error=max(np.max(np.abs(mean-md)),np.max(np.abs(cov-cd)));assert dense_error<1e-8,dense_error
    M=cov+np.outer(mean,mean);rng=np.random.default_rng(seed+1000*d+T);max_error=0.;twintime=0.
    for _ in range(20):
     R=draw_q(rng,d);start=time.perf_counter();tm,tv,tr=twin_filter(c,R);twintime+=time.perf_counter()-start
     z=np.r_[1.,R.ravel()];max_error=max(max_error,abs(tm-z@mean),abs(tv-z@cov@z),abs(tr-value(M,R)))
    assert max_error<1e-8,max_error
    extrema=[];witnesses=[]
    for sign in [1,-1]:
     bound,X,solver=sdp(M,d,sign);starts=[np.eye(d),-np.eye(d)]+[draw_q(rng,d) for _ in range(8)]
     ev,U=np.linalg.eigh(X);sqrt=U*np.sqrt(np.maximum(ev,0.))[None,:]
     for _ in range(12):
      z=sqrt@rng.normal(size=len(M));r=z[1:]*(1 if z[0]>=0 else -1);starts.append(polar(r.reshape(d,d)))
     values=[]
     for R in starts:values.append(local_extreme(M,R,sign))
     best,R,steps=min(values,key=lambda v:sign*v[0]);witnesses.append(R)
     gap=sign*(best-bound);assert gap>-1e-6,(gap,best,bound)
     extrema.append({'kind':'min' if sign==1 else 'max','feasible':best,'SDP_bound':float(bound),'absolute_gap':max(0.,gap),'relative_gap':max(0.,gap)/max(1.,abs(best)),'solver':solver,'orthogonality_residual':float(np.linalg.norm(R.T@R-np.eye(d))),'determinant':float(np.linalg.det(R))})
    exact=exact_o2(M) if d==2 else None
    if exact:assert max(abs(exact[i]-extrema[i]['feasible']) for i in range(2))<1e-6
    row={'seed':seed,'d':d,'T':T,'query_feature_dimension':1+d*d,'augmented_filter_dimension':d+1+d*d,'feature_filter_seconds':seconds,'dense_conditioning_seconds':dense_sec,'dense_feature_max_error':float(dense_error),'twin_filter_seconds_20':twintime,'direct_twin_max_error_20':float(max_error),'extrema':extrema,'exact_O2':exact,'constant_gauge_family_only':True}
    rows.append(row);np.savez_compressed(out/f'case_d{d}_T{T}_s{seed}.npz',mean=mean,cov=cov,M=M,Rmin=witnesses[0],Rmax=witnesses[1],A=np.array(c['A']),L=np.array(c['L']),b=np.array(c['b']),C=c['C'],V=c['V'],ys=c['ys'],af=c['af'],ac=c['ac'],h=c['h'],tau=c['tau'])
    dump(out/'partial_results.json',rows);dump(out/'status.json',{'state':'running','completed':len(rows),'total':18});print(json.dumps(row),flush=True)
 gate=all(e['relative_gap']<=.01 for row in rows for e in row['extrema'])
 result={'rows':rows,'all_generic_SDP_gaps_below_1pct':gate,'decision':'STOP_CUSTOM_SOLVER_IN_THIS_REGIME' if gate else 'COMPUTATIONAL_RESIDUAL_REQUIRES_STRONGER_EXISTING_CONTROLS','research_idea_accepted':False,'objective_achieved':False,'environment':{'numpy':np.__version__,'scipy':__import__('scipy').__version__,'cvxpy':cp.__version__,'solvers':cp.installed_solvers()}}
 dump(out/'results.json',result);dump(out/'status.json',{'state':'complete','completed':18,'decision':result['decision'],'objective_achieved':False})
 print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
