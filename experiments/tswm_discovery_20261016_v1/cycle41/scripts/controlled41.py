"""Exact error-source and crossed-view controls; no proposed neural method."""
import sys,pathlib,json,hashlib,itertools,csv,datetime
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle29/scripts'));import runtime_profile
import numpy as np
from scipy.special import logsumexp
O=R/'outputs/controlled_v1';assert not O.exists();O.mkdir(parents=True)
native=R/'outputs/native_v1';assert json.loads((native/'qualification.json').read_text())['status']=='PASS'
z=np.load(native/'fixture.npz');A=z['phi'].astype(float);C=z['F'].astype(float);Q=.1**2*z['Sigma_d'].astype(float);Robs=.3**2*z['Sigma_p'].astype(float);d=len(A)
def burn(a,q):
 p=np.eye(len(a))
 for _ in range(100):p=a@p@a.T+q
 return (p+p.T)/2
def kfilter(y,mask,a,c,q,rr,p0):
 m=np.zeros(len(a));p=p0.copy();out=[]
 for t in range(y.shape[1]):
  if t:m=a@m;p=a@p@a.T+q
  ix=np.flatnonzero(mask[:,t]);ct=c[ix];r=rr[np.ix_(ix,ix)]
  if len(ix):
   sy=ct@p@ct.T+r;k=np.linalg.solve(sy,ct@p).T;m=m+k@(y[ix,t]-ct@m)
   left=np.eye(len(a))-k@ct;p=left@p@left.T+k@r@k.T
  out.append((m.copy(),(p+p.T)/2))
 return out
def gaussian_block(y,mask,a,c,q,rr,p0,t):
 ps=[p0]
 for _ in range(t):ps.append(a@ps[-1]@a.T+q)
 def zz(i,j):return np.linalg.matrix_power(a,i-j)@ps[j] if i>=j else zz(j,i).T
 obs=[(s,int(k)) for s in range(t+1) for k in np.flatnonzero(mask[:,s])];n=len(obs)
 if not n:return np.zeros(len(a)),ps[t]
 yy=np.array([y[k,s] for s,k in obs]);cov=np.empty((n,n));cross=np.empty((len(a),n))
 for i,(s,k) in enumerate(obs):
  cross[:,i]=zz(t,s)@c[k]
  for j,(v,l) in enumerate(obs):cov[i,j]=c[k]@zz(s,v)@c[l]+(rr[k,l] if s==v else 0.)
 mean=cross@np.linalg.solve(cov,yy);post=ps[t]-cross@np.linalg.solve(cov,cross.T)
 return mean,(post+post.T)/2
def future(m,p,a,c,q,rr,h):
 for _ in range(h):m=a@m;p=a@p@a.T+q
 return c@m,c@p@c.T+rr
p0=burn(A,Q);checks=[];raw={};rng=np.random.default_rng(202641)
u,_=np.linalg.qr(rng.normal(size=(d,d)));transform=u@np.diag(np.linspace(.7,1.7,d));inv=np.linalg.inv(transform)
ag=transform@A@inv;cg=C@inv;qg=transform@Q@transform.T;p0g=transform@p0@transform.T
for record in range(4):
 y=z['X_test_full'][record].astype(float);mask=z['mask_test'][record].astype(bool)
 f=kfilter(y,mask,A,C,Q,Robs,p0);g=kfilter(y,mask,ag,cg,qg,Robs,p0g)
 for t in [0,3,7,11]:
  m,p=gaussian_block(y,mask,A,C,Q,Robs,p0,t);errm=float(np.max(np.abs(m-f[t][0])));errp=float(np.max(np.abs(p-f[t][1])))
  assert max(errm,errp)<1e-9,(record,t,errm,errp)
  checks.append({'record':record,'t':t,'filter_vs_block_mean':errm,'filter_vs_block_cov':errp})
 for h in [1,4,8]:
  m,p=future(*f[23],A,C,Q,Robs,h);mg,pg=future(*g[23],ag,cg,qg,Robs,h)
  err=max(float(np.abs(m-mg).max()),float(np.abs(p-pg).max()));assert err<1e-9,err
  checks.append({'record':record,'h':h,'observable_gauge_max_error':err});raw[f'oracle_r{record}_h{h}_mean']=m;raw[f'oracle_r{record}_h{h}_cov']=p
# Analytic frozen-transition compensation, with an explicit ordinary repair.
n=4;eye=np.eye(n);j=np.ones((n,n))/n;ptrue=.8*eye+.2*j;qbad=.9*eye+.1*j;bcomp=(8/9)*eye+(1/9)*j
comp=[]
for h in [1,4,8]:
 truth=np.linalg.matrix_power(ptrue,h)
 for name,enc,trans in [('wrong_transition_exact_belief',eye,qbad),('wrong_transition_compensating_state',bcomp,qbad),('ordinary_correct_transition_exact_belief',eye,ptrue)]:
  pred=enc@np.linalg.matrix_power(trans,h);regret=float(np.sum(truth*np.log(truth/pred),axis=1).mean());excess=float(np.sum((truth-pred)**2,axis=1).mean())
  comp.append({'h':h,'case':name,'KL_regret':regret,'Brier_excess':excess,'state_coordinate_l2_error':float(np.sum((enc-eye)**2,axis=1).mean())})
  raw[f'comp_{name}_h{h}']=pred
assert abs(next(x['Brier_excess'] for x in comp if x['h']==1 and x['case']=='wrong_transition_compensating_state'))<1e-25
# Crossed-view exact posterior over every possible observation alignment.
p1=ptrue;p2=np.array([[.70,.20,.08,.02],[.05,.65,.20,.10],[.10,.15,.60,.15],[.20,.10,.10,.60]])
perms=np.array(list(itertools.permutations(range(4))));inverse=np.argsort(perms,axis=1);pviews=np.array([p2[v][:,v] for v in inverse]);p1views=np.array([p1[v][:,v] for v in inverse])
train_diff=float(np.abs(p1views-p1).max());assert train_diff==0
heldout_diff=float(np.abs(pviews-p2).max());assert heldout_diff>.01
def chain(p,length,rng):
 a=np.empty(length,dtype=int);a[0]=rng.integers(4)
 for t in range(1,length):a[t]=rng.choice(4,p=p[a[t-1]])
 return a
cross=[];rng=np.random.default_rng(20264141)
for repeat in range(128):
 which=int(rng.integers(24));mapping=perms[which];anchors=chain(p1,16,rng);x=chain(p2,32,rng);obs=mapping[x]
 raw[f'cross_r{repeat}_anchor_states']=anchors;raw[f'cross_r{repeat}_prefix_B']=obs;raw[f'cross_r{repeat}_mapping']=mapping
 for count in [0,1,2,4,8,16]:
  permitted=np.all(perms[:,anchors[:count]]==mapping[anchors[:count]],axis=1) if count else np.ones(24,dtype=bool)
  assert permitted[which]
  for hist in [1,8,32]:
   logw=np.full(24,-np.inf);logw[permitted]=-np.log(4)
   for t in range(1,hist):logw+=np.log(pviews[:,obs[t-1],obs[t]])
   w=np.exp(logw-logsumexp(logw));last=int(obs[hist-1]);pred=np.einsum('i,ij->j',w,pviews[:,last,:]);truth=pviews[which,last,:]
   excess=float(np.sum((pred-truth)**2));kl=float(np.sum(truth*np.log(truth/pred)));entropy=float(-np.sum(w[w>0]*np.log(w[w>0])))
   cross.append({'repeat':repeat,'anchors':count,'history':hist,'anchor_state_coverage':len(np.unique(anchors[:count])),'possible_alignments_after_anchors':int(permitted.sum()),'alignment_entropy':entropy,'known_alignment_Brier_excess':excess,'known_alignment_KL_regret':kl,'true_alignment_posterior':float(w[which])})
   if permitted.sum()==1:assert excess<1e-25 and kl<1e-12
for name,rows in [('compensation.csv',comp),('crossed_view.csv',cross),('oracle_checks.csv',checks)]:
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (O/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
np.savez_compressed(O/'raw_controls.npz',**raw,ptrue=ptrue,qbad=qbad,bcomp=bcomp,p1=p1,p2=p2,permutations=perms,view_transitions=pviews)
means=[]
for count in [0,1,2,4,8,16]:
 for hist in [1,8,32]:
  sub=[r for r in cross if r['anchors']==count and r['history']==hist]
  means.append({'anchors':count,'history':hist,**{k:float(np.mean([r[k] for r in sub])) for k in ['alignment_entropy','known_alignment_Brier_excess','known_alignment_KL_regret','true_alignment_posterior']},'unique_alignment_fraction':float(np.mean([r['possible_alignments_after_anchors']==1 for r in sub]))})
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','protocol_sha256':hashlib.sha256((R/'CONTROLLED_PLAN.md').read_bytes()).hexdigest(),'oracle_checks':checks,'compensation':comp,'cross_train_alignment_law_max_difference':train_diff,'cross_heldout_law_max_difference':heldout_diff,'cross_repetitions':128,'cross_rows':len(cross),'cross_summary':means,'raw_sha256':hashlib.sha256((O/'raw_controls.npz').read_bytes()).hexdigest(),'neural_fits':0,'new_method':False,'test_scored':False,'interpretation':'Privileged exact controls; alignment uncertainty is an information cost, not proof an algorithm fails or a new neural mechanism is required.'}
(O/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['oracle_checks','cross_summary']},indent=2))
