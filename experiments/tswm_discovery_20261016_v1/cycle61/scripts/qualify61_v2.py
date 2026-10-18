"""Independent dense probability recursion vs native causal likelihood and runtime."""
from native61_v2 import *
import numpy as np,time,platform

def dense_matrix(pdur,A):
 K,D=pdur.shape;S=np.flip(np.cumsum(np.flip(pdur,axis=-1),axis=-1),axis=-1);P=np.zeros((K*D,K*D))
 for z in range(K):
  for a in range(D):
   i=z*D+a;hazard=pdur[z,a]/S[z,a]
   if a+1<D:P[i,i+1]=S[z,a+1]/S[z,a]
   for w in range(K):P[i,w*D]+=hazard*A[z,a,w]
 assert np.allclose(P.sum(1),1)
 return P,S

def dense_filter(pi,pdur,A,emission):
 P,S=dense_matrix(pdur,A);K,D=pdur.shape;p=np.zeros(K*D);p[::D]=pi;hist=[];ll=0.
 for t,e in enumerate(emission):
  if t:p=p@P
  p=p*np.repeat(np.exp(e),D);normal=p.sum();ll+=np.log(normal);p=p/normal;hist.append(p.reshape(K,D).copy())
 return np.stack(hist),ll,S

def main():
 torch.set_num_threads(4);source=sources();rng=np.random.default_rng(6100);errors=[];gradient_errors=[]
 for arm in ['hsmm','hmm']:
  for seed in range(4):
   m=model(arm,6100+seed,dtype=torch.float64,duration=4);x=torch.tensor(rng.normal(size=(2,7,2)),dtype=torch.float64);m.eval()
   pi,dur,A=parameters(m);em=m.dist.emission.log_prob(x);trace=filter_model_sequence(m,x).log_posterior.exp().numpy();ll=m.log_likelihood(x)
   for b in range(2):
    q,l,S=dense_filter(pi.detach().exp().numpy(),dur.detach().exp().numpy(),A.detach().exp().numpy(),em[b].detach().numpy());pe=float(np.max(np.abs(q-trace[b])));le=abs(l-float(ll[b].detach()));assert pe<1e-10 and le<1e-10,(arm,pe,le)
    st=HSMMFilterState(torch.tensor(np.log(np.maximum(q[-1],1e-300)))[None]);horizons=[1,2,4];v=active_episode_survival_forecast(st,dur[None],horizons).survival_probability.detach().numpy()[0];ref=[]
    for h in horizons:
     ref.append(sum(q[-1,z,a]*S[z,a+h]/S[z,a] for z in range(3) for a in range(S.shape[1]) if a+h<S.shape[1]))
    se=float(np.max(np.abs(v-ref)));assert se<1e-10
    errors.append({'arm':arm,'seed':seed,'batch':b,'posterior_max_error':pe,'log_likelihood_error':le,'survival_max_error':se})
   pars=[p for p in m.parameters() if p.requires_grad];native=torch.autograd.grad(ll.sum(),pars,allow_unused=True)
   # Probability-space graph, not native log-frontier implementation.
   pi,dur,A=parameters(m);pd=dur.exp();S=torch.flip(torch.cumsum(torch.flip(pd,[-1]),-1),[-1]);haz=pd/S;D=pd.shape[1];matrix=[]
   for z in range(3):
    for a in range(D):
     matrix.append(torch.stack([haz[z,a]*A[z,a,w].exp() if j==0 else (S[z,j]/S[z,a] if w==z and j==a+1 else pd.new_zeros(())) for w in range(3) for j in range(D)]))
   P=torch.stack(matrix);p=torch.stack([pi[z].exp() if a==0 else pi.new_zeros(()) for z in range(3) for a in range(D)])[None].expand(2,-1);value=pi.new_zeros(2);em=m.dist.emission.log_prob(x)
   for t in range(7):
    if t:p=p@P
    p=p*em[:,t].exp().repeat_interleave(D,dim=-1);norm=p.sum(-1);value=value+norm.log();p=p/norm[:,None]
   reference=torch.autograd.grad(value.sum(),pars,allow_unused=True)
   ge=max(float((a-b).abs().max()) for a,b in zip(native,reference) if a is not None and b is not None);assert ge<1e-9,(arm,ge);gradient_errors.append(ge)
 # A chunk boundary is only a caller boundary; causal state is carried unchanged.
 m=model('hsmm',6111,dtype=torch.float64,duration=8).eval();x=torch.tensor(rng.normal(size=(3,20,2)),dtype=torch.float64);q=filter_model_sequence(m,x).log_posterior.exp();runtime=HSMMFilterRuntime(m);stream=[]
 for lo,hi in [(0,7),(7,13),(13,20)]:
  for t in range(lo,hi):stream.append(runtime.step(x[:,t]).log_posterior.exp())
 runtime_error=float((torch.stack(stream,1)-q).abs().max());assert runtime_error<1e-9,runtime_error
 before=runtime.state.filter_state.log_posterior.clone();a=runtime.forecast_survival([1,4,8]).survival_probability.clone();b=runtime.forecast_survival([1,4,8]).survival_probability.clone();assert torch.equal(a,b) and torch.equal(before,runtime.state.filter_state.log_posterior)
 # Profile complete native likelihood/backward updates on synthetic inputs only.
 timings={}
 for dev in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
  m=model('hsmm',6112,device=dev);x=torch.randn(8,64,2,device=dev);opt=torch.optim.Adam(m.parameters(),lr=.01);samples=[]
  for i in range(6):
   if dev=='cuda':torch.cuda.synchronize()
   start=time.perf_counter();opt.zero_grad();loss=-m.log_likelihood(x).mean()/64;loss.backward();opt.step()
   if dev=='cuda':torch.cuda.synchronize()
   if i>=2:samples.append(time.perf_counter()-start)
  timings[dev]=float(np.median(samples))
 chosen=min(timings,key=timings.get)
 result={'utc':utc(),'status':'NATIVE_NHSMM_CAUSAL_CORE_RUNTIME_AND_GRADIENTS_QUALIFIED','cases':errors,'max_gradient_error':max(gradient_errors),'runtime_stream_vs_batch_max_error':runtime_error,'repeat_survival_query_state_unchanged':True,'synthetic_step_seconds':timings,'profile_updates_separate':6*len(timings),'chosen_device':chosen,'python':platform.python_version(),'torch':torch.__version__,'source_python_requirement':'>=3.12','execution_mode':'Unmodified source imported directly under existing Python; package not installed, declared minimum Python version not met. Successful qualified execution is scoped to these exact code paths.','source_receipt_sha256':sha(OUT/'SOURCES61_V1.json'),'script_sha256':sha(__file__),'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
 dump(OUT/'QUALIFICATION61_V2.json',result);print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
