"""Qualify the actual DDE history/gradient path, then discard timing weights."""
from common48_v3 import *
import inspect
def main():
 out=R/'outputs/qualification_v1';assert not out.exists();out.mkdir(parents=True);torch.set_default_dtype(torch.float64)
 # Exact global exponential solution for a constant-delay linear equation.
 lam=-.7;tau=.3;coefficient=lam*math.exp(lam*tau);errors=[];recurrence_errors=[];solutions={}
 for h in [1/24,1/48,1/96]:
  t=torch.arange(1,49,dtype=torch.float64)/24
  def f(t,y,args,*,history):return coefficient*history[0]
  def initial(t):return torch.exp(lam*t).reshape(1,1)
  actual=integrate(f,RK2(),torch.tensor(0.),torch.tensor(2.),t,initial,None,dt0=torch.tensor(h),delays=torch.tensor([tau]),discretize_then_optimize=True).numpy()[0,:,0];true=np.exp(lam*t.numpy());errors.append(float(np.max(abs(actual-true))));tt=np.arange(int(round(2/h))+1)*h;yy=[1.]
  def history_value(s):return math.exp(lam*s) if s<=0 else float(np.interp(s,tt[:len(yy)],yy))
  for i in range(len(tt)-1):s=tt[i];v=yy[-1];k1=coefficient*history_value(s-tau);k2=coefficient*history_value(s+h-tau);yy.append(v+h*(k1+k2)/2)
  reference=np.interp(t.numpy(),tt,yy);difference=float(np.max(abs(reference-actual)));recurrence_errors.append(difference);assert difference<2e-10,(h,difference);solutions[str(h)]=actual
 assert errors[2]<errors[1]<errors[0] and errors[0]<.001,(errors,recurrence_errors)
 # Neural-delay gradient check in the actual adapted model, off knots.
 seed(20264800);model=MemoryModel('learned_dde').double();nn.init.normal_(next(x for x in reversed(list(model.core.mlp.children())) if isinstance(x,nn.Linear)).weight,std=.02);history=(torch.sin(torch.arange(-95,1)[None,:,None]/13)+.1*torch.cos(torch.arange(-95,1)[None,:,None]/7)).repeat(2,1,1);phase=torch.tensor([.125,.625]);scalar=model(history,phase,horizon=24).square().mean();scalar.backward();automatic=model.core.delays.grad.clone();checks=[]
 for j in range(2):
  original=float(model.core.delays[j]);eps=2e-5
  with torch.no_grad():model.core.delays[j]=original+eps;plus=float(model(history,phase,horizon=24).square().mean());model.core.delays[j]=original-eps;minus=float(model(history,phase,horizon=24).square().mean());model.core.delays[j]=original
  fd=(plus-minus)/(2*eps);error=abs(fd-float(automatic[j]));assert error<3e-4+3e-3*abs(fd),(j,fd,float(automatic[j]),error);checks.append({'delay_index':j,'autograd':float(automatic[j]),'finite_difference':fd,'absolute_error':error})
 torch.set_default_dtype(torch.float32);data=load('ett','train');cpu_history=data['history'][:16];cpu_future=data['future'][:16];cpu_phase=data['phase'][:16];timings=[]
 for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
  for method in METHODS:
   seed(20264800);m=MemoryModel(method).to(device);hist=cpu_history.to(device);phase0=cpu_phase.to(device);target=cpu_future.to(device);initial=m(hist,phase0);assert torch.allclose(initial,hist[:,-1:,:].expand_as(initial),atol=1e-6,rtol=1e-6),(method,'persistence boundary');opt=torch.optim.Adam(m.parameters(),lr=1e-3);steps=[]
   for i in range(3):
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter();opt.zero_grad(set_to_none=True);prediction=m(hist,phase0);loss=(prediction-target).square().mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();m.project()
    if device=='cuda':torch.cuda.synchronize()
    steps.append(time.perf_counter()-start)
   timings.append({'device':device,'method':method,'discarded_updates':3,'step_seconds':steps,'median_seconds':float(np.median(steps)),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'nominal_parameters':sum(p.numel() for p in m.parameters()),'native_class_file':inspect.getfile(m.core.__class__)})
 selected=min({r['device'] for r in timings},key=lambda d:sum(r['median_seconds'] for r in timings if r['device']==d));result={'status':'QUALIFIED_REAL_PROCESS_MEMORY_SCREEN','utc':now(),'linear_exponential_max_errors':errors,'independent_heun_max_errors':recurrence_errors,'neural_delay_gradient_checks':checks,'timings':timings,'selected_training_device':selected,'discarded_updates':sum(r['discarded_updates'] for r in timings),'source_sha256':sha(__file__),'common_sha256':sha(R/'scripts/common48_v3.py'),'source_origin_verification_sha256':sha(R/'SOURCE_ORIGIN_VERIFICATION.json'),'EVAL_opened':False};np.savez_compressed(out/'numerical_reference.npz',**solutions);dump(out/'qualification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
