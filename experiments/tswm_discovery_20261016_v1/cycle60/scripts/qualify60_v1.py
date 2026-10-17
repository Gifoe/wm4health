"""Qualify native wiring/gradients and choose a device using only synthetic timings."""
from common60 import *
from model60_v1 import *
import copy,time,statistics
torch.set_num_threads(4)
def main():
    source_check();torch.manual_seed(6000)
    base=PostUpdateSSM().double();hu=torch.randn(7,2,2,dtype=torch.float64)*.1;hy=torch.randn(7,2,1,dtype=torch.float64)*.1;fu=torch.randn(9,2,2,dtype=torch.float64)*.1
    reference=StateSpaceSimulator(base.f,base.g)
    x=base.estimator(hu,hy);expected=reference(x,torch.cat([fu,torch.zeros_like(fu[:1])]))[1:]
    actual=base(hu,hy,fu);loss_ref=expected.square().mean();loss_actual=actual.square().mean();pars=tuple(base.parameters())
    gr=torch.autograd.grad(loss_ref,pars);ga=torch.autograd.grad(loss_actual,pars)
    output_error=float((actual-expected).abs().max());gradient_error=max(float((a-b).abs().max()) for a,b in zip(gr,ga));assert max(output_error,gradient_error)<1e-12
    scripted=torch.jit.script(copy.deepcopy(base));ss=scripted(hu,hy,fu);sg=torch.autograd.grad(ss.square().mean(),tuple(scripted.parameters()))
    script_error=max(float((ss-actual).abs().max()),max(float((a-b).abs().max()) for a,b in zip(sg,ga)));assert script_error<1e-12
    timings={};steps=0;gpu_error=None
    for device in ['cpu','cuda']:
        if device=='cuda' and not torch.cuda.is_available():continue
        torch.manual_seed(6001);model=torch.jit.script(PostUpdateSSM()).to(device);opt=torch.optim.Adam(model.parameters(),lr=.0005)
        hu=torch.randn(100,32,2,device=device);hy=torch.randn(100,32,1,device=device);fu=torch.randn(128,32,2,device=device);target=torch.randn(128,32,1,device=device)
        times=[]
        for k in range(15):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();opt.zero_grad();pred=model(hu,hy,fu);loss=(pred-target).square().mean();assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();steps+=1
            if device=='cuda':torch.cuda.synchronize()
            if k>=5:times.append(time.perf_counter()-start)
        timings[device]={'median_seconds':statistics.median(times),'measured_steps':10,'warmup_steps':5,'parameters':sum(p.numel() for p in model.parameters())}
        del model,opt
        if device=='cuda':torch.cuda.empty_cache()
    chosen=min(timings,key=lambda k:timings[k]['median_seconds'])
    result={'utc':utc(),'status':'PASS_NATIVE_TIMING_ADAPTER_OUTPUT_GRADIENT_AND_SCRIPT_QUALIFICATION','source_commit':source_check()['source_commit'],'float64_output_error':output_error,'float64_gradient_error':gradient_error,'script_error':script_error,'synthetic_step_timings':timings,'chosen_device':chosen,'synthetic_profiling_optimizer_steps':steps,'real_data_training_updates':0,'selection_uses_benchmark_outcomes':False,'torch_version':torch.__version__,'qualifier_sha256':sha(__file__),'model_adapter_sha256':sha(ROOT/'scripts/model60_v1.py'),'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'QUALIFICATION60_V1.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
