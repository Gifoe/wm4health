"""Synthetic-only wiring, numerical gradients and CUDA profiling for the added control."""
from gru60_v1 import *
import copy,time,statistics
torch.set_num_threads(4)
def main():
    torch.manual_seed(6010);cpu=GRUControl().double();assert sum(p.numel() for p in cpu.parameters())==6241
    hu=torch.randn(7,2,2,dtype=torch.float64)*.1;hy=torch.randn(7,2,1,dtype=torch.float64)*.1;fu=torch.randn(9,2,2,dtype=torch.float64)*.1
    expected=cpu(hu,hy,fu);_,hidden=cpu.encoder(torch.cat([hu,hy],-1));values=[]
    for k in range(len(fu)):
        state,hidden=cpu.decoder(fu[k:k+1],hidden);values.append(cpu.output(state))
    manual=torch.cat(values);gr=torch.autograd.grad(expected.square().mean(),tuple(cpu.parameters()));gm=torch.autograd.grad(manual.square().mean(),tuple(cpu.parameters()))
    wiring=max(float((expected-manual).abs().max().detach()),max(float((a-b).abs().max()) for a,b in zip(gr,gm)));assert wiring<1e-12
    fc=copy.deepcopy(cpu).float();fg=copy.deepcopy(fc).cuda();hu=hu.float();hy=hy.float();fu=fu.float()
    pc=fc(hu,hy,fu);pg=fg(hu.cuda(),hy.cuda(),fu.cuda());gc=torch.autograd.grad(pc.square().mean(),tuple(fc.parameters()));gg=torch.autograd.grad(pg.square().mean(),tuple(fg.parameters()))
    gpu_error=max(float((pc-pg.cpu()).abs().max().detach()),max(float((a-b.cpu()).abs().max()) for a,b in zip(gc,gg)));assert gpu_error<2e-5
    model=GRUControl().cuda();opt=torch.optim.Adam(model.parameters(),lr=.0005);hu=torch.randn(100,32,2,device='cuda');hy=torch.randn(100,32,1,device='cuda');fu=torch.randn(128,32,2,device='cuda');target=torch.randn(128,32,1,device='cuda');times=[]
    for i in range(15):
        torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad();loss=(model(hu,hy,fu)-target).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();torch.cuda.synchronize()
        if i>=5:times.append(time.perf_counter()-start)
    obj={'utc':utc(),'status':'PASS_ORDINARY_GRU_WIRING_AND_CPU_CUDA_GRADIENTS','parameters':6241,'float64_wiring_gradient_error':wiring,'cpu_cuda_float32_max_error':gpu_error,'device':'cuda','synthetic_cuda_step_median_seconds':statistics.median(times),'synthetic_optimizer_steps':15,'real_data_training_updates':0,'model_sha256':sha(ROOT/'scripts/gru60_v1.py'),'qualifier_sha256':sha(__file__),'amendment_sha256':sha(ROOT/'GRU_CONTROL60_V1.md'),'new_method':False}
    dump(OUT/'GRU_QUALIFICATION60_V1.json',obj);print(json.dumps(obj),flush=True)
if __name__=='__main__':main()
