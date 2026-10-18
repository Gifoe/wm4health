from native87_v1 import *
from physics87_v1 import *
import time,subprocess,traceback
def main():
 check_native();torch.manual_seed(2026872);assert torch.cuda.is_available();torch.cuda.reset_peak_memory_stats();free,total=torch.cuda.mem_get_info();assert free>12*1024**3
 rng=np.random.default_rng(20268799);p=parameters('stuart_landau',16,rng);ic=rng.normal(0,.8,(16,2,2));x=trajectories('stuart_landau',p,ic,steps=32);loc=rng.uniform(-2.5,2.5,(16,32,2)).astype(np.float32)
 data=batch(x,loc,field('stuart_landau',p,loc),field('stuart_landau',p,x),'cuda');m=load_native();m.train();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);old=next(m.parameters()).detach().clone();trace=[]
 for step in range(5):
  st=time.perf_counter();opt.zero_grad(set_to_none=True);out=m(data);loss=loss_value(out);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step();torch.cuda.synchronize();trace.append(dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),wall_seconds=time.perf_counter()-st));log(dict(stage='native_smoke',**trace[-1]))
 assert not torch.equal(old,next(m.parameters()).detach());m.eval();prediction=predict_field(m,x,loc);assert np.isfinite(prediction).all()
 receipt=dict(utc=utc(),status='PASS_NATIVE_FULL_PARAMETER_GRADIENT_AND_GPU',parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),native_loss_keys=list(out),train_config=vars(m.train_config),model_config=vars(m.model_config),trace=trace,free_gpu_bytes_before=free,total_gpu_bytes=total,gpu_peak_bytes=torch.cuda.max_memory_allocated(),weights_sha256=WEIGHTS['sha256'],native_source_files_verified=len(M80['files']),protocol_sha256=sha(C/'PROTOCOL87_V1.md'),eval_opened=False)
 write(O/'NATIVE_SMOKE87_V1.json',receipt);log(dict(stage='smoke_complete',status=receipt['status'],gpu_peak_bytes=receipt['gpu_peak_bytes']))
if __name__=='__main__':main()
