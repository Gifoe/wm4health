from models65 import *
import time,platform
source_check();torch.set_num_threads(4);torch.manual_seed(65001);np.random.seed(65001);cpu=csdi('cpu');x=torch.randn(8,2,H);m=mask(x,'block');s=side(cpu,x,m);cpu.eval();p=cpu.diffmodel(cpu.set_input_to_diffmodel(x,x,m),s,torch.arange(8)%50);loss=cpu.calc_loss(x,m,torch.ones_like(m),s,1);loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(v.grad).all() for v in cpu.parameters() if v.grad is not None)
device='cuda' if torch.cuda.is_available() else 'cpu';gpu=csdi(device);gpu.load_state_dict(cpu.state_dict());gpu.eval()
with torch.no_grad():pred=gpu.diffmodel(gpu.set_input_to_diffmodel(x.to(device),x.to(device),m.to(device)),s.to(device),torch.arange(8,device=device)%50).cpu()
err=float((pred-p.detach()).abs().max());assert err<2e-4
opt=torch.optim.Adam(gpu.parameters(),lr=.001);xx=x.to(device);mm=m.to(device);ss=side(gpu,xx,mm);gpu.train();start=time.perf_counter()
for j in range(100):opt.zero_grad();v=gpu.calc_loss(xx,mm,torch.ones_like(mm),ss,1);assert torch.isfinite(v);v.backward();opt.step()
assert all(torch.isfinite(v).all() for v in gpu.parameters());gpu.eval()
with torch.no_grad():
 torch.manual_seed(65005);a=gpu.impute(xx,mm,ss,1)
 torch.manual_seed(65005);b=gpu.impute(xx+(1-mm)*111,mm,ss,1)
assert torch.equal(a,b),'Unobserved data values affect native inference'
dump(OUT/'QUALIFICATION65_V1.json',dict(utc=utc(),status='PASS',device=device,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),native_source_files=len(source_check()['files']),architecture_parameters=sum(v.numel() for v in cpu.parameters()),MDN_parameters=sum(v.numel() for v in MDN().parameters()),CPU_GPU_eval_max_abs_difference=err,finite_optimizer_fixture_steps=100,sampler_hidden_value_invariance_max_abs_difference=float((a-b).abs().max()),fixture_runtime_seconds=time.perf_counter()-start,scope='Native CSDI architecture/loss/imputation qualification, not trained performance or original benchmark reproduction'))
print((OUT/'QUALIFICATION65_V1.json').read_text())
