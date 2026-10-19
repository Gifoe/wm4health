"""Check the existing objective degeneracy at trained checkpoints; no refitting."""
import ctypes,json,os,pathlib,types
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
if os.name=='nt':
 fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
import numpy as np
import torch
from native_cofe import ROOT,build
from run_cofe_etth1 import fixed_decode,getbatch
def main():
 torch.set_num_threads(1);out=ROOT/'outputs'/'trained_cofe_audit';out.mkdir(exist_ok=True)
 raw=torch.tensor(np.genfromtxt(ROOT/'data'/'ETTh1.csv',delimiter=',',skip_header=1,usecols=tuple(range(1,8)),dtype=np.float32),device='cuda')
 data=(raw-raw[:8640].mean(0))/(raw[:8640].std(0)+1e-9)
 # Only training samples: this check cannot tune test selection or new methods.
 batch=getbatch(data,torch.arange(200,200+8*100,100,device='cuda'));rows=[]
 for variant in ['released','scale_fix','scale_fix_noaux']:
  model=build(seed=1)
  if variant!='released':model.model.decode=types.MethodType(fixed_decode,model.model)
  if variant=='scale_fix_noaux':model.weight_l1=0.;model.weight_con=0.
  path=ROOT/'outputs'/'cofe_etth1_affinity1'/variant/'best.pt';model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
  model.eval();torch.manual_seed(901);before=model(batch.past_target_cdf)
  torch.manual_seed(902);loss_before=model.loss(batch).item()
  before_noise={'first_stream_mean_abs':before[3].abs().mean().item(),'second_stream_mean_abs':before[4].abs().mean().item()}
  with torch.no_grad():
   for p in model.model.estimatenoise.parameters():p.zero_()
  model.eval();torch.manual_seed(901);after=model(batch.past_target_cdf)
  torch.manual_seed(902);loss_after=model.loss(batch).item()
  delta=max((before[2].loc-after[2].loc).abs().max().item(),(before[2].scale-after[2].scale).abs().max().item())
  assert delta==0 and loss_after<=loss_before+1e-6
  rows.append({'variant':variant,'training_batch_size':8,'checkpoint':path.relative_to(ROOT).as_posix(),'loss_before':loss_before,'loss_after_aux_zero':loss_after,'distribution_parameter_max_change':delta,'estimated_noise_before':before_noise,'estimated_noise_after_max':max(after[3].abs().max().item(),after[4].abs().max().item()),'trained_additional_models':0})
 (out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
