import sys,pathlib,argparse,importlib.metadata
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
p=argparse.ArgumentParser();p.add_argument('--qualification',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args()
acq=read(OUT/'ACQUISITION71_V1.json')
for s in acq['sources']:assert sha(s['server_path'])==s['sha256']
for d in acq['datasets']:assert sha(d['server_path'])==d['subset_sha256']
from native71_v1 import fit,torch,np
if a.qualification:
 r=fit(2026710,1,128,'qualification71_v1');dump(OUT/'QUALIFICATION71_V1.json',dict(status='PASS',fit_result_sha256=sha(OUT/'qualification71_v1_result.json'),module_updates=r['module_updates'],proposal_updates=r['proposal_updates'],seconds=r['wall_seconds'],expected_epochs_per_formal_fit=2000,parameters=r['module_parameters']+r['proposal_parameters'],environment=dict(Python=sys.version,executable=sys.executable,isolated=bool(sys.flags.isolated),torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(),cuda=torch.version.cuda,TF32=False),reserved_eval_rows_accessed=0));print('QUALIFICATION_PASS',flush=True)
else:
 assert read(OUT/'QUALIFICATION71_V1.json')['status']=='PASS';assert a.seed in [2026711,2026712,2026713];fit(a.seed,2000,2048,'formal_seed'+str(a.seed)+'_v1')
