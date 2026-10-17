"""Immutable protocol, pinned native imports and remote-only artifact paths."""
import os
os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['JAX_ENABLE_X64']='true'
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
if os.name=='nt':
 import ctypes
 fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff0000))
import hashlib,json,pathlib,sys,datetime
import numpy as np
import jax,jax.numpy as jnp,jax.random as jr
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent
PLAN_PATH=ROOT/'plan54_v1.json';PLAN=json.loads(PLAN_PATH.read_text(encoding='utf-8'))
SOURCE=PROJECT/'upstream/rp-gssm';sys.path.insert(0,str(SOURCE))
STORAGE=json.loads((PROJECT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
OUT=ROOT/'outputs';OUT.mkdir(parents=True,exist_ok=True)
DATA=pathlib.Path(STORAGE['data_root'])/'cycle54';DATA.mkdir(parents=True,exist_ok=True)
CKPT=pathlib.Path(STORAGE['checkpoint_root'])/'cycle54';CKPT.mkdir(parents=True,exist_ok=True)
ART=pathlib.Path(STORAGE['artifact_root'])/'cycle54';ART.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,o):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(o,indent=2,allow_nan=False),encoding='utf-8')
def source_check():
 actual={r:sha(SOURCE/r) for r in PLAN['source_files_sha256']}
 assert actual==PLAN['source_files_sha256'],'Native source drift'
 return actual
def case(stem):return next(t for t in PLAN['tasks'] if t['name']==stem)
def data_path(task):return DATA/(task+'.npz')
def load_data(task):return dict(np.load(data_path(task)))
def rp_model(task,seed):
 from rp_ssm.config import Config
 from rp_ssm.dists import LGStationaryParam
 from rp_ssm.recognition import MLP,RPMRecognition
 from rp_ssm.distmaps import MVNDiag
 from rp_ssm.rpm import RPSSM,ConstrainedIVFreeEnergy
 K=case(task)['latent_dim']
 prior=LGStationaryParam(start_from_invariant=True,stay_at_invariant=True,opt_params=['A'],A=PLAN['initial_A_diagonal']*jnp.eye(K))
 rec=RPMRecognition(network=MLP(tuple()),dist_map=MVNDiag(K),constant_cov=True)
 energy=ConstrainedIVFreeEnergy(RPSSM(prior,[rec]))
 cfg=Config(num_iter=PLAN['rp_iterations'],batch_size=PLAN['batch_size'],seed=seed,prior_lr=PLAN['learning_rate'],rec_lr=(PLAN['learning_rate'],),stabilize_A=PLAN['stabilize_A'],em=PLAN['em'],jit=True)
 return energy,cfg
