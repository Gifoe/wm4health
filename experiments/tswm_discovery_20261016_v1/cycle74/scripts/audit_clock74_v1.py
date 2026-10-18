import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common74_v1 import *
import numpy as np
from scipy.io import loadmat
r=read(OUT/'DATA_EMPS74_V2.json')['training_members'][0];assert sha(r['path'])==r['sha256'];t=loadmat(r['path'],variable_names=['t'])['t'][:,0];d=np.diff(t);uniq,counts=np.unique(np.round(d,8),return_counts=True)
o=dict(utc=utc(),status='EMPS_TRAIN_CLOCK_INSPECTED_NOT_RESAMPLED',source_sha256=r['sha256'],n=len(t),t_first=float(t[0]),t_last=float(t[-1]),first_20=t[:20].tolist(),first_step=float(d[0]),median_step=float(np.median(d)),min_step=float(d.min()),max_step=float(d.max()),nonpositive_steps=int((d<=0).sum()),all_t_exactly_float32_representable=bool(np.array_equal(t,t.astype(np.float32).astype(np.float64))),rounded_step_histogram=[dict(step=float(v),count=int(n)) for v,n in zip(uniq,counts)],max_abs_deviation_from_first_step=float(abs(d-d[0]).max()),max_abs_deviation_from_uniform_index_clock=float(abs(t-(t[0]+np.arange(len(t))*d[0])).max()),official_test_values_parsed=False)
dump(OUT/'TRAIN_CLOCK74_V1.json',o);print(json.dumps(o),flush=True)
