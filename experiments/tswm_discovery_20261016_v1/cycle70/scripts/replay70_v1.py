import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common70_v1 import *
if os.name=='nt':
 import ctypes
 k=ctypes.windll.kernel32;mask=ctypes.c_size_t();system=ctypes.c_size_t();h=ctypes.c_void_p(-1);assert k.GetProcessAffinityMask(h,ctypes.byref(mask),ctypes.byref(system));left=mask.value;chosen=0
 for _ in range(min(4,mask.value.bit_count())):bit=left&-left;chosen|=bit;left-=bit
 assert k.SetProcessAffinityMask(h,ctypes.c_size_t(chosen))
start=time.perf_counter();q=read(OUT/'QUALIFICATION70_V1.json');assert read(OUT/'qualification70_v1_exit.json')['exit_code']==0
H=np.array([[1.,.3],[-.2,.9]]);R=np.array([[.18,.02],[.02,.24]])

def rts(m,S,As,bs,Q,y,scales):
 # Separate forward Kalman filter/backward RTS; no joint precision inversion.
 T=len(y);pred_m=[];pred_S=[];fm=[];fc=[];eye=np.eye(2)
 for t in range(T):
  if t:m=As[t-1]@fm[-1]+bs[t-1];S=As[t-1]@fc[-1]@As[t-1].T+Q
  pred_m.append(m.copy());pred_S.append(S.copy());V=H@S@H.T+R/scales[t];K=np.linalg.solve(V,H@S).T
  m=m+K@(y[t]-H@m);S=(eye-K@H)@S@(eye-K@H).T+K@(R/scales[t])@K.T;fm.append(m.copy());fc.append(S.copy())
 sm=np.asarray(fm).copy();ss=np.asarray(fc).copy();cross=np.zeros((T-1,2,2))
 for t in range(T-2,-1,-1):
  J=np.linalg.solve(pred_S[t+1],As[t]@fc[t]).T;sm[t]=fm[t]+J@(sm[t+1]-pred_m[t+1]);ss[t]=fc[t]+J@(ss[t+1]-pred_S[t+1])@J.T;cross[t]=ss[t+1]@J.T
 return sm,ss,cross

rows=[];maxima={};sources=[]
for s in q['source_files']:
 p=SOURCE/s['path'];b=p.read_bytes();assert sha(p)==s['sha256'];assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==s['git_blob_sha1'];sources.append(s['path'])
for row in q['cells']:
 p=pathlib.Path(row['array_path']);assert sha(p)==row['array_sha256'];z=np.load(p,allow_pickle=False);errs={}
 for i in range(4):
  sm,ss,cross=rts(z['m0'],z['S0'],z['As'][i],z['bs'][i],z['Q'],z['obs_y'][i],z['scale'][i]);dm=z['dense_posterior_mean'][i];df=z['dense_posterior_full_cov'][i];ds=np.stack([df[2*t:2*t+2,2*t:2*t+2] for t in range(32)]);dx=np.stack([df[2*t+2:2*t+4,2*t:2*t+2] for t in range(31)])
  checks={'rts_dense_mean':abs(sm-dm).max(),'rts_dense_cov':abs(ss-ds).max(),'rts_dense_cross':abs(cross-dx).max(),'rts_native_mean':abs(sm-z['posterior_mean'][i]).max(),'rts_native_cov_minus_padding':abs(ss-(z['posterior_cov'][i]-np.eye(2)*1e-5)).max(),'rts_native_cross_raw':abs(cross-z['posterior_cross'][i]).max()}
  pm,pc,_=rts(z['m0'],z['S0'],z['As'][i,:15],z['bs'][i,:15],z['Q'],z['obs_y'][i,:16],z['scale'][i,:16]);checks.update(prefix_forecast_initial_mean=abs(pm[-1]-z['forecast_mean'][:,i,0]).max(),prefix_forecast_initial_cov=abs(pc[-1]-z['forecast_cov'][:,i,0]).max())
  for key,val in checks.items():errs[key]=max(errs.get(key,0.),float(val))
 for key,val in errs.items():assert val<(1e-4 if key=='rts_native_cross_raw' else 1e-7),(row['id'],key,val);maxima[key]=max(maxima.get(key,0.),val)
 assert row['native_update_retained_flag']==bool(row['start_from_invariant'])
 if not row['start_from_invariant']:assert row['native_updated_initial_mean_error']>.5 and row['native_updated_initial_cov_error']>.3
 rows.append(dict(id=row['id'],sha256=row['array_sha256'],independent_errors=errs))
flag=q['constant_flag_control'];assert sha(flag['artifact'])==flag['artifact_sha256'];f=np.load(flag['artifact'],allow_pickle=False)
assert np.array_equal(f['fixed_zero_mean'],f['fixed_one_mean']) and np.array_equal(f['fixed_zero_cov'],f['fixed_one_cov']);assert float(abs(f['native_one_mean']-f['native_zero_mean']).max())==flag['native']['mean_max_abs']>0
from native70_v1 import one_cell
fresh,rec=one_cell(70001,'continuous','additive',False);saved=next(x for x in q['cells'] if x['id']=='continuous_additive_invariant0_seed70001');old=np.load(saved['array_path'],allow_pickle=False);fresh_diffs={key:float(abs(fresh[key]-old[key]).max()) for key in fresh};assert max(fresh_diffs.values())<1e-12
obj=dict(utc=utc(),status='PASS',independent_algorithm='Joseph-form Kalman filter and Rauch-Tung-Striebel smoother; no dense joint conditioning code imported',counts=dict(source_blobs=len(sources),cells=len(rows),sequences=96,full_smoothers=96,prefix_smoothers=96,fresh_native_cells=1,flag_control_archives=1),cells=rows,max_errors=maxima,fresh_native_array_max_abs=fresh_diffs,qualification_sha256=sha(OUT/'QUALIFICATION70_V1.json'),runtime_seconds=time.perf_counter()-start,training=False,known_system_and_likelihood_privileged=True,accepted_final_directions=0)
dump(OUT/'INDEPENDENT_REPLAY70_V1.json',obj);print(json.dumps({k:obj[k] for k in ['status','counts','max_errors','runtime_seconds']}),flush=True)
