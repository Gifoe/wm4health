import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native68_v1 import *
from operators68_v1 import *
native=NormWear();meta=MetaID();rows=[]
for i,family in enumerate(['linear','cubic']):
 raw,params=generate(689900+i,family);h=raw[:,:L];known=raw[1,L:];target=raw[0,L:];pred,_=raw_forecast(h,known,3,0.);err=float(abs(pred-target).max());assert err<1e-6,err
 native.set_case(h,known);a=native.forecast('bayes_indexed');original=native.z.clone();native.z[:,0,L:]=111.;b=native.forecast('bayes_indexed');native.z=original;assert np.array_equal(a,b)
 out={}
 for mode in ['native_predict','clamped_simulate','bayes_indexed','bayes_same_view']:p=native.forecast(mode);out[mode]=dict(shape=list(p.shape),finite=bool(np.isfinite(p).all()))
 future,fit=latent_forecast(native.E,h,known,16,.1);p=native.decode(future);assert p.shape==(H,) and np.isfinite(p).all();mp,log=meta.forecast(h,known);changed,_=meta.forecast(h,known,hidden_fill=111.);assert np.array_equal(mp,changed)
 rows.append(dict(family=family,fixture_seed=689900+i,raw_cubic_exact_recovery_max_abs=err,native_hidden_future_max_abs=0,meta_hidden_future_max_abs=float(abs(mp-changed).max()),native_paths=out,latent_operator_shape=list(p.shape),meta_shape=list(mp.shape),meta_aux=log))
dump(OUT/'QUALIFICATION68_V1.json',dict(utc=utc(),status='PASS',fixtures=rows,native_parameters=sum(p.numel() for p in native.model.parameters()),meta_parameters=meta.parameter_count,meta_state_tensors=meta.state_tensors,meta_model_args=meta.args,meta_checkpoint_keys=meta.checkpoint_keys,meta_cfg=vars(meta.cfg) if hasattr(meta.cfg,'__dict__') else meta.cfg,native_sampler_AST_sha256=native.sampler_AST_sha256,source_receipt_sha256=sha(OUT/'MODEL_FILES68_V1.json'),protocol_sha256=sha(ROOT/'PROTOCOL68_V1.md'),qualification_only_no_benchmark_scores=True));print(json.dumps(dict(status='QUALIFICATION_PASS',meta_parameters=meta.parameter_count,fixtures=rows)),flush=True)
