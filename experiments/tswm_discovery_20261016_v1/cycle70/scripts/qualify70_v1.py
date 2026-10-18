import sys,pathlib,time,importlib.metadata
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common70_v1 import *
if os.name=='nt':
 import ctypes
 mask=ctypes.c_size_t();system_mask=ctypes.c_size_t();h=ctypes.c_void_p(-1);k=ctypes.windll.kernel32;assert k.GetProcessAffinityMask(h,ctypes.byref(mask),ctypes.byref(system_mask));chosen=0;left=mask.value
 for _ in range(min(4,mask.value.bit_count())):bit=left&-left;chosen|=bit;left-=bit
 assert k.SetProcessAffinityMask(h,ctypes.c_size_t(chosen))
selection=read(ROOT/'SOURCE_SELECTION70_V1.json');source=[]
for row in selection['files']:
 p=SOURCE/row['path'];b=p.read_bytes();assert len(b)==row['bytes'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['git_blob_sha1'];source.append(dict(**row,sha256=sha(p)))
from native70_v1 import *
start=time.perf_counter();errors=[]
try:
 original=ACLGParam(action_mapper=DiscreteChoiceActionMapper(2),transition_matrix='additive',transition_bias='action_dependent',Q_dist_map=MVNDiag(2),opt_params=['A','Q'],start_from_invariant=True,A=jnp.asarray(A),Q=jnp.array([-2.7,-3.]),b=jnp.asarray(B));original.init(jr.PRNGKey(70001),jnp.zeros((4,31,1),dtype=jnp.int32));errors.append(dict(test='native_discrete_initialization',status='SUCCEEDED'))
except TypeError as ex:errors.append(dict(test='native_discrete_initialization',status='TYPE_ERROR_RETAINED_ADAPTER_REQUIRED',exception=str(ex)))
rows=[];base=storage()
for seed in [70001,70002,70003]:
 for kind in ['continuous','discrete']:
  for mode in ['additive','multiplicative']:
   for invariant in [True,False]:
    arrays,rec=one_cell(seed,kind,mode,invariant);name=f'{kind}_{mode}_invariant{int(invariant)}_seed{seed}';p=base/(name+'.npz');assert not p.exists();np.savez_compressed(p,**arrays);rec.update(id=name,array_path=str(p),array_sha256=sha(p));rows.append(rec);dump(OUT/'cells'/(name+'.json'),rec);print(json.dumps(dict(cell=name,errors=rec['errors'],native_update_retained_flag=rec['native_update_retained_flag'])),flush=True)
# One explicit constant/no-bias flag control, without learning.
mapper=ContinuousActionMapper(ActionNetwork(network=nn.Sequential([nn.Dense(12),nn.tanh,nn.Dense(12)]),latent_dim=2));zero=jnp.zeros((1,31,2));one=jnp.ones((1,31,2));prior=ACLGParam(action_mapper=mapper,transition_matrix='constant',transition_bias='none',Q_dist_map=MVNDiag(2),opt_params=['A','Q'],start_from_invariant=True,A=jnp.asarray(A),Q=jnp.array([-2.7,-3.]),b=jnp.zeros(2));pp=prior.init(jr.PRNGKey(70099),zero);native=preserving_update(prior,pp);cz,_=native.to_chain(32,zero);co,_=native.to_chain(32,one);flag_native=dict(mean_max_abs=float(abs(np.asarray(cz.params['means'])-np.asarray(co.params['means'])).max()),cov_max_abs=float(abs(np.asarray(cz.params['covs'])-np.asarray(co.params['covs'])).max()))
bridge=ContinuousFlagBridge(mapper,'constant','none');fixed=ACLGParam(action_mapper=bridge,transition_matrix='constant',transition_bias='none',Q_dist_map=MVNDiag(2),opt_params=['A','Q'],start_from_invariant=True,A=jnp.asarray(A),Q=jnp.array([-2.7,-3.]),b=jnp.zeros(2));pp=fixed.init(jr.PRNGKey(70099),zero);fixed=preserving_update(fixed,pp);fz,_=fixed.to_chain(32,zero);fo,_=fixed.to_chain(32,one);flag_fixed=dict(mean_max_abs=float(abs(np.asarray(fz.params['means'])-np.asarray(fo.params['means'])).max()),cov_max_abs=float(abs(np.asarray(fz.params['covs'])-np.asarray(fo.params['covs'])).max()));assert flag_fixed['mean_max_abs']==flag_fixed['cov_max_abs']==0
fp=base/'CONSTANT_FLAG_CONTROL70_V1.npz';np.savez_compressed(fp,native_zero_mean=np.asarray(cz.params['means']),native_one_mean=np.asarray(co.params['means']),native_zero_cov=np.asarray(cz.params['covs']),native_one_cov=np.asarray(co.params['covs']),fixed_zero_mean=np.asarray(fz.params['means']),fixed_one_mean=np.asarray(fo.params['means']),fixed_zero_cov=np.asarray(fz.params['covs']),fixed_one_cov=np.asarray(fo.params['covs']))
obj=dict(utc=utc(),status='ACTION_CONDITIONED_COMPONENT_QUALIFICATION_COMPLETE',source_repo=selection['repo'],source_revision=selection['revision'],source_files=source,protocol_sha256=sha(ROOT/'PROTOCOL70_V1.md'),forecast_addendum_sha256=sha(ROOT/'FORECAST_ADDENDUM70_V1.md'),cells=rows,interface_checks=errors,constant_flag_control=dict(native=flag_native,ordinary_output_mask_adapter=flag_fixed,artifact=str(fp),artifact_sha256=sha(fp)),runtime_seconds=time.perf_counter()-start,environment=dict(Python=sys.version,executable=sys.executable,isolated=bool(sys.flags.isolated),devices=[str(x) for x in jax.devices()],jax_enable_x64=bool(jax.config.jax_enable_x64),packages={x:importlib.metadata.version(x) for x in ['jax','jaxlib','flax','optax','numpy','dynamax','tensorflow-probability']}),neural_training=False,learned_representation_test=False,known_system_and_likelihood_privileged=True,upstream_covariance_padding=1e-5,accepted_final_directions=0)
dump(OUT/'QUALIFICATION70_V1.json',obj);print(json.dumps(dict(status=obj['status'],cells=len(rows),runtime_seconds=obj['runtime_seconds'],interface_checks=errors,constant_flag_control=obj['constant_flag_control'])),flush=True)
