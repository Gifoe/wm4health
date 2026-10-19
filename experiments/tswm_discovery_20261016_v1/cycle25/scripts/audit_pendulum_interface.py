import runtime_profile
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];PREV=ROOT.parent/'cycle23'
sys.path.insert(0,str(PREV/'scripts'))
from native_context_runtime import load,verify_source
import numpy as np
from context_exploration.data.envs import make_env
from context_exploration.data.envs.envs import NoActionRendererPendulumQuadrantActionFactor,HierarchicalBoxSpace
OUT=ROOT/'outputs/interface';assert not OUT.exists();OUT.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def physical(gain,state,u,steps=1):
    env=NoActionRendererPendulumQuadrantActionFactor(np.asarray(gain));env.state=np.asarray(state,dtype='float64').copy();obs=[];domains=[];vel=[]
    for _ in range(steps*2):
        domains.append(env.get_domain());o,_,_,_=env.step(np.array([u]));obs.append(o);vel.append(float(o[2]))
    return np.asarray(obs),domains,vel
start=time.perf_counter();revision=verify_source();manifest=json.loads((PREV/'outputs/native_context_weights/manifest.json').read_text())
records=[r for r in manifest['models'] if r['config']['env_name']=='pendulum_quadrantactionfactorar2bd'];assert len(records)==6
configs=[]
for r in records:
    cfg=r['config'];assert cfg['context_dim']==16 and cfg['max_ctx_card']>=33
    assert sha(PREV/'outputs/native_context_weights'/r['path'])==r['sha256']
    configs.append(dict(seed=cfg['seed'],positive_weights=cfg['context_encoder_kwargs']['positive_weights'],rescale_raw_stddev=cfg['context_encoder_kwargs']['rescale_raw_stddev'],max_context=cfg['max_ctx_card'],n_training_steps=cfg['n_training_steps'],weight_path=r['path'],weight_sha256=r['sha256'],config=cfg))
contexts=[];env=make_env('pendulum_quadrantactionfactorar2bd')
for i in range(4096):
    env.initialize_context(30262501+i);contexts.append(env.context.copy());env.release_context()
contexts=np.asarray(contexts);np.save(OUT/'native_prior_gains.npy',contexts)
nominal_low=np.mean((np.abs(contexts)>=.5)&(np.abs(contexts)<1.25));minabs=float(abs(contexts).min())
zero_error=0.;q_error=0.;max_velocity=0.;cross_gain_error=0.;centers=np.array([3*np.pi/4,np.pi/4,-np.pi/4,-3*np.pi/4])
rng=np.random.default_rng(31262501)
for i in range(64):
    state=rng.uniform([-np.pi,-8],[np.pi,8]);ref,_,_=physical(contexts[0],state,0,steps=10);other,_,_=physical(contexts[i+1],state,0,steps=10)
    zero_error=max(zero_error,float(abs(ref-other).max()))
    for quadrant,theta in enumerate(centers):
        g=contexts[i].copy();a,domains,vel=physical(g,[theta,0],1.5)
        assert domains==[quadrant,quadrant],domains;max_velocity=max(max_velocity,max(abs(x) for x in vel))
        gg=contexts[i+64].copy();gg[quadrant]=g[quadrant];b,_,_=physical(gg,[theta,0],1.5);cross_gain_error=max(cross_gain_error,float(abs(a-b).max()))
for extreme in [-2.,-.5,.5,2.]:
    for quadrant,theta in enumerate(centers):
        a,domains,vel=physical(np.full(4,extreme),[theta,0],1.5);assert domains==[quadrant,quadrant];max_velocity=max(max_velocity,max(abs(x) for x in vel))
assert zero_error==cross_gain_error==0 and max_velocity<8
# Preserve context identity across source wrapper resets and match raw two steps.
env.initialize_context(32262501);gain=env.context.copy();env.seed(33262501);env.reset();raw=env.env.env.env;raw.state=np.array([np.pi/4,0.]);obs,_,_,_=env.step(np.array([1.5]));ref,_,_=physical(gain,[np.pi/4,0],1.5)
q_error=float(abs(obs-ref[-1]).max());assert q_error==0
env.reset();assert np.array_equal(env.context,gain);assert env.env.dt==.1
result=dict(status='pass',source_revision=revision,protocol_sha256=sha(ROOT/'PROTOCOL.md'),script_sha256=sha(pathlib.Path(__file__)),models=configs,
    physical=dict(state_observation=['cos(theta)','sin(theta)','angular_velocity'],observation_noise=False,process_noise=False,base_dt=.05,action_repeat=2,coarse_dt=.1,context_fixed_across_reset=True,zero_action_gain_dependence_max_error=zero_error,other_quadrant_gain_dependence_max_error=cross_gain_error,wrapper_raw_kernel_error=q_error,max_query_substep_velocity=max_velocity),
    native_prior=dict(samples=len(contexts),shape=list(contexts.shape),min_abs_gain=minabs,max_abs_gain=float(abs(contexts).max()),fraction_abs_in_nominal_lower_half=float(nominal_low),sign_positive_fraction=(contexts>0).mean(0).tolist(),sample_correlation=np.corrcoef(contexts.T).tolist(),sha256=sha(OUT/'native_prior_gains.npy'),
        note='Nominal Box ranges do not determine the joint sampler. HierarchicalBoxSpace seeds mixture RNG and both Box RNGs identically; audit actual sign/magnitude coupling before specifying a reference prior.'),
    comparison_caveat='Published positive/unconstrained variants also differ in stddev rescaling; not an isolated constraint ablation.',elapsed_seconds=time.perf_counter()-start)
(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='models'}),flush=True)
