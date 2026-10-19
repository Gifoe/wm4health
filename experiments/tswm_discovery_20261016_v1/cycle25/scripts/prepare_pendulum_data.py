import runtime_profile
import hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];PREV=ROOT.parent/'cycle23'
sys.path.insert(0,str(PREV/'scripts'))
from native_context_runtime import verify_source
from context_exploration.data.envs import make_env
import numpy as np
OUT=ROOT/'outputs/data';assert not OUT.exists();OUT.mkdir(parents=True)
gate=json.loads((ROOT/'outputs/interface/results.json').read_text());assert gate['status']=='pass'
N=64;env=make_env('pendulum_quadrantactionfactorar2bd');rng=np.random.default_rng(22262501)
actions=rng.uniform(-2,2,(32,N,1));states=np.zeros((33,N,3));rawstates=np.zeros((33,N,2));gains=[]
centers=np.array([3*np.pi/4,np.pi/4,-np.pi/4,-3*np.pi/4]);probe=np.zeros((2,N,3));query=np.zeros((N,4,3))
for i in range(N):
    env.initialize_context(20262501+i);gains.append(env.context.copy());env.seed(21262501+i);states[0,i]=env.reset();raw=env.env.env.env;rawstates[0,i]=raw.state
    for t in range(32):
        states[t+1,i],_,_,_=env.step(actions[t,i]);rawstates[t+1,i]=raw.state
    for j,u in enumerate([0.,1.5]):
        env.reset();raw.state=np.array([np.pi/4,0.]);probe[j,i],_,_,_=env.step(np.array([u]))
    for q,theta in enumerate(centers):
        env.reset();raw.state=np.array([theta,0.]);query[i,q],_,_,_=env.step(np.array([1.5]))
    assert np.array_equal(env.context,gains[-1]);env.release_context()
f=OUT/'pendulum.npz';np.savez_compressed(f,states=states,actions=actions,raw_states=rawstates,gains=gains,probe=probe,query=query,query_theta=centers,probe_state=np.array([np.cos(np.pi/4),np.sin(np.pi/4),0.]),context_seeds=np.arange(20262501,20262501+N),trajectory_seeds=np.arange(21262501,21262501+N))
receipt=dict(status='pass',systems=N,source_revision=verify_source(),file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),context_sizes=[0,8,32],noise_added=False,parameter_identity_fixed_across_resets=True)
(OUT/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
