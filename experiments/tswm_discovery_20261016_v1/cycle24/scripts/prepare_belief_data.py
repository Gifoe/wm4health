import runtime_profile
import hashlib,json,pathlib,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];PREV=ROOT.parent/'cycle23'
sys.path.insert(0,str(PREV/'scripts'));from native_context_runtime import verify_source
import numpy as np
import torch
from context_exploration.data.envs.envs import ActionSquashEnvBase
OUT=ROOT/'outputs/data';assert not OUT.exists();OUT.mkdir(parents=True)
rows=[]
for kind,seed in [(1,20262401),(2,20262402)]:
    rng=np.random.default_rng(seed);alpha=rng.uniform(-1,1,size=(64,1)).astype('float32');states=[rng.normal(size=(64,2)).astype('float32')]
    actions=rng.uniform(-2,2,size=(50,64,1)).astype('float32');torch.manual_seed(seed+70)
    env=ActionSquashEnvBase('relu(1-abs(u))' if kind==1 else 'relu(abs(u)-1)',0.,True)
    noises=[]
    for t in range(50):
        state=torch.from_numpy(states[-1]);action=torch.from_numpy(actions[t]);before=torch.get_rng_state()
        out=env.step_single(state,action,torch.from_numpy(alpha));after=torch.get_rng_state()
        torch.set_rng_state(before);noise=torch.randn_like(state).numpy()*.01;torch.set_rng_state(after)
        states.append(out.numpy());noises.append(noise)
    torch.manual_seed(seed+71);common_probe_noise=torch.randn(64,2).numpy()*.01
    torch.manual_seed(seed+72);query_noise=torch.randn(64,2).numpy()*.01
    probes=[]
    for u in [0.,1.5]:
        g=max(0.,1-abs(u)) if kind==1 else max(0.,abs(u)-1)
        probes.append(np.c_[g*alpha[:,0],np.zeros(64,dtype='float32')]+common_probe_noise)
    query_u=0. if kind==1 else 1.5;g=1. if kind==1 else .5
    query=np.c_[g*alpha[:,0],np.zeros(64,dtype='float32')]+query_noise
    f=OUT/f'action_squash_{kind}.npz';np.savez_compressed(f,alpha=alpha,states=np.stack(states),actions=actions,noise=np.stack(noises),probe=np.stack(probes),probe_noise=common_probe_noise,query=query,query_noise=query_noise,query_u=np.asarray(query_u),seed=np.asarray(seed))
    # Independent double recurrence from the exact saved noise and complete data.
    A=np.array([[.8,.2],[-.2,.8]]);s=np.stack(states);g=np.maximum(0,1-np.abs(actions)) if kind==1 else np.maximum(0,np.abs(actions)-1)
    expected=s[:-1].astype('float64')@A.T+np.concatenate((g*alpha[None],np.zeros_like(g)),axis=-1)+np.stack(noises)
    error=float(np.max(np.abs(expected-s[1:])));assert error<1e-6
    rows.append(dict(kind=kind,seed=seed,path=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),independent_recurrence_error=error,systems=64))
receipt=dict(source_revision=verify_source(),protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),files=rows,context_sizes=[0,2,5,9],sampling='Complete prefixes; alpha-independent initial state; no random omitted history.',new_training=False)
(OUT/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
