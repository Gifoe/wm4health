from common88_v1 import *
import numpy as np

DATA=Path(STORAGE['data_root'])/'cycle88';DATA.mkdir(parents=True,exist_ok=True)
PROJECTION=np.random.default_rng(2026880).normal(size=(9,16))/3

def observe(state):
    x,z=state[...,0],state[...,1]
    features=np.stack([x,z,x*x,z*z,x*z,np.sin(x),np.sin(z),np.cos(x),np.cos(z)],axis=-1)
    return features@PROJECTION

def rhs(state,action,theta,family):
    x,z,h=state.T;u,p=action.T;k,d,g,n,tau=theta.T
    if family==0:
        return np.stack([z,-k*x-n*x**3-(d+.6*h)*z+g*u,(.8*p-h)/tau],axis=-1)
    return np.stack([-.6*x+np.tanh(k*z+g*u)-.1*h*x,-d*z+np.tanh(-n*x+.5*u)+.25*h,(.8*p-h)/tau],axis=-1)

def advance(state,action,theta,family):
    state=state.copy();dt=.025
    for _ in range(4):
        k1=rhs(state,action,theta,family);k2=rhs(state+dt*k1/2,action,theta,family)
        k3=rhs(state+dt*k2/2,action,theta,family);k4=rhs(state+dt*k3,action,theta,family)
        state=state+dt/6*(k1+2*k2+2*k3+k4)
    assert np.isfinite(state).all()
    return state

def generate(seed,systems,trials):
    rng=np.random.default_rng(seed);all_state=[];all_actions=[];all_theta=[];all_family=[];all_system=[]
    for family in [0,1]:
        theta_sys=np.stack([rng.uniform(.6,1.3,systems),rng.uniform(.25,.65,systems),rng.uniform(.7,1.3,systems),rng.uniform(.3,.8,systems),rng.uniform(2.,5.,systems)],axis=-1)
        theta=np.repeat(theta_sys,trials,axis=0);n=len(theta)
        a=np.repeat(np.stack([rng.uniform(-1,1,(n,24)),rng.uniform(0,1,(n,24))],axis=-1),8,axis=1)
        s=np.zeros((n,193,3));s[:,0,:2]=rng.uniform(-.6,.6,(n,2));s[:,0,2]=rng.uniform(0,.8,n)
        for t in range(192):s[:,t+1]=advance(s[:,t],a[:,t],theta,family)
        all_state.append(s);all_actions.append(a);all_theta.append(theta);all_family.extend([family]*n);all_system.extend(family*10000+np.repeat(np.arange(systems),trials))
    state=np.concatenate(all_state);noise=rng.normal(0,.01,(len(state),193,16));obs=observe(state)+noise
    return dict(observations=obs.astype(np.float32),actions=np.concatenate(all_actions).astype(np.float32),states=state,parameters=np.concatenate(all_theta),family=np.array(all_family,dtype=np.int64),system_id=np.array(all_system,dtype=np.int64),observation_noise=noise)

def main():
    train=generate(2026881,32,8);val=generate(2026882,8,4)
    center=train['observations'].mean(axis=(0,1),dtype=np.float64);scale=np.maximum(train['observations'].std(axis=(0,1),dtype=np.float64),.05)
    assert set(train['system_id']).isdisjoint(set(val['system_id']+100000))
    files=[]
    for split,data in [('train',train),('val',val)]:
        data['observations']=((data['observations']-center)/scale).astype(np.float32)
        p=DATA/f'{split}_plan88_v1.npz'
        with p.open('xb') as f:np.savez_compressed(f,**data,center=center,scale=scale,projection=PROJECTION)
        files.append(dict(split=split,path=str(p),sha256=sha(p),trajectories=len(data['actions']),observations_shape=list(data['observations'].shape),systems_per_family=32 if split=='train' else 8,seed=2026881 if split=='train' else 2026882,state_abs_max=float(abs(data['states']).max())))
    write(O/'PLAN_DATA88_V1.json',dict(utc=utc(),files=files,source_rejected_trajectories=0,future_controls_exogenous=True,normalization='TRAIN-only16channelmean/std floor.05',protocol_sha256=sha(C/'PLAN_CAUSALITY_PROTOCOL88_V1.md'),hidden_state_and_parameter_features_used_by_models=False,held_out_eval_generated=False))
    log(dict(stage='plan_data_generated',files=files))

if __name__=='__main__':main()
