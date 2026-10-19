"""New exact controlled-system diagnostic; NOT a DoTime suite reproduction."""
import hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def simulate(u,x0,a=.94,b=.06,noise=None):
    out=np.empty_like(u,dtype=np.float64);x=x0.astype(np.float64).copy()
    for t in range(u.shape[1]):
        x=a*x+b*u[:,t]+(noise[:,t] if noise is not None else 0)
        out[:,t]=x
    return out
def generate(n,seed,cfg,excitation='smooth'):
    r=np.random.default_rng(seed);L,H=cfg['history'],cfg['horizon'];T=L+H+128
    rho=cfg['action_autocorrelation'] if excitation=='smooth' else 0.
    u=np.empty((n,T));u[:,0]=r.normal(size=n)
    for t in range(1,T):u[:,t]=rho*u[:,t-1]+np.sqrt(1-rho*rho)*r.normal(size=n)
    e=r.normal(scale=cfg['process_noise_std'],size=(n,T))
    x=simulate(u,r.normal(size=n),cfg['state_coefficient'],cfg['action_coefficient'],e)
    h=np.stack([x[:,128:128+L],u[:,128:128+L]],-1).astype('float32')
    future=u[:,-H:,None].astype('float32');y=x[:,-H:,None].astype('float32')
    return {'history':h,'future':future,'target':y,'noise':e[:,-H:].astype('float32')}
def split(name,cfg,excitation='smooth'):
    spec={'train':(cfg['train_episodes'],0),'validation':(cfg['validation_episodes'],1),
          'discovery':(cfg['discovery_episodes'],2),'confirmation':(cfg['confirmation_episodes'],3)}
    n,offset=spec[name];return generate(n,cfg['data_seed']+offset*100003,cfg,excitation)
def pulse_truth(data,cfg,pos,amplitude=1.):
    # Exact expected effect; identical noise and initial state cancel in both arms.
    H=cfg['horizon'];effect=np.zeros((H,1),dtype=np.float32)
    effect[pos:,0]=amplitude*cfg['action_coefficient']*cfg['state_coefficient']**np.arange(H-pos)
    return np.broadcast_to(effect,(len(data['history']),H,1)).copy()
