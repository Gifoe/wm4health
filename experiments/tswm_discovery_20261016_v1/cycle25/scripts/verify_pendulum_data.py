"""Independent NumPy integrator; no native environment imported."""
import hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];data=np.load(ROOT/'outputs/data/pendulum.npz')
def kernel(state,u,gains):
    state=np.array(state,dtype=np.float64).copy()
    for _ in range(2):
        th,vel=state;angle=(th+np.pi)%(2*np.pi)-np.pi;q=3 if angle < -np.pi/2 else 2 if angle<0 else 1 if angle<np.pi/2 else 0
        vel2=vel+.05*(15*np.sin(th)+3*np.clip(u,-2,2)*gains[q]);th2=th+.05*vel2;state=np.array([th2,np.clip(vel2,-8,8)])
    return state,np.array([np.cos(state[0]),np.sin(state[0]),state[1]])
errors=[]
for i,g in enumerate(data['gains']):
    for t in range(32):
        s,y=kernel(data['raw_states'][t,i],data['actions'][t,i,0],g);errors.extend([abs(s-data['raw_states'][t+1,i]).max(),abs(y-data['states'][t+1,i]).max()])
    for j,u in enumerate([0.,1.5]):
        _,y=kernel([np.pi/4,0],u,g);errors.append(abs(y-data['probe'][j,i]).max())
    for q,th in enumerate(data['query_theta']):
        _,y=kernel([th,0],1.5,g);errors.append(abs(y-data['query'][i,q]).max())
maximum=float(max(errors));assert maximum<1e-12,maximum
assert np.isfinite(data['states']).all() and data['states'].shape==(33,64,3)
r=dict(status='pass',recurrence_and_reset_comparisons=len(errors),max_abs_error=maximum,data_sha256=hashlib.sha256((ROOT/'outputs/data/pendulum.npz').read_bytes()).hexdigest(),verifier_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
(ROOT/'outputs/data/independent_verification.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))
