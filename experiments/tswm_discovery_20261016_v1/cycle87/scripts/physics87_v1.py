import numpy as np
FAMILIES=('duffing','stuart_landau')
def parameters(family,n,rng):
 if family=='duffing':return np.column_stack([rng.uniform(-.5,1.5,n),rng.uniform(.5,1.5,n),rng.uniform(.1,.8,n),rng.uniform(-.3,.3,n),rng.uniform(.6,1.6,n)])
 if family=='stuart_landau':return np.column_stack([rng.uniform(-.4,1.4,n),rng.uniform(.3,1.2,n),rng.uniform(.5,2,n),rng.uniform(.6,1.6,n),np.zeros(n)])
 raise ValueError(family)
def field(family,p,x):
 # p: [B,5], x:[B,...,2]. Broadcast parameter axes explicitly.
 p=p.reshape((len(p),)+(1,)*(x.ndim-2)+(5,));a=x[...,0];b=x[...,1]
 if family=='duffing':return np.stack((b,p[...,0]*a-p[...,1]*a**3-p[...,2]*b+p[...,3]),axis=-1)*p[...,4,None]
 r2=a*a+b*b;c=p[...,0]-p[...,1]*r2
 return np.stack((c*a-p[...,2]*b,p[...,2]*a+c*b),axis=-1)*p[...,3,None]
def trajectories(family,p,initial,steps=200,dt=.05,substeps=4):
 x=np.array(initial,dtype=np.float64,copy=True);out=np.empty((len(p),x.shape[1],steps+1,2),dtype=np.float32);out[:,:,0]=x;h=dt/substeps
 for j in range(steps):
  for _ in range(substeps):
   k1=field(family,p,x);k2=field(family,p,x+h*k1/2);k3=field(family,p,x+h*k2/2);k4=field(family,p,x+h*k3)
   x=x+h*(k1+2*k2+2*k3+k4)/6
  out[:,:,j+1]=x
 assert np.isfinite(out).all(),(family,'Nonfinite bounded-family numerical integration')
 return out
def save(p,**arrays):
 with p.open('xb') as f:np.savez_compressed(f,**arrays)
