"""NumPy replay of the disclosed ReLU field / GRU and Heun history semantics."""
import numpy as np,math
def _calendar(t,phase):return np.stack((np.sin(2*math.pi*(phase+t)),np.cos(2*math.pi*(phase+t))),-1)
def _pack(t,value,phase):return np.concatenate((np.asarray(value).reshape(-1,1),_calendar(t,phase)),-1)
def _field(inputs,weights):
 indices=sorted(int(k.split('.')[2]) for k in weights if k.startswith('core.mlp.') and k.endswith('.weight'));x=inputs
 for j,index in enumerate(indices):
  x=x@weights[f'core.mlp.{index}.weight'].T+weights[f'core.mlp.{index}.bias']
  if j<len(indices)-1:x=np.maximum(x,0)
 return x[:,0]
def _sigmoid(x):return 1/(1+np.exp(-np.clip(x,-700,700)))
def replay(method,weights,history,phase,horizon):
 weights={k:np.asarray(v,dtype='float64') for k,v in weights.items()};history=np.asarray(history,dtype='float64')[:,:,0];phase=np.asarray(phase,dtype='float64');n=len(history)
 if method=='gru':
  hidden=np.zeros((n,32));wih=weights['core.gru.weight_ih_l0'];whh=weights['core.gru.weight_hh_l0'];bih=weights['core.gru.bias_ih_l0'];bhh=weights['core.gru.bias_hh_l0']
  def update(inputs,hidden):
   x=inputs@wih.T+bih;h=hidden@whh.T+bhh;xr,xz,xn=np.split(x,3,axis=-1);hr,hz,hn=np.split(h,3,axis=-1);reset=_sigmoid(xr+hr);gate=_sigmoid(xz+hz);candidate=np.tanh(xn+reset*hn);return (1-gate)*candidate+gate*hidden
  for j in range(96):hidden=update(_pack((j-94)/24,history[:,j],phase),hidden)
  value=history[:,-1]+(hidden@weights['core.linear.weight'].T+weights['core.linear.bias'])[:,0];values=[value]
  for lead in range(2,horizon+1):hidden=update(_pack(lead/24,value,phase),hidden);value=value+(hidden@weights['core.linear.weight'].T+weights['core.linear.bias'])[:,0];values.append(value)
  return np.stack(values,1)[:,:,None]
 states=np.empty((n,96+horizon));states[:,:96]=history;grid=np.arange(-95,horizon+1)/24;h=1/24;delays=weights.get('core.delays',np.array([]))
 for i in range(horizon):
  def field(t,value):
   pieces=[_pack(t,value,phase)]
   for tau in delays:
    s=t-tau;k=int(np.clip(np.searchsorted(grid,s,side='left')-1,0,94+i));fraction=(s-grid[k])/(grid[k+1]-grid[k]);past=states[:,k]+fraction*(states[:,k+1]-states[:,k]);pieces.append(_pack(s,past,phase))
   return _field(np.concatenate(pieces,-1),weights)
  current=states[:,95+i];t=i/24;k1=field(t,current);k2=field(t+h,current+h*k1);states[:,96+i]=current+h*(k1+k2)/2
 return states[:,96:][:,:,None]
