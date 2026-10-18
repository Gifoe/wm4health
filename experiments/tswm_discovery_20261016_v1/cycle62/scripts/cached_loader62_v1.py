"""Tensor cache avoids per-window NumPy/Torch interoperability."""
from native62 import *
def install():
 util=importlib.import_module('ts_benchmark.baselines.utils');Base=util.DatasetForTransformer;old=Base.__getitem__;original_init=Base.__init__
 def initialize(self,*args,**kwargs):
  original_init(self,*args,**kwargs)
  self._values62=torch.from_numpy(np.array(self.dataset.values,dtype=np.float32,order='C',copy=True)).clone()
  self._marks62=torch.from_numpy(np.array(self.data_stamp,dtype=np.float32,order='C',copy=True)).clone()
 def cached_getitem(self,index):
  s=index;e=s+self.history_length;r=e-self.label_length;end=r+self.label_length+self.prediction_length
  return (self._values62[s:e].clone(),self._values62[r:end].clone(),self._marks62[s:e].clone(),self._marks62[r:end].clone())
 Base.__init__=initialize;Base.__getitem__=cached_getitem
 return Base,old,cached_getitem
