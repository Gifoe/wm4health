"""Qualified data-copy compatibility, never edits upstream or changes windows."""
from native62 import *
def install():
 util=importlib.import_module('ts_benchmark.baselines.utils');Base=util.DatasetForTransformer;old=Base.__getitem__
 def contiguous_getitem(self,index):
  s=index;e=s+self.history_length;r=e-self.label_length;end=r+self.label_length+self.prediction_length
  values=[self.dataset[s:e].values,self.dataset[r:end].values,self.data_stamp[s:e],self.data_stamp[r:end]]
  # Explicit independent contiguous float32 arrays; matches native torch.tensor(..., float32).
  return tuple(torch.from_numpy(np.array(v,dtype=np.float32,order='C',copy=True)) for v in values)
 Base.__getitem__=contiguous_getitem
 return Base,old,contiguous_getitem
