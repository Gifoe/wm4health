"""Small native RP parameter trees as exact typed JSON arrays for audit replay."""
import numpy as np
def encode(tree):
 if isinstance(tree,tuple):return {'kind':'tuple','items':[encode(v) for v in tree]}
 if isinstance(tree,list):return {'kind':'list','items':[encode(v) for v in tree]}
 if isinstance(tree,dict):return {'kind':'dict','items':[[k,encode(v)] for k,v in tree.items()]}
 a=np.asarray(tree);assert a.dtype.kind in 'fiu' and np.isfinite(a).all()
 return {'kind':'array','dtype':a.dtype.str,'shape':list(a.shape),'values':a.tolist()}
def decode(card):
 kind=card['kind']
 if kind=='tuple':return tuple(decode(v) for v in card['items'])
 if kind=='list':return [decode(v) for v in card['items']]
 if kind=='dict':return {k:decode(v) for k,v in card['items']}
 assert kind=='array';a=np.asarray(card['values'],dtype=np.dtype(card['dtype']));assert list(a.shape)==card['shape'] and np.isfinite(a).all();return a
