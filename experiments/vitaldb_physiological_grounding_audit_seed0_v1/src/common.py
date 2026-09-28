"""Frozen source model/data and patient-balanced statistics."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'config.yaml').read_text())
SOURCE=(ROOT/CFG['source_experiment']).resolve()
sys.path.append(str(SOURCE/'src'))
from core import Store,Windows,RSSM,to_device,seed_all,H,F  # noqa: E402
OUT=ROOT/'outputs';ARRAYS=OUT/'arrays';HORIZONS=CFG['horizons']

def write_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))

def load(split,name,mmap=True):
    return np.load(ARRAYS/f'{split}_{name}.npy',mmap_mode='r' if mmap else None)

def save_array(split,name,shape,dtype='float32'):
    return np.lib.format.open_memmap(ARRAYS/f'{split}_{name}.npy',mode='w+',dtype=dtype,shape=shape)

def patient_weights(subject,valid):
    ids,count=np.unique(subject[valid],return_counts=True)
    lookup=dict(zip(ids,1/count))
    w=np.zeros(len(subject),np.float64)
    w[valid]=[lookup[s] for s in subject[valid]]
    return w/w.sum() if w.sum() else w

def weighted_r2(y,p,subject,valid):
    valid=valid&np.isfinite(y)&np.isfinite(p)
    if valid.sum()<2:return np.nan
    w=patient_weights(subject,valid)[valid];a=y[valid];b=p[valid]
    mean=np.sum(w*a);den=np.sum(w*(a-mean)**2)
    return float(1-np.sum(w*(a-b)**2)/den) if den>0 else np.nan

def weighted_metrics(y,p,subject,valid):
    valid=valid&np.isfinite(y)&np.isfinite(p)
    if valid.sum()<3:return dict(r2=np.nan,mae=np.nan,pearson=np.nan,patients=0,windows=int(valid.sum()))
    w=patient_weights(subject,valid)[valid];a=y[valid];b=p[valid]
    my=np.sum(w*a);mp=np.sum(w*b)
    cov=np.sum(w*(a-my)*(b-mp));vy=np.sum(w*(a-my)**2);vp=np.sum(w*(b-mp)**2)
    return dict(r2=weighted_r2(y,p,subject,valid),mae=float(np.sum(w*np.abs(a-b))),
                pearson=float(cov/np.sqrt(vy*vp)) if vy*vp>0 else np.nan,
                patients=int(np.unique(subject[valid]).size),windows=int(valid.sum()))

def patient_bootstrap_stat(y,p,subject,valid,stat='r2',reps=1000):
    valid=valid&np.isfinite(y)&np.isfinite(p)
    ids=np.unique(subject[valid]);rng=np.random.default_rng(0)
    if len(ids)<2:return np.nan,np.nan
    # Patient sufficient statistics make 1000 R2 resamples cheap.
    sums=[]
    for s in ids:
        a=y[valid&(subject==s)].astype(np.float64);b=p[valid&(subject==s)].astype(np.float64)
        sums.append([a.mean(),np.mean(a*a),b.mean(),np.mean(b*b),np.mean(a*b),np.mean((a-b)**2),np.mean(np.abs(a-b))])
    q=np.asarray(sums);draw=q[rng.integers(len(ids),size=(reps,len(ids)))].mean(1)
    if stat=='r2':
        den=draw[:,1]-draw[:,0]**2;value=1-draw[:,5]/den
    elif stat=='mae':value=draw[:,6]
    else:
        den=np.sqrt((draw[:,1]-draw[:,0]**2)*(draw[:,3]-draw[:,2]**2))
        value=(draw[:,4]-draw[:,0]*draw[:,2])/den
    return tuple(np.nanquantile(value,[.025,.975]).astype(float))
