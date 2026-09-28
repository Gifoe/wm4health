"""Reuse the exact Round-2 prospective RSSM and Round-1 case arrays."""
import json, sys
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'config.yaml').read_text())
SOURCE=(ROOT/CFG['source_experiment']).resolve()
ROUND3=(ROOT/CFG['round3_experiment']).resolve()
sys.path.insert(0,str(SOURCE/'src'))
from core import Store,Windows,RSSM,to_device,seed_all,F,H
from train import loss_fn,validate
OUT=ROOT/'outputs'
ARRAYS=OUT/'arrays'
R3A=ROUND3/'outputs/arrays'
OUT.mkdir(exist_ok=True)
ARRAYS.mkdir(exist_ok=True)

def arr(split,name):
    return np.load(R3A/f'{split}_{name}.npy',mmap_mode='r')

def own(split,name):
    return np.load(ARRAYS/f'{split}_{name}.npy',mmap_mode='r')

def jwrite(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
