"""Unmodified Round-2 data/model imports and shared paths."""
import json,sys
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'config.yaml').read_text())
SOURCE=(ROOT/CFG['source_experiment']).resolve()
sys.path.append(str(SOURCE/'src'))
from core import Store,Windows,RSSM,to_device,seed_all,H,F  # noqa: E402
OUT=ROOT/'outputs'; ARRAYS=OUT/'arrays'
HORIZONS=CFG['horizons']

def write_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))

def physical(y,store):
    return y*store.norm['state_std'][[0,2]]+store.norm['state_mean'][[0,2]]
