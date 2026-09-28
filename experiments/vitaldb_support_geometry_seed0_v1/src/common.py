"""Frozen Round-2 model/data imports and shared experiment paths."""
import json,sys
from pathlib import Path
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'config.yaml').read_text())
SOURCE=(ROOT/CFG['source_experiment']).resolve()
sys.path.insert(0,str(SOURCE/'src'))
from core import Store,Windows,RSSM,to_device,seed_all,F,H  # noqa: E402

OUT=ROOT/'outputs'
ARRAYS=OUT/'arrays'

def jwrite(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def model_for(seed,store,device):
    import torch
    model=RSSM().to(device)
    model.set_action_stats(store)
    ck=torch.load(OUT/'checkpoints'/f'seed{seed}.pt',map_location=device,weights_only=False)
    model.load_state_dict(ck['model']);model.eval()
    return model
