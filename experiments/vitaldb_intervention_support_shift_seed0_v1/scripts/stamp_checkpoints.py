"""Attach immutable support-manifest identity to completed training checkpoints."""
import hashlib
from pathlib import Path
import numpy as np,torch
p=Path(__file__).resolve().parents[1]
assert all((p/f'outputs/checkpoints/{c}_seed{seed}.pt').exists() for c in 'LZR' for seed in range(5))
v=hashlib.sha256(np.load(p/'outputs/arrays/val_trainlike.npy').tobytes()).hexdigest()
for c in 'LZR':
    h=hashlib.sha256(np.load(p/f'outputs/arrays/train_keep_{c}.npy').tobytes()).hexdigest()
    for seed in range(5):
        path=p/f'outputs/checkpoints/{c}_seed{seed}.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        ck['train_pool_sha256']=h;ck['val_trainlike_sha256']=v
        torch.save(ck,path)
print('CHECKPOINT_MANIFEST_STAMPS_COMPLETE')
