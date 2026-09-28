"""Extract identical-window VAL/TEST predictions for all four ensembles."""
import hashlib,sys
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from common import *

@torch.no_grad()
def predict(model,loader,store,device):
    model.eval();out=[]
    scale=store.norm['state_std'][[0,2]];offset=store.norm['state_mean'][[0,2]]
    for j,b in enumerate(loader):
        pred,_,_=model(to_device(b,device),'true')
        out.append((pred.cpu().numpy()*scale+offset).astype('float32'))
    return np.concatenate(out)

def main():
    seed_all(0);store=Store();device='cuda' if torch.cuda.is_available() else 'cpu'
    conditions=tuple(sys.argv[1:]) if len(sys.argv)>1 else ('F','L','Z','R')
    assert set(conditions)<=set('FLZR')
    # Independent numerical re-inference audits the reused checkpoint, while
    # the full saved arrays are copied byte-for-byte from Round 3.
    audit_ds=Windows(store,'test');audit_loader=DataLoader(Subset(audit_ds,list(range(512))),batch_size=512,
                                                         num_workers=0,pin_memory=True)
    audit={}
    for seed in CFG['seeds']:
        ck=torch.load(ROUND3/'outputs/checkpoints'/f'seed{seed}.pt',map_location=device,weights_only=False)
        model=RSSM().to(device);model.set_action_stats(store);model.load_state_dict(ck['model'])
        fresh=predict(model,audit_loader,store,device)
        original=np.asarray(arr('test',f'pred_seed{seed}'))[:512]
        audit[str(seed)]=float(np.max(np.abs(fresh-original)))
        assert audit[str(seed)]<1e-4
        del model;torch.cuda.empty_cache()
    for split in ('val','test'):
        ds=Windows(store,split);assert np.array_equal(ds.indices,arr(split,'index'))
        loader=DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True)
        for condition in conditions:
            for seed in CFG['seeds']:
                path=ARRAYS/f'{split}_{condition}_pred_seed{seed}.npy'
                if path.exists():continue
                if condition=='F':
                    source=R3A/f'{split}_pred_seed{seed}.npy'
                    # Link or copy the exact full-support predictions; no inference
                    # can silently change the original control.
                    np.save(path,np.load(source,mmap_mode='r'))
                    assert np.array_equal(np.load(path,mmap_mode='r'),np.load(source,mmap_mode='r'))
                    continue
                ck=torch.load(OUT/'checkpoints'/f'{condition}_seed{seed}.pt',map_location=device,weights_only=False)
                model=RSSM().to(device);model.set_action_stats(store);model.load_state_dict(ck['model'])
                p=predict(model,loader,store,device);np.save(path,p)
                assert p.shape==(len(ds),F,2)
                del model;torch.cuda.empty_cache()
                print('PREDICTED',split,condition,seed,flush=True)
    checks={'full_support_seed0_checkpoint_sha256':hashlib.sha256((OUT/'checkpoints/F_seed0.pt').read_bytes()).hexdigest(),
            'round3_seed0_checkpoint_sha256':hashlib.sha256((ROUND3/'outputs/checkpoints/seed0.pt').read_bytes()).hexdigest(),
            'independent_reinference_max_abs_difference_by_seed_first512_test':audit,
            'all_full_support_predictions_bitwise_equal':all(np.array_equal(
               own(split,f'F_pred_seed{seed}'),arr(split,f'pred_seed{seed}'))
               for split in ('val','test') for seed in CFG['seeds'])}
    assert checks['full_support_seed0_checkpoint_sha256']==checks['round3_seed0_checkpoint_sha256']
    if 'F' in conditions:assert checks['all_full_support_predictions_bitwise_equal']
    jwrite('full_support_reuse_check.json',checks)
    print('PREDICTIONS_COMPLETE',flush=True)

if __name__=='__main__':main()
