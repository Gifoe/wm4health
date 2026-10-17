"""TRAIN-only native RQ2/RQ3 equivalence and complete checkpoint restoration."""
from native53 import *
import torch
from dataclasses import replace
from latent_space.train import TrainConfig,Trainer
from fusion_train import FusionTrainer
from common.gpu_dataset import prepare_gpu

torch.set_num_threads(8)
out=R/'outputs/concat_qualification_v1';assert not out.exists();out.mkdir()
cfg=TrainConfig(dataset='greenhouse',model='CrossLinear',ae_dir=str(CODEC),codec='ae',ae_mode='frozen',context_length=256,horizon=16,stride=80,batch_size=128,epochs=100,patience=20,lr=1e-3,grad_clip=1.,loss='mse',seed=424253,data_seed=0,device='cpu',root=str(DATA),cache_dir=str(CACHE),max_train_batches=1,wandb=False)
a=FusionTrainer(cfg,'concat');b=Trainer(replace(cfg))
def canonical(state):
    return {(('fusion' if key=='fusion_mlp' else key),name):tensor for key,values in state.items() for name,tensor in values.items()}
sa=canonical(a._state_dict());sb=canonical(b.state_dict());assert set(sa)==set(sb)
assert all(torch.equal(sa[key],sb[key]) for key in sa)
rng=torch.get_rng_state().clone();at=a._epoch(a.train_loader,train=True)
torch.set_rng_state(rng);bt=b.train_epoch()
sa=canonical(a._state_dict());sb=canonical(b.state_dict())
max_weight_error=max(float((sa[k]-sb[k]).abs().max()) for k in sa)
assert max_weight_error<1e-7 and abs(at-bt['total'])<1e-7,(max_weight_error,at,bt)

sys.path.insert(0,str(UP/'RQ6_intervention/scripts'))
import probe_intervention as probe
from replay53 import load_compatible_cell
probe.AE_DIR=CODEC;probe.DEVICE='cpu'
data=prepare_gpu('greenhouse',context_length=256,horizon=16,stride=80,val_ratio=.2,seed=0,root=DATA,device='cpu',cache_dir=CACHE)
# Existing selected checkpoint is used on TRAIN only, not the transfer batch.
bank={'n':len(data.train),'tensors':data.train.tensors,'schema':vars(data.schema),'combo_to_id':None}
checkpoint=CKPT/'CrossLinear__ae_concat__seed0.pt'
cell,receipt=load_compatible_cell(probe,'CrossLinear','concat',0,checkpoint,bank,data.schema)
batch={k:v[:4] for k,v in data.train.tensors.items()}
with torch.no_grad():
    f=probe.Forward(cell,batch,'concat');pred=f.predict(batch['continuous_future'],batch['categorical_future'],batch['exog_future'])
    from latent_space.engine import forward_loss
    direct=forward_loss(cell.adapter,cell.embedder,cell.target_ae,batch,cell.n_latent,4,'cpu',torch.nn.MSELoss())[1]
    error=float((pred-direct).abs().max());assert error<1e-7
result={'utc':now(),'status':'PASS_NATIVE_CONCAT_TRAINING_AND_CHECKPOINT_FORMAT','qualification_updates':2,'qualification_updates_used_in_formal_fits':False,'test_or_transfer_used':False,'initial_native_states_identical':True,'one_update_max_weight_error':max_weight_error,'one_update_loss_error':abs(at-bt['total']),'native_probe_vs_training_forward_error':error,'restore_receipt':receipt,'source_files_rechecked':check_source()}
dump(out/'qualification.json',result);print(json.dumps(result),flush=True)
