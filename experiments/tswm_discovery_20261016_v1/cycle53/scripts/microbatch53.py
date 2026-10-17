"""Effective-batch-preserving accumulation for native sample-separable PatchTST."""
import torch

def install_microbatch_epoch(trainer,microbatch=64):
    original=trainer._epoch
    modules=[trainer.adapter.model,trainer.embedder,trainer.target_ae]
    if trainer.adapter.CovariateFusion is not None:modules.append(trainer.adapter.CovariateFusion)
    if trainer.fusion is not None:modules.append(trainer.fusion)
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for parent in modules for m in parent.modules())
    def epoch(loader,train):
        if not train:return original(loader,False)
        trainer._set_train(True)
        total=0.;seen=0
        for i,batch in enumerate(loader):
            if trainer.cfg.max_train_batches is not None and i>=trainer.cfg.max_train_batches:break
            n=len(batch['target_history']);trainer.optimizer.zero_grad();batch_loss=0.
            for start in range(0,n,microbatch):
                part={k:v[start:start+microbatch] for k,v in batch.items()}
                count=len(part['target_history'])
                loss,_,_,stats=trainer._fwd(part)
                assert stats['aux']==0.,'Qualification covers additive per-sample MSE, not coupled auxiliary losses'
                (loss*(count/n)).backward();batch_loss+=float(loss.item())*(count/n)
            if trainer.cfg.grad_clip:torch.nn.utils.clip_grad_norm_(trainer._trainable_params(),trainer.cfg.grad_clip)
            trainer.optimizer.step();total+=batch_loss;seen+=1
        return total/max(seen,1)
    trainer._epoch=epoch
