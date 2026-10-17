"""Check lossless saved-tensor CPU storage before resuming native batch128 fits."""
from native53 import *
import gc
import time
import traceback
import numpy as np
import torch
from dataclasses import replace
from latent_space.train import TrainConfig
from fusion_train import FusionTrainer
from microbatch53 import install_microbatch_epoch

torch.set_num_threads(8)
out=R/'outputs/microbatch_qualification_v2';assert not out.exists();out.mkdir()
dump(out/'status.json',{'status':'RUNNING','pid':os.getpid(),'utc':now()})
try:
    cfg=TrainConfig(dataset='greenhouse',model='PatchTST',ae_dir=str(CODEC),codec='ae',ae_mode='frozen',context_length=256,horizon=16,stride=80,batch_size=4,epochs=100,patience=20,lr=1e-3,grad_clip=1.,loss='mse',seed=535353,data_seed=0,device='cuda',root=str(DATA),cache_dir=str(CACHE),max_train_batches=1,wandb=False)
    a=FusionTrainer(cfg,'concat');b=FusionTrainer(replace(cfg),'concat')
    for trainer in [a,b]:
        for module in [trainer.adapter.model,trainer.embedder,trainer.target_ae,trainer.adapter.CovariateFusion]:
            if module is not None:
                for layer in module.modules():
                    if isinstance(layer,torch.nn.Dropout):layer.p=0.
    def flat(tr):return {(key,name):tensor for key,values in tr._state_dict().items() for name,tensor in values.items()}
    sa,sb=flat(a),flat(b);assert set(sa)==set(sb);assert all(torch.equal(sa[k],sb[k]) for k in sa)
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
    ordinary=a._epoch(a.train_loader,train=True)
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    install_microbatch_epoch(b,microbatch=2)
    offloaded=b._epoch(b.train_loader,train=True)
    sa,sb=flat(a),flat(b)
    max_weight_error=max(float((sa[k]-sb[k]).abs().max()) for k in sa)
    differences=sorted([{'group':k[0],'tensor':k[1],'max_abs_error':float((sa[k]-sb[k]).abs().max())} for k in sa],key=lambda v:v['max_abs_error'],reverse=True)
    a._set_train(False);b._set_train(False)
    samples={k:v[:8] for k,v in a.train_ds.tensors.items()}
    with torch.no_grad():
        pa=a._fwd(samples)[1];pb=b._fwd(samples)[1]
        functional_error=float((pa-pb).abs().max())
    diagnostic={'max_weight_error':max_weight_error,'top_parameter_differences':differences[:12],'post_update_prediction_max_error':functional_error,'training_loss_error':abs(ordinary-offloaded)}
    dump(out/'numerical_diagnostic.json',diagnostic)
    assert functional_error<2e-5 and abs(ordinary-offloaded)<1e-6,diagnostic
    del pa,pb,samples
    del sa,sb,a,b;gc.collect();torch.cuda.empty_cache()
    full=FusionTrainer(replace(cfg,batch_size=128),'concat')
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    install_microbatch_epoch(full,microbatch=64)
    full_loss=full._epoch(full.train_loader,train=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()
    assert np.isfinite(full_loss) and peak<28*1024**3,(full_loss,peak)
    result={'status':'PASS_NATIVE_EFFECTIVE_BATCH_MICROBATCHING','utc':now(),'small_batch':4,'small_batch_update_max_weight_error':max_weight_error,'post_update_prediction_max_error':functional_error,'top_parameter_differences':differences[:6],'small_batch_loss_error':abs(ordinary-offloaded),'full_native_batch':128,'full_native_one_update_loss':full_loss,'full_native_one_update_seconds':elapsed,'peak_cuda_allocated_bytes':peak,'formal_model_weights_used':False,'qualification_updates':3,'qualification_warmstarts_formal_fits':False,'source_files_rechecked':check_source(),'runtime_change':'effective batch128, microbatch64, one optimizer/clipping operation per full batch; no AMP','deterministic_update_check_dropout_disabled':True,'dropout_caveat':'Formal dropout masks remain iid but CUDA RNG draw ordering differs from a single128-sample forward. This is a disclosed runtime adaptation, not bitwise original-run reproduction.'}
    dump(out/'qualification.json',result);dump(out/'status.json',{'status':result['status'],'utc':now(),'pid':os.getpid()});print(json.dumps(result),flush=True)
except BaseException as error:
    dump(out/'status.json',{'status':'FAILED','utc':now(),'pid':os.getpid(),'error':str(error),'traceback':traceback.format_exc()});raise
