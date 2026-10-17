"""Source/data/dependency qualification, without a training update."""
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import traceback

R=pathlib.Path(__file__).resolve().parents[1]
P=R.parent
STORAGE=json.loads((P/'STORAGE_LOCATION_V1.json').read_text())
OUT=R/'outputs/native_probe_v2'
assert not OUT.exists();OUT.mkdir(parents=True)
def save(obj): (OUT/'probe.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
source=P/'upstream/TSWM-Official-Repo'
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'status':'STARTED','upstream_commit':'2e5e48540ff0da48e8192163969a6c58df1d5a3e','formal_training_updates':0}
save(result)
try:
    pyfiles=sorted(source.rglob('*.py'))
    records=[{'path':p.relative_to(source).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in pyfiles if '__pycache__' not in p.parts]
    (OUT/'source_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    result['packages']={name:(importlib.metadata.version({'sklearn':'scikit-learn','pywt':'PyWavelets'}.get(name,name)) if importlib.util.find_spec(name.replace('-','_')) else None) for name in ['torch','numpy','pandas','sklearn','einops','pyarrow','openpyxl','pywt']}
    for path in [source,source/'rq2_latent_vs_ob',source/'rq2_latent_vs_ob/common',source/'RQ3_fusion_benchmark',source/'RQ4_action_encoding',source/'RQ7_causal_fusion']:
        sys.path.insert(0,str(path))
    import torch
    torch.set_num_threads(8)
    result['cuda_available']=torch.cuda.is_available()
    assert result['cuda_available']
    result['gpu_name']=torch.cuda.get_device_name(0)
    result['gpu_runtime']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
    from observational_space.train import Trainer,TrainConfig
    from observational_space.engine import forward_loss
    root=pathlib.Path(STORAGE['data_root'])/'greenhouse3_original_v1/experiment_1/TimeSeries'
    cache=pathlib.Path(STORAGE['cache_root'])/'cycle53/native_windows_v1'
    cfg=TrainConfig(dataset='greenhouse',model='CrossLinear',context_length=256,horizon=16,stride=80,batch_size=128,epochs=100,patience=20,lr=1e-3,grad_clip=1.,loss='mse',seed=0,data_seed=0,device='cuda',root=str(root),cache_dir=str(cache),wandb=False)
    trainer=Trainer(cfg)
    assert len(trainer.train_ds)==815 and len(trainer.val_ds)==163
    trainer.adapter.model.eval();trainer.embedder.eval()
    if trainer.adapter.CovariateFusion is not None:trainer.adapter.CovariateFusion.eval()
    batch=next(iter(trainer.val_loader))
    with torch.no_grad():
        loss,pred,target,stats=forward_loss(trainer.adapter,trainer.embedder,batch,trainer.schema.n_target,trainer.device,trainer.criterion)
    assert pred.shape==target.shape==(128,16,4)
    assert torch.isfinite(pred).all() and torch.isfinite(loss)
    result.update(status='QUALIFIED_NATIVE_SOURCE_DATA_AND_FORWARD',train_windows=len(trainer.train_ds),val_windows=len(trainer.val_ds),output_shape=list(pred.shape),initial_loss=float(loss),predictive_params=trainer.n_params,source_files=len(records),formal_training_updates=0)
    save(result)
    print(json.dumps(result),flush=True)
except BaseException as error:
    result.update(status='FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(result);raise

