"""Official original-mode DLinear; original numerical bodies with recording hooks only."""
import sys,pathlib,json,hashlib,time,difflib
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=pathlib.Path('D:/baseline_reproduction_seed2021_v1')
sys.path.insert(0,str(BASE/'scripts'))
import common as base
import numpy as np,torch
seed=int(sys.argv[1]);H=int(sys.argv[2]);assert seed in (42,43,44) and H in (336,720)
cfg=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
ART=pathlib.Path(cfg['artifact_root'])/ROOT.name/'raw_control_v1';ART.mkdir(exist_ok=True,parents=True)
tag=f'Electricity_DLinear_H{H}_seed{seed}';out=ART/tag;out.mkdir(exist_ok=False)
prior=BASE/f'results/latenttsf/Electricity_DLinear_H{H}_seed42/config.json'
c=json.loads(prior.read_text(encoding='utf-8-sig'));args=dict(c['arguments'])
args.update(seed=seed,model_id='raw_'+tag,load_pretrained_ae=0,result_csv=str(out/'official_result.csv'),result_txt=str(out/'official_result.txt'),checkpoints=str(out/'checkpoints'))
SRC=BASE/'upstream/LatentTSF';sys.path.insert(0,str(SRC));native=SRC/'my_train.py'
source=native.read_text(encoding='utf-8-sig');source_sha=base.sha(native)
sys.argv=[str(native)]+[v for k,x in args.items() for v in ['--'+k,str(x)]]
base.write_json(out/'config.json',dict(arguments=args,flags=[],source_commit=c['source_commit'],data_sha256=c['data_sha256'],ae_checkpoint_sha256=None,registered_control_protocol_sha256=base.sha(ROOT/'RAW_MODE_CONTROL_PROTOCOL_V1.md')))
state=dict(history=[],start=time.time())
def setup(g):
    assert g['encoder'] is None and not g['args'].use_latent and g['args'].step==60
    state['parameters']=sum(x.numel() for x in g['model'].parameters());assert state['parameters']==(484512 if H==336 else 1038240)
    state['train_start']=time.time();state['steps_per_epoch']=len(g['train_loader'])
    base.write_json(out/'resolved_config.json',vars(g['args']))
    return 0
def record(g):
    state['history'].append(dict(epoch=g['epoch']+1,valid_loss=float(g['valid_loss'])))
    base.write_csv(out/'history.csv',state['history'])
def finish(g):state.update(train_seconds=time.time()-state['train_start'],checkpoint=pathlib.Path(g['path'])/'checkpoint.pth',eval_start=time.time())
def metric(pred,true):
    from utils.metrics import metric as official
    assert pred.shape==true.shape and pred.shape[1:]==(H,321) and np.isfinite(pred).all() and np.isfinite(true).all()
    np.savez_compressed(out/'predictions.npz',pred=pred,true=true)
    ss=sa=0.;n=0
    for j in range(0,len(pred),16):
        e=pred[j:j+16].astype(np.float64)-true[j:j+16];ss+=np.square(e).sum(dtype=np.float64);sa+=abs(e).sum(dtype=np.float64);n+=e.size
    values=official(pred,true);assert abs(values[1]-ss/n)<1e-5 and abs(values[0]-sa/n)<1e-5
    state.update(mse=ss/n,mae=sa/n,test_windows=len(pred));return values
patched=source.replace('wandb.Settings(start_method="thread")','wandb.Settings()')
anchor='        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")'
changes=[(anchor,'        setting = _audit_run_id\n'+anchor),
         ('        for epoch in range(args.train_epochs):','        _audit_setup(locals())\n        for epoch in range(args.train_epochs):'),
         ('            early_stopping(valid_loss, model, path)','            early_stopping(valid_loss, model, path)\n            _audit_record(locals())'),
         ('        # Total training time','        _audit_finish(locals())\n        # Total training time'),
         ('    mae, mse, rmse, mape, mspe = metric(preds, trues)','    mae, mse, rmse, mape, mspe = _audit_metric(preds, trues)')]
for a,b in changes:assert patched.count(a)==1,(a,patched.count(a));patched=patched.replace(a,b)
patch=''.join(difflib.unified_diff(source.splitlines(True),patched.splitlines(True),fromfile='official/my_train.py',tofile='raw_audit/my_train.py'));(out/'recording_hooks.patch').write_bytes(patch.encode('utf-8'))
import os
os.chdir(out)
ns=dict(__name__='__main__',__file__=str(native),_audit_run_id='raw_'+tag,_audit_setup=setup,_audit_record=record,_audit_finish=finish,_audit_metric=metric)
exec(compile(patched,str(native)+'::raw_recording_hooks','exec'),ns)
assert base.sha(native)==source_sha
result=dict(model='official_original_DLinear',dataset='Electricity',horizon=H,seed=seed,reproduced_mse=state['mse'],reproduced_mae=state['mae'],best_epoch=min(state['history'],key=lambda x:x['valid_loss'])['epoch'],epochs=len(state['history']),num_parameters=state['parameters'],frozen_ae_parameters=0,checkpoint_path=str(state['checkpoint']),checkpoint_sha256=base.sha(state['checkpoint']),predictions_sha256=base.sha(out/'predictions.npz'),data_sha256=c['data_sha256'],source_commit=c['source_commit'],train_seconds=state['train_seconds'],test_windows=state['test_windows'],steps_per_epoch=state['steps_per_epoch'],actual_training_updates=state['steps_per_epoch']*len(state['history']),config_path=str(out/'config.json'),source_native_unchanged=True,new_method=False)
base.write_json(out/'metrics.json',result);base.write_json(out/'SOURCE_UNCHANGED_AFTER_V1.json',dict(upstream_unchanged=True,source_sha256=source_sha))
print(json.dumps(dict(status='RAW_MODE_CONTROL_COMPLETE',seed=seed,horizon=H,mse=state['mse'],mae=state['mae'])),flush=True)
