"""Execute native upstream fits; wrappers record diagnostics without changing updates."""
from native53 import *
import argparse
import gc
import time
import traceback
import contextlib
from microbatch53 import install_microbatch_epoch

import faulthandler
faulthandler.enable(all_threads=True)
import torch

torch.set_num_threads(8)
p=argparse.ArgumentParser();p.add_argument('--job',choices=['codec','fit'],required=True);p.add_argument('--arm',choices=['obs','ae_concat','ae_gate','ae_gate_soft']);p.add_argument('--model',choices=['CrossLinear','PatchTST']);p.add_argument('--seed',type=int);args=p.parse_args()
check_source()
tag='shared_codec' if args.job=='codec' else f'{args.model}__{args.arm}__seed{args.seed}'
out=ART/'runtime_microbatch_v7'/tag
assert not out.exists(), 'Inspect/version an existing fit rather than overwrite'
out.mkdir(parents=True)
state=out/'state.json'
dump(state,{'status':'RUNNING','pid':os.getpid(),'tag':tag,'utc':now(),'protocol_sha256':sha(R/'PROTOCOL.md')})
started=time.perf_counter()
try:
    if args.job=='codec':
        from latent_space.train_ae import train_ae
        CODEC.mkdir(parents=True,exist_ok=True)
        ck=CODEC/'greenhouse__target.pt';assert not ck.exists()
        meta=train_ae('greenhouse',signal='target',context_length=256,horizon=16,d_model=32,d_ff=None,use_revin=False,epochs=100,patience=15,batch_size=128,lr=1e-3,loss='mse',val_ratio=.2,seed=0,device='cpu',root=str(DATA),options={},ckpt=ck)
        dump(out/'summary.json',{'job':'codec','native_meta':meta,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'elapsed_seconds':time.perf_counter()-started,'source_files_rechecked':check_source()})
    else:
        assert args.seed in range(5)
        assert torch.cuda.is_available()
        from dataclasses import asdict
        from observational_space.train import Trainer as ObsTrainer,TrainConfig as ObsConfig
        from latent_space.train import TrainConfig as LatentConfig
        from fusion_train import FusionTrainer
        from soft_train import SoftConstraintTrainer
        kw=dict(dataset='greenhouse',model=args.model,context_length=256,horizon=16,stride=80,batch_size=128,epochs=100,patience=20,lr=1e-3,weight_decay=0.,grad_clip=1.,loss='mse',lr_scheduler=True,lr_patience=5,val_ratio=.2,seed=args.seed,data_seed=0,num_workers=0,device='cuda',root=str(DATA),cache_dir=str(CACHE),wandb=False)
        if args.arm=='obs':
            config=ObsConfig(**kw);trainer=ObsTrainer(config)
        else:
            config=LatentConfig(**kw,ae_dir=str(CODEC),ae_mode='frozen',latent_scope='target',codec='ae')
            trainer=SoftConstraintTrainer(config,prior_split='default',rho=1.,seed=args.seed) if args.arm=='ae_gate_soft' else FusionTrainer(config,'concat' if args.arm=='ae_concat' else 'gate')
        assert len(trainer.train_ds)==815 and len(trainer.val_ds)==163
        dump(out/'config.json',asdict(config))
        if args.model=='PatchTST' and args.arm=='ae_concat':install_microbatch_epoch(trainer,microbatch=64)
        curve=[];last_train={}
        def rng_hash():return hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()
        if args.arm=='obs':
            orig_train,orig_val=trainer.train_epoch,trainer.val_loss
            def train_epoch():
                nonlocal_dummy=None
                last_train['rng_before']=rng_hash();last_train['begin']=time.perf_counter()
                value=orig_train();last_train['train']=value;return value
            def val_loss():
                value=orig_val()
                curve.append({'epoch':len(curve)+1,'train':last_train['train'],'val':value,'lr_before_scheduler':trainer.optimizer.param_groups[0]['lr'],'cpu_rng_before_train':last_train['rng_before'],'cpu_rng_after_val':rng_hash(),'elapsed_seconds':time.perf_counter()-started})
                dump(out/'curve.json',curve)
                return value
            trainer.train_epoch,trainer.val_loss=train_epoch,val_loss
        else:
            orig_epoch=trainer._epoch
            def epoch(loader,train):
                if train:last_train['rng_before']=rng_hash()
                value=orig_epoch(loader,train)
                if train:last_train['train']=value
                else:
                    curve.append({'epoch':len(curve)+1,'train':last_train['train'],'val':value,'lr_before_scheduler':trainer.optimizer.param_groups[0]['lr'],'cpu_rng_before_train':last_train['rng_before'],'cpu_rng_after_val':rng_hash(),'elapsed_seconds':time.perf_counter()-started})
                    dump(out/'curve.json',curve)
                return value
            trainer._epoch=epoch
        # Read-only snapshots do not advance either RNG or alter the native loop.
        init_state=trainer.state_dict() if args.arm=='obs' else trainer._state_dict()
        initial_path=out/'initial_state.pt';torch.save(init_state,initial_path)
        del init_state;gc.collect()
        torch.cuda.reset_peak_memory_stats()
        ck=CKPT/'runtime_microbatch_v7'/(tag+'.pt');assert not ck.exists()
        fit_begin=time.perf_counter()
        mode='effective_batch128_microbatch64' if args.model=='PatchTST' and args.arm=='ae_concat' else 'native_full_batch'
        if args.model=='PatchTST':
            assert json.loads((R/'outputs/microbatch_qualification_v2/qualification.json').read_text())['status']=='PASS_NATIVE_EFFECTIVE_BATCH_MICROBATCHING'
        if tag=='PatchTST__ae_concat__seed0':
            aborted=torch.load(ART/tag/'initial_state.pt',map_location='cpu',weights_only=False)
            fresh=trainer._state_dict()
            assert set(aborted)==set(fresh)
            for key,values in aborted.items():
                assert set(values)==set(fresh[key])
                for name,tensor in values.items():assert torch.equal(tensor,fresh[key][name].detach().cpu()),(key,name)
            del aborted,fresh
        if tag=='PatchTST__obs__seed3':
            previous=ART/'runtime_microbatch_v6'/tag
            assert (previous/'state.json').exists() and not (previous/'summary.json').exists()
            old_config=json.loads((previous/'config.json').read_text())
            assert old_config==asdict(config)
            old_initial=torch.load(previous/'initial_state.pt',map_location='cpu',weights_only=False)
            fresh_initial=trainer.state_dict()
            assert set(old_initial)==set(fresh_initial)
            compared=0
            for key,values in old_initial.items():
                assert set(values)==set(fresh_initial[key])
                for name,tensor in values.items():
                    assert torch.equal(tensor,fresh_initial[key][name].detach().cpu()),(key,name)
                    compared+=1
            dump(out/'retry_initialization.json',{'utc':now(),'status':'EXACT_CONFIG_AND_INITIAL_TENSOR_MATCH_TO_NATIVE_CRASH','tensor_count':compared,'previous_initial_sha256':sha(previous/'initial_state.pt'),'new_initial_sha256':sha(initial_path),'optimizer_or_checkpoint_resume':False,'protocol_sha256':sha(R/'PROTOCOL.md')})
            del old_initial,fresh_initial
        result=trainer.fit(ckpt_path=ck,verbose=True)
        result['runtime_storage_mode']=mode
        fit_seconds=time.perf_counter()-fit_begin
        assert len(curve)>0 and all(torch.isfinite(torch.tensor(float(c['val']['total'] if isinstance(c['val'],dict) else c['val']))) for c in curve)
        result.update(tag=tag,arm_name=args.arm,training_params=trainer.n_params,epochs_run=len(curve),formal_updates=len(curve)*len(trainer.train_loader),hit_epoch_cap=len(curve)==100,selected_final_epoch=result['best_epoch']==len(curve),native_fit_seconds=fit_seconds,total_instrumented_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint=str(ck),checkpoint_sha256=sha(ck),initial_state_sha256=sha(initial_path),shared_codec_sha256=None if args.arm=='obs' else sha(CODEC/'greenhouse__target.pt'),protocol_sha256=sha(R/'PROTOCOL.md'),wrapper_sha256=sha(pathlib.Path(__file__)),source_files_rechecked=check_source(),transfer_evaluation_opened=False)
        dump(out/'summary.json',result)
    dump(state,{'status':'COMPLETE','pid':os.getpid(),'tag':tag,'utc':now(),'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'status':'COMPLETE','tag':tag,'elapsed_seconds':time.perf_counter()-started}),flush=True)
except BaseException as error:
    dump(state,{'status':'FAILED','pid':os.getpid(),'tag':tag,'utc':now(),'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()});raise
