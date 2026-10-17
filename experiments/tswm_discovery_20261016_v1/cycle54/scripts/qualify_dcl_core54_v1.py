"""Native DCL numerical/input-contract checks and32 discarded GPU solver updates."""
import ctypes, datetime, hashlib, importlib.metadata, json, pathlib, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]; PROJECT=ROOT.parent; OUT=ROOT/'outputs'
fn=ctypes.windll.kernel32.SetProcessAffinityMask; fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]; assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff))
manifest=json.loads((ROOT/'dcl_source_manifest54_v1.json').read_text()); SOURCE=PROJECT/'upstream/dcl_2cd46d13'; sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
from dcl.criterions.contrastive import MseInfoNCE
from dcl.datasets.timeseries import TensorDataset
from dcl.loader.contrastive import DiscreteTimeContrastiveDataLoader
from dcl.models.dynamics.linear_dynamics import LinearDynamicsModel
from dcl.models.dynamics.switching_dynamics import MSESwitchingModel
from dcl.models.encoder import Offset1ModelMLP
from dcl.solver.contrastive_solver import DynamicsContrastiveLearningSolver
from dcl.solver.optimizer import DCLAdamOptimizer
from dcl.utils.datatypes import ErrorBasedSwitchingDynamicsData,LinearDynamicsInput,TimeContrastiveLossBatch
torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
STORAGE=json.loads((PROJECT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def source_check():
    result={}
    for r in manifest['selected_files']:
        b=(SOURCE/r['path']).read_bytes(); assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['git_blob_sha1'],r['path']
        result[r['path']]=hashlib.sha256(b).hexdigest()
    return result

class TraceLoader(DiscreteTimeContrastiveDataLoader):
    """Pass through native sampled batches unchanged; retain actual pairing indices."""
    def sample_batch(self):
        batch=super().sample_batch()
        self.pair_trace.append((batch.reference_indices.detach().cpu().numpy().copy(),batch.positive_indices.detach().cpu().numpy().copy()))
        return batch

def main():
    start=time.perf_counter(); output=OUT/'dcl_preflight_v1'; assert not output.exists(); output.mkdir()
    before=source_check(); assert len(before)==53
    assert torch.cuda.is_available() and torch.__version__=='2.8.0+cu128'
    rng=np.random.default_rng(54); ref=rng.normal(size=(7,3)); pos=rng.normal(size=(7,3)); neg=rng.normal(size=(11,3)); temperature=.7
    scores=-np.concatenate([((ref-pos)**2).sum(-1)[:,None],((ref[:,None]-neg[None])**2).sum(-1)],axis=1)/temperature
    maxs=scores.max(1,keepdims=True); probs=np.exp(scores-maxs); probs/=probs.sum(1,keepdims=True)
    expected_loss=float((np.log(np.exp(scores-maxs).sum(1))+maxs[:,0]-scores[:,0]).mean())
    centers=np.concatenate([pos[:,None,:],np.broadcast_to(neg,(len(ref),len(neg),3))],axis=1)
    expected_grad=2*((probs[:,:,None]*centers).sum(1)-pos)/(temperature*len(ref))
    checks=[]
    for device in ['cpu','cuda']:
        r=torch.tensor(ref,dtype=torch.float64,device=device,requires_grad=True)
        p=torch.tensor(pos,dtype=torch.float64,device=device); n=torch.tensor(neg,dtype=torch.float64,device=device)
        criterion=MseInfoNCE(temperature=temperature,infonce_type='infonce_full_denominator')
        loss,_,_=criterion(TimeContrastiveLossBatch(reference=r,positive=p,negative=n))
        gradient=torch.autograd.grad(loss,r)[0].detach().cpu().numpy()
        record={'device':device,'loss_abs_error':abs(float(loss.detach().cpu())-expected_loss),'reference_gradient_max_abs_error':float(np.max(abs(gradient-expected_grad)))}
        assert max(record['loss_abs_error'],record['reference_gradient_max_abs_error'])<1e-10,record; checks.append(record)
    A=np.array([[.85,-.15,0.],[.2,.9,0.],[0.,.1,.7]])
    linear=LinearDynamicsModel(dim=3).double()
    with torch.no_grad():linear.A.copy_(torch.tensor(A[None]))
    pred=linear(LinearDynamicsInput(x=torch.tensor(ref[:,None,:]),x_index=torch.arange(7)[:,None])).x.detach().numpy()
    linear_error=float(np.max(abs(pred-ref@A.T))); assert linear_error<1e-12
    candidates=np.stack([ref,-ref],axis=1); future=ref+.125
    gate=MSESwitchingModel(num_modes=2).double()
    logits=gate.logits(ErrorBasedSwitchingDynamicsData(x_next=torch.tensor(future),x_next_pred=torch.tensor(candidates))).detach().numpy()
    expected_logits=1/(((future[:,None]-candidates)**2).mean(-1)+gate.mse_to_logits.eps)
    logits_error=float(np.max(abs(logits-expected_logits))); assert logits_error<1e-10
    other=gate.logits(ErrorBasedSwitchingDynamicsData(x_next=torch.tensor(-ref+.125),x_next_pred=torch.tensor(candidates))).detach().numpy()
    changed=int(np.sum(logits.argmax(1)!=other.argmax(1))); assert changed>0
    artifact=pathlib.Path(STORAGE['artifact_root'])/'cycle54/dcl_preflight_v1'; artifact.mkdir(parents=True,exist_ok=True)
    fp=artifact/'numerical_fixtures.npz'; assert not fp.exists(); np.savez_compressed(fp,reference=ref,positive=pos,negative=neg,expected_gradient=expected_grad,gate_logits=logits,gate_changed_target_logits=other)
    # Independent toy observations only; no C54 signal or hidden-state arrays loaded.
    toy=rng.normal(scale=.1,size=(16,256,5)).astype(np.float32)
    for t in range(1,256):toy[:,t]+=.8*toy[:,t-1]
    trial_ids=np.repeat(np.arange(16),256); trial_time=np.tile(np.arange(256),16)
    dataset=TensorDataset(data=torch.tensor(toy.reshape(-1,5)),trial_id=torch.tensor(trial_ids),trial_time=torch.tensor(trial_time))
    loader=TraceLoader(batch_size=64,batch_size_neg=256,num_iterations=32,seed=54);loader.pair_trace=[];loader.lazy_init(dataset);loader.to(torch.device('cuda'))
    encoder=Offset1ModelMLP(input_dim=5,output_dim=3,hidden_dim=180,num_layers=3,normalize=False,seed=54)
    dynamics=LinearDynamicsModel(dim=3,seed=54)
    solver=DynamicsContrastiveLearningSolver(model=encoder,dynamics_model=dynamics,optimizer=DCLAdamOptimizer(encoder_learning_rate=3e-4,dynamics_learning_rate=3e-4),criterion=MseInfoNCE(temperature=1.,infonce_type='infonce_full_denominator'),seed=54)
    initial=[p.detach().cpu().clone() for p in list(encoder.parameters())+list(dynamics.parameters())]
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); fit_start=time.perf_counter();solver.fit(loader);torch.cuda.synchronize();fit_seconds=time.perf_counter()-fit_start
    losses=np.asarray(solver.logs['train_loss']);assert solver.current_step==32 and len(losses)==32 and len(loader.pair_trace)==32 and np.isfinite(losses).all()
    parameters=list(encoder.parameters())+list(dynamics.parameters());assert all(torch.isfinite(p).all() for p in parameters)
    parameter_change=float(sum(torch.sum((p.detach().cpu()-old)**2).item() for p,old in zip(parameters,initial)))
    assert parameter_change>0 and next(encoder.parameters()).device.type=='cuda'
    for ri,pi in loader.pair_trace:
        assert np.array_equal(trial_ids[ri[..., -1]],trial_ids[pi[..., -1]])
    save=artifact/'gpu_toy_preflight_weights.pt';assert not save.exists();torch.save({'encoder':encoder.state_dict(),'dynamics':dynamics.state_dict(),'optimizer':solver.optimizer.state_dict(),'losses':losses},save)
    pairs=artifact/'actual_training_pairs.npz';np.savez_compressed(pairs,reference_indices=np.stack([p[0] for p in loader.pair_trace]),positive_indices=np.stack([p[1] for p in loader.pair_trace]))
    after=source_check();assert before==after
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_NATIVE_DCL_CORE_ARITHMETIC_AND_32_DISCARDED_GPU_UPDATES','source_revision':manifest['source_revision'],'source_files_verified_unchanged':before,'numerical_checks':checks,'linear_prediction_max_abs_error':linear_error,'native_switching_logit_max_abs_error':logits_error,'mode_preference_changes_when_only_future_target_changes':changed,'mode_contract_scope':'Retrospective target-based identification control, not a forecasting quality or data-leakage accusation.','discarded_native_optimizer_updates':32,'formal_fits':0,'paper_dataset_downloads':0,'C54_heldout_read':False,'C54_model_selection_changed':False,'toy_trials':16,'toy_points_per_trial':256,'actual_training_pair_count':32*64,'trial_boundaries_respected':True,'loader_instrumentation':'TraceLoader returns the unchanged native sampled batch; only recording its actual pair indices.','actual_encoder_linear_shapes':[{'in_features':m.in_features,'out_features':m.out_features} for m in encoder.net if isinstance(m,torch.nn.Linear)],'constructor_hidden_dim_num_layers_ignored_by_native_Offset1ModelMLP':True,'parameter_count':sum(p.numel() for p in parameters),'parameter_squared_change':parameter_change,'all_losses':losses.tolist(),'fit_wall_seconds':fit_seconds,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'python':sys.executable,'protocol_sha256':sha(ROOT/'DCL_PREFLIGHT54_V1.md'),'script_sha256':sha(__file__),'artifacts':[{'path':str(p),'sha256':sha(p)} for p in [fp,save,pairs]],'runtime_seconds':time.perf_counter()-start,'scope':'Native implementation and numerical qualification only. No scientific learning/forecast result, no full paper reproduction, no new method or accepted direction.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(result),flush=True)
if __name__=='__main__':main()
