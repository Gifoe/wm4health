"""Unchanged native WorldTS classes and explicit ordinary new-task controls."""
import runtime_profile
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,hashlib,json,pathlib,sys,time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1];NATIVE=ROOT.parent/'upstream/WorldTS/DAG-bench'
sys.path.insert(0,str(NATIVE))
from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel
from ts_benchmark.baselines.worldts.losses.latent_state_loss import LatentStateLoss
METHODS=['worldts32','worldts4','worldts4_fixed','worldts32_endtoend','gru64','mlp128']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_source():
    pin=json.loads((ROOT/'native_source_pin.json').read_text())
    for name,digest in pin['files'].items():assert sha(NATIVE/name)==digest,name
    return pin
def config(method):
    d=4 if method in ['worldts4','worldts4_fixed'] else 32
    return dict(seq_len=64,pred_len=16,horizon=16,output_dim=11,future_x_dim=5,latent_dim=d,condition_dim=d,
        state_patch_len=3,state_hidden_multiplier=2,decoder_type='causal_patch',decoder_patch_len=3,decoder_hidden_multiplier=2,
        use_revin=method!='worldts4_fixed',use_future_x_condition=True,allow_missing_future_covariate=False,
        use_modal_condition=False,use_time_condition=False,latent_regularizer_type='legacy_vc',
        predictor_patch_len=8,predictor_patch_stride=4,predictor_d_model=64,predictor_n_heads=4,predictor_e_layers=1,
        predictor_d_ff=128,predictor_dropout=.05)
class GRU(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.GRU(16,64,batch_first=True);self.transition=nn.GRU(5,64,batch_first=True);self.decoder=nn.Linear(64,11)
    def forecast(self,x,x_future):
        _,h=self.encoder(x);states,_=self.transition(x_future,h);return self.decoder(states)+x[:,-1:,:11]
class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(1104,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,176))
    def forecast(self,x,x_future):return self.net(torch.cat([x.flatten(1),x_future.flatten(1)],1)).reshape(-1,16,11)+x[:,-1:,:11]
def make(method):
    return (GRU() if method=='gru64' else MLP() if method=='mlp128' else WorldTSModel(SimpleNamespace(**config(method)))).cuda()
def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def fingerprint(m):
    h=hashlib.sha256()
    for key,value in sorted(m.state_dict().items()):h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def latent_loss():return LatentStateLoss(latent_mse_weight=10,latent_alignment_loss='mae',latent_cosine_weight=15,
    latent_sigreg_weight=.1,latent_detach_target=True,sigreg_std_weight=1,sigreg_cov_weight=.04,sigreg_eps=1e-4).cuda()
def mode(model,stage,training):
    model.train(training)
    if stage=='latent':model.configure_for_stage1()
    elif stage=='decoder':
        model.configure_for_stage2();model.set_stage1_eval();model.decoder.train(training)
    else:
        for parameter in model.parameters():parameter.requires_grad_(True)
def loss(model,stage,x,u,y,criterion):
    if stage=='latent':
        out=model(input=x,x_future=u,future_target=y);parts=criterion(out['h_pred'],out['h_target']);return parts['loss'],dict(latent=parts['loss'])
    if stage=='decoder':
        with torch.no_grad():out=model(input=x,x_future=u,future_target=y)
        def decode(key):return model.decode(out[key].detach(),target_revin_stats=out['target_revin_stats'],history_tokens=out['h_hist'].detach())
        pred=decode('h_pred');oracle=decode('h_target');forecast=(pred-y).square().mean();warm=(oracle-y).square().mean()
        return forecast+.3*warm,dict(forecast_MSE=forecast,oracle_MSE=warm)
    pred=model.forecast(x,x_future=u);score=(pred-y).square().mean();return score,dict(forecast_MSE=score)
def load_split(name):
    saved=np.load(ROOT/'outputs/real_data'/(name+'.npz'),allow_pickle=False)
    return {key:torch.tensor(saved[value],device='cuda') for key,value in [('x','history'),('u','future_conditions'),('y','target')]}
def stage_fit(model,stage,train,val,seed):
    mode(model,stage,True);parameters=[p for p in model.parameters() if p.requires_grad];optim=torch.optim.Adam(parameters,lr=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,factor=.5,patience=5,min_lr=1e-5);criterion=latent_loss() if stage=='latent' else None
    rng=torch.Generator(device='cuda').manual_seed(seed+100000);best=float('inf');best_state=None;history=[];stale=0;best_epoch=0;updates=0;tick=time.perf_counter()
    for epoch in range(1,(50 if stage=='latent' else 100)+1):
        mode(model,stage,True);order=torch.randperm(len(train['x']),generator=rng,device='cuda');training=0.
        for start in range(0,len(order),128):
            idx=order[start:start+128];optim.zero_grad(set_to_none=True);score,parts=loss(model,stage,train['x'][idx],train['u'][idx],train['y'][idx],criterion)
            assert torch.isfinite(score);score.backward();norm=nn.utils.clip_grad_norm_(parameters,1);optim.step();updates+=1;training+=float(score.detach())*len(idx)
        mode(model,stage,False);validation=0.;count=0
        with torch.no_grad():
            for start in range(0,len(val['x']),128):
                stop=start+128;score,parts=loss(model,stage,val['x'][start:stop],val['u'][start:stop],val['y'][start:stop],criterion)
                selection=score if stage=='latent' else parts['forecast_MSE'];n=len(val['x'][start:stop]);validation+=float(selection)*n;count+=n
        validation/=count;scheduler.step(validation)
        history.append(dict(epoch=epoch,training_objective=training/len(order),validation_criterion=validation,lr=optim.param_groups[0]['lr'],updates=updates))
        if validation<best:
            best=validation;best_state=state(model);best_epoch=epoch;stale=0
        else:stale+=1
        if epoch%5==0:print(json.dumps(dict(phase='epoch',stage=stage,epoch=epoch,validation=validation,best_epoch=best_epoch,seconds=time.perf_counter()-tick)),flush=True)
        if stale>=(10 if stage=='latent' else 15):break
    assert best_state is not None;model.load_state_dict(best_state);model.eval()
    return dict(stage=stage,history=history,best_epoch=best_epoch,best_validation=best,updates=updates,seconds=time.perf_counter()-tick)
def train(method,seed,train,val):
    torch.manual_seed(seed);model=make(method);stages=['latent','decoder'] if method.startswith('worldts') and not method.endswith('endtoend') else ['supervised'];records=[]
    for i,stage in enumerate(stages):records.append(stage_fit(model,stage,train,val,seed+1000*i))
    return model,records
def setup():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();check_source()
