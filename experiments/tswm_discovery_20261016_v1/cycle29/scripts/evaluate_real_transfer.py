"""Frozen synthetic-trained models on measured systems. Historical-input simulation only."""
import runtime_profile
import csv,json,pathlib,sys,time
import numpy as np
import torch
from world_data import ROOT,sha,dump
from world_models_v2 import build
from world_models import scale_context as legacy_scale
from train_world_comparison_v2 import setup,KINDS,SEEDS,hashes
from evaluate_world_comparison_v2 import arx_fit,arx_predict,pretrained,csvwrite
SPLITS=ROOT/'outputs/real_splits';OUT=ROOT/'outputs/real_transfer'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def make_inputs(manifest,record):
    path=SPLITS/record['source_npz'];assert sha(path)==record['sha256']
    with np.load(path,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    origins=record['selected_origins'];mean=np.asarray(record['train_means']);std=np.asarray(record['train_stds']);features=[];targets=[]
    for origin in origins:
        u=(raw['test_u'][origin-128:origin+128]-mean[0])/std[0];cy=(raw['test_y'][origin-128:origin]-mean[1])/std[1]
        features.append((cy,u[:128],u[128:]));targets.append(raw['test_y'][origin:origin+128])
    cy,cu,fu=[torch.tensor(np.stack([x[i] for x in features]),dtype=torch.float32,device='cuda')[:,:,None] for i in range(3)]
    return raw,cy,cu,fu,np.stack(targets)
@torch.no_grad()
def neural(model,cy,cu,fu,chunk=False,legacy=False):
    ym=ys=None
    if legacy:cy,cu,fu,_,ym,ys=legacy_scale(cy,cu,fu)
    if chunk:
        pieces=[]
        for k in range(0,128,32):
            u=fu[:,k:k+32];p=model(cy,cu,u);pieces.append(p);cy=torch.cat((cy,p),1)[:,-128:];cu=torch.cat((cu,u),1)[:,-128:]
        p=torch.cat(pieces,1)
    else:p=model(cy,cu,fu)
    if legacy:p=p*ys+ym
    return p[:,:,0].cpu().numpy()
def score_rows(record,method,seed,pred,target):
    result=[]
    for j,origin in enumerate(record['selected_origins']):
        for h in [32,64,128]:
            error=pred[j,:h].astype('float64')-target[j,:h];rmse=float(np.linalg.norm(error)/np.sqrt(h))
            result.append(dict(family=record['family'],record=record['record'],origin=origin,method=method,seed=seed,horizon=h,rmse=rmse,mae=float(np.mean(np.abs(error))),nrmse_train_std=rmse/record['train_stds'][1],finite=bool(np.isfinite(pred[j,:h]).all())))
    return result
def main():
    setup();proof=read(ROOT/'outputs/world_verification/model_comparison_v2.json');assert proof['status']=='passed'
    assert proof['training_summary_sha256']==sha(ROOT/'outputs/world_training_v2/summary.json')
    manifest=read(SPLITS/'manifest.json');assert manifest['status']=='passed' and manifest['protocol_sha256']==sha(ROOT/'REAL_TRANSFER_PLAN.md')
    assert not OUT.exists();OUT.mkdir();start=time.monotonic();rows=[];files=[];validation=[];fits=[];records=[r for r in manifest['records'] if r['include_aggregate']];inputs={}
    for record in records:inputs[(record['family'],record['record'])]=make_inputs(manifest,record)
    def save(record,method,seed,pred,**metadata):
        key=(record['family'],record['record']);target=inputs[key][-1];assert pred.shape==target.shape
        suffix='' if seed is None else '_'+str(seed);path=OUT/(record['family']+'__'+record['record']+'__'+method+suffix+'.npz')
        np.savez_compressed(path,prediction=pred,target=target,origins=np.asarray(record['selected_origins']))
        files.append(dict(path=path.name,sha256=sha(path),family=record['family'],record=record['record'],method=method,seed=seed,**metadata));rows.extend(score_rows(record,method,seed,pred,target))
    for seed in SEEDS:
        for kind in KINDS:
            folder=ROOT/'outputs/world_training_v2'/f'{kind}_{seed}';state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);assert state['hashes']==hashes()
            model=build(kind).cuda().eval();model.load_state_dict(state['model'],strict=True)
            for record in records:
                _,cy,cu,fu,_=inputs[(record['family'],record['record'])]
                for mode in (['direct','chunk32'] if kind=='native_attention64' else ['direct']):
                    p=neural(model,cy,cu,fu,chunk=(mode=='chunk32'));pred=p.astype('float64')*record['train_stds'][1]+record['train_means'][1]
                    assert np.isfinite(pred).all();save(record,kind+('_chunk32' if mode=='chunk32' else ''),seed,pred,weight_sha256=sha(folder/'best.pt'),point='deterministic mean regression')
            del model;torch.cuda.empty_cache();print(json.dumps(dict(phase='real_neural_done',kind=kind,seed=seed)),flush=True)
    for record in records:
        raw,cy,cu,fu,target=inputs[(record['family'],record['record'])]
        observed_y=np.stack([(raw['test_y'][o-128:o]-record['train_means'][1])/record['train_stds'][1] for o in record['selected_origins']])
        observed_u=np.stack([(raw['test_u'][o-128:o]-record['train_means'][0])/record['train_stds'][0] for o in record['selected_origins']])
        future_u=np.stack([(raw['test_u'][o:o+128]-record['train_means'][0])/record['train_stds'][0] for o in record['selected_origins']]);pred=[]
        for j,origin in enumerate(record['selected_origins']):
            candidates=[]
            for order in [1,2,4,8,16]:
                for lam in [1e-4,.01,1.]:
                    fit=arx_fit(observed_y[j,:96],observed_u[j,:96],order,lam);p=arx_predict(fit,observed_y[j,:96],observed_u[j],32);mse=float(np.mean((p-observed_y[j,96:])**2));candidates.append((mse,order,lam));validation.append(dict(family=record['family'],record=record['record'],origin=origin,order=order,lam=lam,context_validation_mse=mse))
            best=min(candidates);fit=arx_fit(observed_y[j],observed_u[j],best[1],best[2]);u=np.r_[observed_u[j],future_u[j]];p=arx_predict(fit,observed_y[j],u,128);pred.append(p*record['train_stds'][1]+record['train_means'][1]);fits.append(dict(family=record['family'],record=record['record'],origin=origin,validation_mse=best[0],**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fit.items()}))
        save(record,'arx_context_selected',None,np.array(pred),selection='within allowed 128-step context only',fit_dtype='float64 original measured values')
        save(record,'persistence',None,np.repeat(raw['test_y'][np.asarray(record['selected_origins'])-1,None],128,axis=1))
        save(record,'train_mean',None,np.full_like(target,record['train_means'][1]))
    model,weight=pretrained()
    for record in records:
        _,cy,cu,fu,_=inputs[(record['family'],record['record'])];p=neural(model,cy,cu,fu,legacy=True);save(record,'pretrained_native_icl',None,p.astype('float64')*record['train_stds'][1]+record['train_means'][1],weight_sha256=weight,point='native Gaussian mean',budget='external pretraining, not matched')
    del model;torch.cuda.empty_cache()
    source=read(ROOT.parent/'cycle28/outputs/source/pcfm_manifest.json')
    for file in source['files']:assert sha(ROOT.parent/'cycle28/upstream/PCFM'/file['path'])==file['sha256']
    sys.path.insert(0,str(ROOT.parent/'cycle28/upstream/PCFM'));from model import PCFMPredictor
    weightpath=ROOT.parent/'cycle28/outputs/external/PCFM/model.safetensors';weight=sha(weightpath);assert weight=='746ff3eb69f9fcaa0e934f13b5afda17be7591a987eb64c671fa05b1d87b6755';model=PCFMPredictor.from_pretrained(str(weightpath),device='cuda')
    for record in records:
        _,cy,cu,fu,_=inputs[(record['family'],record['record'])];pred=[]
        for j in range(len(cy)):
            u=torch.cat((cu[j,:,0],fu[j,:,0])).cpu().numpy();y=np.r_[cy[j,:,0].cpu().numpy(),np.zeros(128)];window=np.stack((u,y))
            q=model.what_if(window,context_length=128,masked_nodes=[1],graph='CD',glue=False,return_quantiles=True)[1,128:].cpu().numpy();pred.append(q[:,4]*record['train_stds'][1]+record['train_means'][1])
        save(record,'pretrained_pcfm_cd_glue0',None,np.array(pred),weight_sha256=weight,source_revision=source['revision'],graph='CD',glue=False,point='native marginal median',budget='external pretraining, not matched')
    csvwrite(OUT/'per_window_metrics.csv',rows);csvwrite(OUT/'arx_context_validation.csv',validation);dump(OUT/'arx_fits.json',fits)
    result=dict(status='completed',script_sha256=sha(pathlib.Path(__file__)),protocol_sha256=sha(ROOT/'REAL_TRANSFER_PLAN.md'),split_manifest_sha256=sha(SPLITS/'manifest.json'),main_comparison_verification_sha256=sha(ROOT/'outputs/world_verification/model_comparison_v2.json'),n_metric_rows=len(rows),predictions=files,physical_systems=manifest['physical_systems'],forecast_windows=manifest['n_forecast_windows'],elapsed_s=time.monotonic()-start,scope='predeclared recorded-input rolling-origin simulation; not official full open-loop ranking, unseen-plan truth, or real-data neural finetuning',accepted_final_directions=0)
    dump(OUT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='predictions'}),flush=True)
if __name__=='__main__':main()
