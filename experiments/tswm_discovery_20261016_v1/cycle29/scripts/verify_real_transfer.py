"""Independent real-window construction, all scores/ARX fits, frozen-weight replay."""
import runtime_profile
import csv,json,pathlib,sys,time
import numpy as np
import torch
from world_data import ROOT,sha,dump
from world_models_v2 import build
from train_world_comparison_v2 import setup,KINDS,SEEDS,hashes
from verify_world_comparison_v2 import ridge_lstsq,free_arx
from evaluate_world_comparison_v2 import pretrained
from world_models import scale_context as legacy_scale
OUT=ROOT/'outputs/real_transfer';SPLITS=ROOT/'outputs/real_splits'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def key(family,record,method,seed,origin,h):return(family,record,method,'' if seed is None else str(seed),origin,h)
@torch.no_grad()
def independent_predict(model,record,raw,origin,chunk=False,legacy=False):
    y=(raw['test_y'][origin-128:origin]-record['train_means'][1])/record['train_stds'][1];u=(raw['test_u'][origin-128:origin+128]-record['train_means'][0])/record['train_stds'][0]
    cy=torch.tensor(y,dtype=torch.float32,device='cuda')[None,:,None];cu=torch.tensor(u[:128],dtype=torch.float32,device='cuda')[None,:,None];fu=torch.tensor(u[128:],dtype=torch.float32,device='cuda')[None,:,None]
    if legacy:cy,cu,fu,_,ym,ys=legacy_scale(cy,cu,fu)
    if chunk:
        parts=[]
        for t in range(4):
            x=fu[:,32*t:32*t+32];p=model(cy,cu,x);parts.append(p);cy=torch.cat((cy,p),1)[:,-128:];cu=torch.cat((cu,x),1)[:,-128:]
        p=torch.cat(parts,1)
    else:p=model(cy,cu,fu)
    if legacy:p=p*ys+ym
    return p[0,:,0].cpu().numpy().astype('float64')*record['train_stds'][1]+record['train_means'][1]
def main():
    setup();started=time.monotonic();result=read(OUT/'results.json');manifest=read(SPLITS/'manifest.json');assert result['status']=='completed' and manifest['status']=='passed'
    assert result['split_manifest_sha256']==sha(SPLITS/'manifest.json') and result['protocol_sha256']==sha(ROOT/'REAL_TRANSFER_PLAN.md')
    records={(r['family'],r['record']):r for r in manifest['records'] if r['include_aggregate']};raws={};windows=0
    for name,record in records.items():
        path=SPLITS/record['source_npz'];assert sha(path)==record['sha256']
        with np.load(path,allow_pickle=False) as f:raws[name]={k:f[k] for k in f.files}
        raw=raws[name];assert np.array_equal(np.array([raw['train_u'].mean(),raw['train_y'].mean()]),record['train_means']);assert np.array_equal(np.array([raw['train_u'].std(),raw['train_y'].std()]),record['train_stds'])
        available=list(range(128,len(raw['test_y'])-128+1,128));selected=[available[i] for i in np.linspace(0,len(available)-1,min(32,len(available)),dtype=int)];assert selected==record['selected_origins'];windows+=len(selected)
    assert windows==manifest['n_forecast_windows']==71 and manifest['physical_systems']==2
    with (OUT/'per_window_metrics.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    by={key(r['family'],r['record'],r['method'],int(r['seed']) if r['seed'] else None,int(r['origin']),int(r['horizon'])):r for r in rows};assert len(by)==len(rows)
    n=0;maxerr=0.;predictions={}
    for file in result['predictions']:
        path=OUT/file['path'];assert sha(path)==file['sha256'];name=(file['family'],file['record']);record=records[name];raw=raws[name]
        with np.load(path,allow_pickle=False) as f:pred=f['prediction'].astype('float64');target=f['target'];origins=f['origins']
        assert list(origins)==record['selected_origins'] and pred.shape==(len(origins),128)
        expected=np.stack([raw['test_y'][o:o+128] for o in origins]);assert np.array_equal(target,expected)
        predictions[(name,file['method'],file['seed'])]=pred
        if file['seed'] is not None:
            kind=file['method'].replace('_chunk32','');path=ROOT/'outputs/world_training_v2'/f'{kind}_{file["seed"]}'/'best.pt';assert file['weight_sha256']==sha(path)
        for i,origin in enumerate(origins):
            for h in [32,64,128]:
                r=by[key(*name,file['method'],file['seed'],int(origin),h)];err=pred[i,:h]-expected[i,:h];rmse=np.linalg.norm(err)/np.sqrt(h)
                vals=dict(rmse=rmse,mae=np.sum(np.abs(err))/h,nrmse_train_std=rmse/record['train_stds'][1])
                for metric,value in vals.items():
                    difference=abs(float(r[metric])-value)/(1+abs(value));assert difference<1e-9,(name,file['method'],metric,difference);maxerr=max(maxerr,difference)
                n+=1
    assert n==result['n_metric_rows']==len(rows)==4260
    with (OUT/'arx_context_validation.csv').open(encoding='utf-8',newline='') as f:vr=list(csv.DictReader(f))
    vb={(r['family'],r['record'],int(r['origin']),int(r['order']),float(r['lam'])):float(r['context_validation_mse']) for r in vr};arxmax=0.;refits=0
    for fit in read(OUT/'arx_fits.json'):
        name=(fit['family'],fit['record']);record=records[name];raw=raws[name];o=fit['origin'];y=(raw['test_y'][o-128:o]-record['train_means'][1])/record['train_stds'][1];u=(raw['test_u'][o-128:o+128]-record['train_means'][0])/record['train_stds'][0];candidates=[]
        for order in [1,2,4,8,16]:
            for lam in [1e-4,.01,1.]:
                f=ridge_lstsq(y[:96],u[:96],order,lam);p=free_arx(f,y[:96],u[:128],32);mse=float(np.mean((p-y[96:])**2));old=vb[(*name,o,order,lam)];difference=abs(mse-old)/(1+abs(mse));assert difference<2e-6;arxmax=max(arxmax,difference);candidates.append(mse);refits+=1
        assert vb[(*name,o,fit['order'],fit['lam'])]<=min(candidates)+2e-6*(1+abs(min(candidates)))
        f=ridge_lstsq(y,u[:128],fit['order'],fit['lam']);p=free_arx(f,y,u,128)*record['train_stds'][1]+record['train_means'][1];index=record['selected_origins'].index(o);expected=predictions[(name,'arx_context_selected',None)][index];difference=float(np.max(np.abs(p-expected)/(1+np.abs(expected))));assert difference<2e-5,(name,o,difference);arxmax=max(arxmax,difference);refits+=1
    replays=[]
    for seed in SEEDS:
        for kind in KINDS:
            folder=ROOT/'outputs/world_training_v2'/f'{kind}_{seed}';state=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);assert state['hashes']==hashes();model=build(kind).cuda().eval();model.load_state_dict(state['model'],strict=True)
            for name,record in records.items():
                for method in ([kind,kind+'_chunk32'] if kind=='native_attention64' else [kind]):
                    p=independent_predict(model,record,raws[name],record['selected_origins'][0],chunk=method.endswith('_chunk32'));expected=predictions[(name,method,seed)][0];error=float(np.max(np.abs(p-expected)/(1+np.abs(expected))));assert error<2e-4,(name,method,seed,error);replays.append(dict(family=name[0],record=name[1],method=method,seed=seed,error=error))
            del model;torch.cuda.empty_cache()
    model,_=pretrained()
    for name,record in records.items():
        p=independent_predict(model,record,raws[name],record['selected_origins'][0],legacy=True);expected=predictions[(name,'pretrained_native_icl',None)][0];error=float(np.max(np.abs(p-expected)/(1+np.abs(expected))));assert error<2e-4;replays.append(dict(family=name[0],record=name[1],method='pretrained_native_icl',seed=None,error=error))
    del model;torch.cuda.empty_cache()
    sys.path.insert(0,str(ROOT.parent/'cycle28/upstream/PCFM'));from model import PCFMPredictor
    weight=ROOT.parent/'cycle28/outputs/external/PCFM/model.safetensors';assert sha(weight)=='746ff3eb69f9fcaa0e934f13b5afda17be7591a987eb64c671fa05b1d87b6755';model=PCFMPredictor.from_pretrained(str(weight),device='cuda')
    for name,record in records.items():
        raw=raws[name];origin=record['selected_origins'][0];u=(raw['test_u'][origin-128:origin+128]-record['train_means'][0])/record['train_stds'][0];y=(raw['test_y'][origin-128:origin]-record['train_means'][1])/record['train_stds'][1]
        window=np.vstack((u,np.r_[y,np.zeros(128)]));q=model.what_if(window,context_length=128,masked_nodes=[1],graph='CD',glue=False,return_quantiles=True)[1,128:].cpu().numpy();p=q[:,4]*record['train_stds'][1]+record['train_means'][1];expected=predictions[(name,'pretrained_pcfm_cd_glue0',None)][0];error=float(np.max(np.abs(p-expected)/(1+np.abs(expected))));assert error<2e-4;replays.append(dict(family=name[0],record=name[1],method='pretrained_pcfm_cd_glue0',seed=None,error=error))
    verification=dict(status='passed',results_sha256=sha(OUT/'results.json'),script_sha256=sha(pathlib.Path(__file__)),metric_rows_rescored=n,metric_relative_error_max=maxerr,frozen_train_unit_transforms=len(records),independent_origins_reconstructed=windows,arx_refits=refits,arx_relative_error_max=arxmax,sampled_neural_replays=replays,neural_relative_error_max=max(r['error'] for r in replays),elapsed_s=time.monotonic()-started,scope='all measured-data windows,unit transforms,all scores,ARX fits,frozen weights and sampled neural outputs;not real intervention ground truth')
    dump(OUT/'verification.json',verification);print(json.dumps({k:v for k,v in verification.items() if k!='sampled_neural_replays'}),flush=True)
if __name__=='__main__':main()
