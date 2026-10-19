"""Exploratory internal-noise/history/normalization controls; no training or source changes."""
import runtime_profile
import hashlib,json,math,pathlib,sys,time
import numpy as np
import torch
from safetensors.torch import load_file
from path_event_audit import summarize,compare
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT.parent/'cycle18'
OUT=ROOT/'outputs/attribution'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.perf_counter();torch.set_num_threads(1);assert sys.version_info>=(3,11)
    assert not OUT.exists();OUT.mkdir(parents=True)
    source=BASE/'upstream/DynaMix-python';manifest=json.loads((BASE/'outputs/source/DynaMix/manifest.json').read_text())
    for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
    weight=BASE/'outputs/external/DynaMix/dynamix-3d-alrnn-v1.0.safetensors'
    expected='c950aef03f24e600bb159a05bf64aaea0fc89a6c66310baac624ac4b7e10ed5b';assert sha(weight)==expected
    cfg=json.loads((BASE/'outputs/external/DynaMix/config_3d-alrnn-v1.0.json').read_text())['architecture']
    sys.path.insert(0,str(source/'src'))
    from dynamix.model.dynamix import DynaMix
    from dynamix.model.forecaster import DynaMixForecaster
    from dynamix.metrics.metrics import geometrical_misalignment,temporal_misalignment,MASE
    def model_for(zero_noise=False):
        model=DynaMix(**cfg).eval();model.load_state_dict(load_file(str(weight)),strict=True)
        if zero_noise:
            before={k:v.clone() for k,v in model.state_dict().items()}
            with torch.no_grad():model.gating_network.sigma.zero_()
            assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items() if k!='gating_network.sigma')
        return model
    control_rng=np.random.default_rng(20261903);rows=[]
    def evaluate(label,name,prediction,truth,reference_context,seed,seconds,origin):
        array=prediction.numpy();path=OUT/(label+'_'+name+'_'+str(seed)+'.npz')
        np.savez_compressed(path,prediction=array,truth=truth.numpy(),reference_context=reference_context.numpy())
        finite=bool(torch.isfinite(prediction).all());record={'condition':label,'system':name,'seed':seed,'origin':origin,
            'finite':finite,'seconds':seconds,'array_file':path.name,'array_sha256':sha(path)}
        if finite:
            truth_summary=summarize(truth[:,0].numpy(),reference_context[:,0].numpy())
            pred_summary=summarize(array[:,0],reference_context[:,0].numpy())
            record['event_summaries']=pred_summary[0];record['event_comparison']=compare(pred_summary,truth_summary,control_rng)
            degenerate=bool((prediction.std(0)<1e-8).any())
            geometry=float(geometrical_misalignment(prediction,truth,n_bins=30))
            record['global_metrics']={'D_stsp':geometry,'D_H':None if degenerate else float(temporal_misalignment(prediction,truth,smoothing=20)),
                'MASE_true_denominator':float(MASE(prediction,truth,steps=10)),
                'RMSE10_context_scaled':float(torch.sqrt(torch.mean(((prediction[:10]-truth[:10])/reference_context.std(0))**2))),
                'any_degenerate_coordinate':degenerate}
        rows.append(record);print(json.dumps({'condition':label,'system':name,'seed':seed,'seconds':seconds,'finite':finite}),flush=True)
    # Removing only in-memory stochastic gating, across all original cases.
    for name,cl in [('lorenz63',2000),('chua',1028),('sprottM',512)]:
        data=np.load(source/'notebooks/test_data'/(name+'.npy'),allow_pickle=False)
        ctx=torch.tensor(data[:cl],dtype=torch.float32);truth=torch.tensor(data[cl:cl+10000],dtype=torch.float32)
        model=model_for(True);torch.manual_seed(2021);tick=time.perf_counter()
        pred=DynaMixForecaster(model).forecast(ctx,10000,standardize=True)
        evaluate('zero_attention_noise',name,pred,truth,ctx,2021,time.perf_counter()-tick,cl)
    # Common-origin Chua: same future and common metric/threshold coordinates for every arm.
    data=np.load(source/'notebooks/test_data/chua.npy',allow_pickle=False)
    assert len(data)==12000
    short=torch.tensor(data[972:2000],dtype=torch.float32);long=torch.tensor(data[:2000],dtype=torch.float32)
    truth=torch.tensor(data[2000:12000],dtype=torch.float32);mean=short.mean(0);std=short.std(0).clamp(min=1e-6)
    model=model_for(False);forecaster=DynaMixForecaster(model)
    torch.manual_seed(2021);native=forecaster.forecast(short,64,standardize=True)
    torch.manual_seed(2021);fixed=forecaster.forecast((short-mean)/std,64,standardize=False)*std+mean
    coordinate_error=float((native-fixed).abs().max());assert coordinate_error<1e-5,coordinate_error
    for label,context,standardize,inverse in [('short_native',short,True,False),
        ('long_fixed_short_coordinates',(long-mean)/std,False,True),('long_native_coordinates',long,True,False)]:
        for seed in [2021,2022,2023]:
            torch.manual_seed(seed);tick=time.perf_counter();pred=forecaster.forecast(context,10000,standardize=standardize)
            if inverse:pred=pred*std+mean
            evaluate(label,'chua',pred,truth,short,seed,time.perf_counter()-tick,2000)
    for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
    assert sha(weight)==expected
    def clean(v):
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    result={'rows':rows,'short_fixed_coordinate_equivalence64_error':coordinate_error,
        'source_revision':manifest['revision'],'checkpoint_sha256':expected,'source_files_unchanged':len(manifest['files']),
        'elapsed_seconds':time.perf_counter()-started,'script_sha256':sha(pathlib.Path(__file__)),
        'attribution_plan_sha256':sha(ROOT/'ATTRIBUTION_PLAN.md'),'event_helper_sha256':sha(ROOT/'scripts/path_event_audit.py'),
        'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'device':'cpu'},
        'scope':'Exploratory fixed controls on an already inspected released fixture; not independent confirmation, new training or a new method.',
        'unavailable_stat_reason':'Null means nonfinite/empty conditional cohort, never imputed as zero.',
        'accepted_novel_directions':0,'objective_achieved':False}
    (OUT/'results.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'completed':len(rows),'seconds':result['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
