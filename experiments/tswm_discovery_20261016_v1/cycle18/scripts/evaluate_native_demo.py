"""Frozen original DynaMix DSR demonstration protocol, with explicitly separate diagnostics."""
import runtime_profile
import hashlib,json,math,pathlib,random,sys,time
import numpy as np
import torch
from safetensors.torch import load_file
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.perf_counter();torch.set_num_threads(1);assert sys.version_info>=(3,11)
    out=ROOT/'outputs/native_demo';assert not out.exists();out.mkdir(parents=True)
    source=ROOT/'upstream/DynaMix-python';meta=ROOT/'outputs/source/DynaMix'
    manifests=[json.loads((meta/f).read_text()) for f in ['manifest.json','demo_manifest.json']]
    for manifest in manifests:
        for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
    sys.path.insert(0,str(source/'src'))
    from dynamix.model.dynamix import DynaMix
    from dynamix.model.forecaster import DynaMixForecaster
    from dynamix.metrics.metrics import geometrical_misalignment,temporal_misalignment,MASE
    cfg=json.loads((ROOT/'outputs/external/DynaMix/config_3d-alrnn-v1.0.json').read_text())['architecture']
    weight=ROOT/'outputs/external/DynaMix/dynamix-3d-alrnn-v1.0.safetensors'
    assert sha(weight)=='c950aef03f24e600bb159a05bf64aaea0fc89a6c66310baac624ac4b7e10ed5b'
    torch.manual_seed(2021);np.random.seed(2021)
    model=DynaMix(**cfg).cpu().eval();model.load_state_dict(load_file(str(weight),device='cpu'),strict=True)
    forecaster=DynaMixForecaster(model);rows=[]
    for name,context_length in [('lorenz63',2000),('chua',1028),('sprottM',512)]:
        path=source/'notebooks/test_data'/(name+'.npy');data=np.load(path,allow_pickle=False)
        assert data.ndim==2 and data.shape[1]==3 and len(data)>=context_length+10000 and np.isfinite(data).all()
        context=torch.tensor(data[:context_length],dtype=torch.float32)
        truth=torch.tensor(data[context_length:context_length+10000],dtype=torch.float32)
        lo=truth.min(0).values;hi=truth.max(0).values
        def score(prediction):
            coord=(30*(prediction-lo)/(hi-lo)).long()
            retained=((coord>0)&(coord<30)).all(1).float().mean()
            true_coord=(30*(truth-lo)/(hi-lo)).long();true_retained=((true_coord>0)&(true_coord<30)).all(1).float().mean()
            escape=((prediction<lo)|(prediction>hi)).any(1).float().mean()
            g=float(geometrical_misalignment(prediction,truth,n_bins=30))
            is_constant=bool((prediction.std(0)==0).all())
            h=None if is_constant else float(temporal_misalignment(prediction,truth,smoothing=20))
            notebook_mase=float(MASE(truth,prediction,steps=10))
            correct_mase=float(MASE(prediction,truth,steps=10))
            return {'D_stsp_native':g if math.isfinite(g) else None,'D_H_native':h if h is not None and math.isfinite(h) else None,
                'D_H_unavailable_reason':'constant-series power undefined' if is_constant else ('nonfinite native result' if h is None or not math.isfinite(h) else None),
                'MASE_native_notebook_argument_order':notebook_mase,'MASE_true_denominator_diagnostic':correct_mase,
                'native_histogram_prediction_retention':float(retained),'native_histogram_truth_retention':float(true_retained),
                'truth_bbox_escape_fraction':float(escape),
                'RMSE10_context_scaled':float(torch.sqrt(torch.mean(((prediction[:10]-truth[:10])/context.std(0))**2)))}
        predictions=[]
        for seed in [2021,2022,2023]:
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);tick=time.perf_counter()
            prediction=forecaster.forecast(context=context,horizon=10000,standardize=True)
            assert prediction.shape==truth.shape and torch.isfinite(prediction).all()
            metrics=score(prediction);predictions.append(prediction.numpy())
            row={'system':name,'seed':seed,'context_length':context_length,'horizon':10000,
                'metrics':metrics,'seconds':time.perf_counter()-tick}
            rows.append(row);print(json.dumps(row),flush=True)
        replay=context.repeat((math.ceil(10000/context_length),1))[:10000]
        persistence=context[-1:].expand(10000,-1).clone()
        for control,prediction in [('context_replay',replay),('persistence',persistence)]:
            rows.append({'system':name,'control':control,'context_length':context_length,'horizon':10000,'metrics':score(prediction)})
        np.savez_compressed(out/(name+'.npz'),data=data,predictions=np.stack(predictions),
            context_replay=replay.numpy(),persistence=persistence.numpy(),seeds=np.asarray([2021,2022,2023]),context_length=np.asarray(context_length))
    for manifest in manifests:
        for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
    result={'rows':rows,'source_revision':manifests[0]['revision'],'model_sha256':sha(weight),
        'source_and_demo_files_unchanged':sum(len(m['files']) for m in manifests),
        'data_sha256':{r['path']:r['sha256'] for r in manifests[1]['files'] if r['path'].endswith('.npy')},
        'arrays_sha256':{p.name:sha(p) for p in out.glob('*.npz')},'script_sha256':sha(pathlib.Path(__file__)),
        'plan_sha256':sha(ROOT/'DEMO_REPLICATION_PLAN.md'),'elapsed_seconds':time.perf_counter()-started,
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu'},
        'scope':'Three source-native released demonstration cases; not the54-system aggregate or a novel method.',
        'MASE_discrepancy_scope':'Current Python notebook call order; no assertion about original Julia/published table implementation.',
        'new_training':False,'accepted_novel_directions':0,'objective_achieved':False}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'completed_rows':len(rows),'elapsed_seconds':result['elapsed_seconds']}))
if __name__=='__main__':main()
