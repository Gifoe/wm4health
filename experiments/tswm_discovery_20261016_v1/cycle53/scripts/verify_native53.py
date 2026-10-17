"""Independent retained-array arithmetic and native-score denominator audit."""
from native53 import *
import numpy as np
import torch

OUT=ART/'evaluation_v1'
complete=json.loads((OUT/'COMPLETE.json').read_text())
assert complete['status']=='NATIVE_AND_NEW_CROP_PREDICTIONS_COMPLETE_UNVERIFIED'
assert not (OUT/'verification.json').exists()
rows=json.loads((OUT/'summary.json').read_text())
manifest=json.loads((OUT/'raw_manifest.json').read_text())
assert len(rows)==len(manifest)==84
by_id={(r['tag'],r['bank']):r for r in rows};assert len(by_id)==84
inputs=json.loads((OUT/'input_manifest.json').read_text())
banks={}
for bank,item in inputs.items():
    path=OUT/item['path'];assert sha(path)==item['sha256']
    blob=torch.load(path,map_location='cpu',weights_only=False)
    assert blob['n']==item['n'];banks[bank]=blob
max_error=0.;total_windows=0;native_cells=0
for entry in manifest:
    path=OUT/entry['path'];assert sha(path)==entry['sha256']
    with np.load(path,allow_pickle=False) as z:
        pred=z['prediction'].astype(np.float64);truth=z['target']
    assert pred.shape==truth.shape==(entry['n'],16,4)
    assert np.isfinite(pred).all() and np.isfinite(truth).all()
    assert np.array_equal(truth,banks[entry['bank']]['tensors']['target_future'].numpy())
    error=pred-truth.astype(np.float64)
    expected={'MAE':np.abs(error).mean(),'MSE':np.square(error).mean(),'RMSE':np.sqrt(np.square(error).mean()),'channel_MAE':np.abs(error).mean((0,1)),'channel_MSE':np.square(error).mean((0,1)),'horizon_MAE':np.abs(error).mean((0,2)),'horizon_MSE':np.square(error).mean((0,2))}
    row=by_id[(entry['tag'],entry['bank'])]
    for key,value in expected.items():
        diff=float(np.max(np.abs(np.asarray(row[key])-value)));max_error=max(max_error,diff);assert diff<1e-12,(entry['tag'],entry['bank'],key,diff)
    total_windows+=entry['n']
    if entry['tag'] not in ['persistence','ridge']:
        p=OUT/(entry['tag']+'__'+entry['bank']+'__native_mechanisms.json')
        native=json.loads(p.read_text());counts=native['_native_score_counts'];totals=native['_native_score_totals']
        assert native['n_val']==entry['n'] and native['R_same_max']==0.
        for key,count in counts.items():
            assert count>0 and abs(native[key]-totals[key]/count)<1e-12,(entry['tag'],key)
        assert abs(native['MAE']-row['MAE'])<2e-5 and abs(native['MSE']-row['MSE'])<2e-5
        native_cells+=1
seal_path=R/'outputs/SELECTION_SEAL_V6.json'
seal=json.loads(seal_path.read_text());assert len(seal['selected_checkpoints'])==40
for selected in seal['selected_checkpoints']:
    assert sha(selected['checkpoint'])==selected['sha256']
    fit=json.loads(pathlib.Path(selected['summary_path']).read_text())
    assert fit['checkpoint_sha256']==selected['sha256'] and not fit['transfer_evaluation_opened']
forward=json.loads((OUT/'forward_checks.json').read_text());assert len(forward)==40
assert all(e['native_probe_vs_train_forward_max_error']<2e-5 and e['future_target_corruption_max_error']==0 for e in forward)
assert total_windows==40572 and native_cells==80
result={'utc':now(),'status':'PASS_RETAINED_FORECAST_ARITHMETIC_AND_NATIVE_SCORE_DENOMINATORS','groups':84,'model_window_predictions':total_windows,'scalar_predictions':total_windows*64,'selected_models':40,'native_mechanism_cells':native_cells,'max_forecast_arithmetic_error':max_error,'native_probe_training_forward_checks':40,'future_target_corruption_checks':40,'source_files_rechecked':check_source(),'scope':'Independent NumPy forecast score arithmetic, input/target/selected-checkpoint provenance, native rate numerator/denominator arithmetic; native neural/shift functions are retained source, not an independent neural implementation or independent counterfactual truth.','summary_sha256':sha(OUT/'summary.json'),'raw_manifest_sha256':sha(OUT/'raw_manifest.json'),'selection_seal_sha256':sha(seal_path)}
dump(OUT/'verification.json',result);print(json.dumps(result),flush=True)
