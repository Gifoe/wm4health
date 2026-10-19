"""Independent NumPy arithmetic and source-preservation check; no fitting."""
import runtime_profile
import datetime, hashlib, json, pathlib, zipfile
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OUT=ROOT/'outputs/native_chart'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def diff(a,b):
    delta=(a-b).reshape(len(a),-1).astype(np.float64)
    left=a.reshape(len(a),-1).astype(np.float64)
    right=b.reshape(len(b),-1).astype(np.float64)
    return dict(relative_L2_mean=float(np.mean(np.linalg.norm(delta,axis=1)/np.maximum(np.linalg.norm(right,axis=1),1e-12))),
                absolute_max=float(np.max(np.abs(delta))),
                cosine_mean=float(np.mean(np.sum(left*right,axis=1)/np.maximum(np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1),1e-12))))

def main():
    result=json.loads((OUT/'results.json').read_text())
    verification=[]; arrays=[]; tensor_count=0; boundary_count=0
    for native in result['native_models']:
        dataset,horizon=native['dataset'],native['horizon']; cfg=native['config']
        path=OUT/f'{dataset}_H{horizon}_arrays.npz'
        with np.load(path,allow_pickle=False) as data:
            assert all(np.isfinite(data[key]).all() for key in data.files)
            tensor_count+=len(data.files)
            hist=data['historical_inputs']; length=cfg['seq_len']; cut=cfg['state_patch_len']-1
            np.testing.assert_array_equal(data['origins'],native['training_origins'])
            assert np.all(data['origins']+horizon+24<native['train_boundary'])
            boundary_count+=len(data['origins'])
            for shift in [1,8,24]:
                moved=data[f'inputs_shift{shift}']
                np.testing.assert_array_equal(hist[:,shift+cut:],moved[:,cut:length-shift])
                eps=cfg.get('revin_eps',1e-5)
                np.testing.assert_allclose(data[f'mean_shift{shift}'],moved.mean(1,keepdims=True),atol=2e-6,rtol=2e-6)
                np.testing.assert_allclose(data[f'std_shift{shift}'],np.sqrt(moved.var(1,keepdims=True)+eps),atol=2e-6,rtol=2e-6)
                row=next(r for r in result['rows'] if r['dataset']==dataset and r['horizon']==horizon and r['shift']==shift)
                entry={'dataset':dataset,'horizon':horizon,'shift':shift,'metrics':{}}
                for family in ['target','condition']:
                    old=data[f'old_{family}_states'][:,shift+cut:length]
                    for chart,suffix in [('current',''),('fixed','_fixed')]:
                        other=data[f'{family}{suffix}_shift{shift}'][:,cut:length-shift]
                        actual=diff(old,other)
                        key=f'{family}_{chart}_chart_'+('difference' if chart=='current' else 'control')
                        expected=row[key]
                        for metric,value in actual.items():
                            assert abs(value-expected[metric])<2e-6,(key,metric,value,expected[metric])
                        if chart=='fixed':np.testing.assert_array_equal(old,other)
                        entry['metrics'][key]=actual
                verification.append(entry)
        arrays.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    source_root=PROJECT/'upstream/WorldTS'
    assert source_root.is_dir()
    source_records=[]
    archive=ROOT/'outputs/native_worldts_executed_source.zip'
    assert not archive.exists(), 'Do not overwrite a collected source snapshot'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as bundle:
        for record in result['executed_source_files']:
            p=PROJECT/record['path']
            assert p.resolve().is_relative_to(source_root.resolve())
            assert p.is_file() and p.suffix=='.py'
            assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
            bundle.writestr(record['path'],p.read_bytes())
            source_records.append(record)
    payload={'verified':True,'independent_arithmetic':'NumPy float64 from saved GPU float32 tensors; not a new training run',
             'metric_tolerance':2e-6,'rows':verification,'npz_files':arrays,'arrays_checked':tensor_count,'train_origins_checked':boundary_count,
             'native_source_files':source_records,'source_zip_sha256':sha(archive),'source_zip_bytes':archive.stat().st_size,
             'timing_interpretation':{'component_wall_fraction_descriptive_only':True,
               'not_an_additive_decomposition':True,'native_forecast_contains_validation_sync_omitted_by_component_helper':True,
               'legacy_field_encoder_elimination_ideal_speedup_upper_bound_is_not_a_rigorous_bound':True,
               'actual_cache_method_or_speedup_measured':False},
             'limitations':['No CPU native checkpoint forecast replay in this verifier','No new fit, held-out forecast test, causal evaluation, or cached-state implementation'],
             'results_sha256':sha(OUT/'results.json'),'objective_achieved':False,'novel_direction_accepted':False,
             'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    output=ROOT/'outputs/native_chart_independent_verification.json'
    assert not output.exists()
    output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({'verified':True,'rows':len(verification),'source_files':len(source_records),'npz_files':len(arrays),'array_count':tensor_count}))

if __name__=='__main__':main()
