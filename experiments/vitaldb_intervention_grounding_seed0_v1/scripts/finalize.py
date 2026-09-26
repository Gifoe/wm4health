"""Verify ledger consistency and package compact, secret-free deliverables."""
import datetime,hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'src'))
from core import filename,MODEL_NAMES
out=root/'outputs'
required=['data_summary.json','split_caseids.json','forecast_metrics.csv','ce_probe_metrics.csv',
          'time_shift_metrics.csv','action_ablation_metrics.csv','transition_event_metrics.csv','FINAL_REPORT.md']
for name in required: assert (out/name).stat().st_size>0,name
summary=json.loads((out/'data_summary.json').read_text())
assert summary['selected_before_qc']<=500 and summary['waveform_downloads']==0
splits=json.loads((out/'split_subjectids.json').read_text())
assert sum(map(len,splits.values()))==len(set(sum(splits.values(),[])))
ref=np.load(out/'test_reference.npz'); target=ref['y'][:,:,0]; mask=ref['mask'][:,:,0]; subjects=ref['subject']
forecast=pd.read_csv(out/'forecast_metrics.csv'); timing=pd.read_csv(out/'time_shift_metrics.csv')
consistency={}
for name in MODEL_NAMES:
    pred=np.load(out/f'predictions_{filename(name)}.npz')['prediction'][:,:,0]
    e=np.where(mask,np.abs(pred-target),np.nan)
    for h in [0,3,6,18,30]:
        v=np.nanmean(e,axis=1) if h==0 else e[:,h-1]
        actual=pd.DataFrame({'p':subjects,'v':v}).groupby('p').v.mean().mean()
        saved=forecast[(forecast.model==name)&(forecast.subset=='overall')&(forecast.target=='BIS')&(forecast.horizon_seconds==h*10)].mae.iloc[0]
        assert abs(actual-saved)<2e-5,(name,h,actual,saved)
    consistency[name]=True
for (name,subset,h),d in timing.groupby(['model','subset','horizon_seconds']):
    zero=d[d.shift_seconds==0].iloc[0]
    f=forecast[(forecast.model==name)&(forecast.subset==subset)&(forecast.horizon_seconds==h)&(forecast.target=='BIS')]
    assert abs(zero.mae-f.mae.iloc[0])<2e-5
    assert abs(zero.normalized_degradation)<1e-10
probe_check={}
for name in MODEL_NAMES[1:]:
    z=np.load(out/f'predictions_{filename(name)}.npz')['latent'][::79]
    saved=np.load(out/f'ce_prediction_{filename(name)}.npy')[::79]
    for j in range(2):
        p=np.load(out/f'probe_{filename(name)}_{j}.npz')
        actual=((z-p['mean'])/p['std'])@p['coef']+p['intercept']
        assert np.allclose(actual,saved[:,j],atol=1e-6)
    probe_check[name]=True
for number in range(1,9):
    assert len(list((root/'plots').glob(f'figure{number}_*.png')))==1
    assert len(list((root/'plots').glob(f'figure{number}_*.pdf')))==1
report=(out/'FINAL_REPORT.md').read_text()
assert '**Outcome C' in report
assert 'Outcome A' not in report and 'Outcome B' not in report
assert 'post-diagnostic' in report and 'supplementary' in report
check={'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'outcome':'C',
       'required_outputs_present':True,'patients_disjoint':True,'numeric_downloads_only':True,
       'forecast_ledger_recomputed_from_saved_predictions':consistency,
       'timing_zero_matches_forecast':True,'saved_frozen_probes_reproduce_predictions':probe_check,
       'figures_png_and_pdf':8,'post_diagnostic_protocol_extension_disclosed':True}
(out/'completion_checks.json').write_text(json.dumps(check,indent=2))
(out/'run_status.json').write_text(json.dumps({'stage':'complete','returncode':0,'outcome':'C',
    'completed_at_utc':check['completed_at_utc'],'report':'outputs/FINAL_REPORT.md'},indent=2))
paths=[root/'config.yaml',root/'README.md']+list((root/'src').glob('*.py'))+list((root/'scripts').glob('*.py'))
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(out/'final_source_sha256.json').write_text(json.dumps(manifest,indent=2))
bundle=root/'vitaldb_seed0_v1_results.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for folder in ['src','scripts','outputs','plots','logs']:
        for p in (root/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts: continue
            if p.suffix in ['.npy','.npz'] and not p.name.startswith('probe_'): continue
            if p.name=='remote_session.py': continue
            z.write(p,p.relative_to(root).as_posix())
    for p in [root/'config.yaml',root/'README.md']: z.write(p,p.relative_to(root).as_posix())
    for name in ['cases.csv','trks.csv','track_names.csv']:
        p=root/'data/raw'/name; z.write(p,p.relative_to(root).as_posix())
print(json.dumps(check),flush=True)
print('BUNDLE',bundle,bundle.stat().st_size,flush=True)
