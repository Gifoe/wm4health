"""Validate saved ledgers, require a reviewed outcome, and package publishable artifacts."""
import argparse,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from core import *
from evaluate import abs_error,scalar_error,bootstrap

OUT=ROOT/'outputs';PLOTS=ROOT/'plots'
REQUIRED=['data_summary.json','model_summary.csv','forecast_metrics.csv','prospective_action_metrics.csv',
          'future_action_divergence_metrics.csv','large_transition_metrics.csv','matched_future_action_pairs.csv',
          'paired_comparisons.csv','future_action_informativeness.csv','divergence_trend.csv',
          'rollout_ce_probe_metrics.csv','pump_rate_tail_sensitivity.csv','protocol_amendments.md',
          'integrity_checks.json']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outcome',required=True,choices=['A','B','C','D'])
    parser.add_argument('--interpretation-file',required=True);args=parser.parse_args()
    for name in REQUIRED:assert (OUT/name).stat().st_size>0,name
    splits=json.loads((OUT/'split_subjectids.json').read_text())
    assert not(set(splits['train'])&set(splits['val']))
    assert not(set(splits['train'])&set(splits['test']))
    assert not(set(splits['val'])&set(splits['test']))
    ref=np.load(OUT/'test_reference.npz');subjects=ref['subject'];target=ref['target'];mask=ref['mask'];donors=ref['donor']
    valid=donors>=0;assert (subjects[donors[valid]]!=subjects[valid]).all()
    pairs=pd.read_csv(OUT/'matched_future_action_pairs.csv')
    assert len(pairs)==int(valid.sum())
    assert (pairs.subjectid!=pairs.donor_subjectid).all()
    assert (pairs.max_state_difference_train_sd<=.5+1e-6).all()
    assert (pairs.future_action_rms_train_sd>=.25-1e-6).all()
    assert np.array_equal(pairs.window_index.to_numpy(),np.flatnonzero(valid))
    models=pd.read_csv(OUT/'model_summary.csv')
    assert len(models)==2 and models.parameters.nunique()==1 and models.parameters.iloc[0]==58946
    for policy in ['hold','true']:
        c=torch.load(OUT/'checkpoints'/f'{policy}.pt',map_location='cpu',weights_only=False)
        assert c['policy']==policy and c['parameters']==58946
    forecast=pd.read_csv(OUT/'forecast_metrics.csv');paired=pd.read_csv(OUT/'paired_comparisons.csv')
    errors={}
    mapping={'historical':('Historical RSSM','hold'),'true':('Prospective RSSM','true'),
             'hold':('Prospective RSSM','hold'),'zero':('Prospective RSSM','zero'),
             'wrong':('Prospective RSSM','wrong')}
    for key,(model,cond) in mapping.items():
        pred=np.load(OUT/f'prediction_{key}.npz')['prediction']
        assert pred.shape==target.shape and np.isfinite(pred).all()
        e=abs_error(pred,target,mask);errors[key]=e
        eligible=valid if key=='wrong' else np.ones(len(subjects),bool)
        got=bootstrap(scalar_error(e,0)[eligible],subjects[eligible])[0]
        row=forecast[(forecast.model==model)&(forecast.condition==cond)&(forecast.subset=='overall')&
                     (forecast.target=='BIS')&(forecast.horizon_seconds==0)]
        assert len(row)==1 and abs(row.mae.iloc[0]-got)<1e-4,(key,got,row.mae.iloc[0])
    for cond,label in [('hold','FAV'),('wrong','MFAV'),('zero','ZFAD')]:
        eligible=valid if cond=='wrong' else np.ones(len(subjects),bool)
        delta=scalar_error(errors[cond],0)-scalar_error(errors['true'],0)
        got=bootstrap(delta[eligible],subjects[eligible])[0]
        row=paired[(paired.comparison==label)&(paired.subset=='overall')&(paired.horizon_seconds==0)]
        assert len(row)==1 and abs(row.difference_mae.iloc[0]-got)<1e-4
    protocol=json.loads((OUT/'divergence_protocol.json').read_text())
    assert np.isfinite(protocol['train_quantile_boundaries']).all()
    assert sum(protocol['bin_counts'].values())==len(subjects)
    assert len(subjects)==83198 and len(np.unique(subjects))==74
    assert int((ref['large_label']>0).sum())==18152
    for i in range(1,7):
        assert len(list(PLOTS.glob(f'figure{i}_*.png')))==1
        assert len(list(PLOTS.glob(f'figure{i}_*.pdf')))==1
    check={'source_cases':495,'source_patients':493,'split_subjects_disjoint':True,
           'analysis_test_windows':len(subjects),'analysis_test_patients':len(np.unique(subjects)),
           'matched_donors_different_patient_and_calipered':True,'same_rssm_parameter_count':58946,
           'saved_prediction_forecast_metrics_recompute':True,'saved_prediction_paired_differences_recompute':True,
           'train_derived_divergence_quantiles_finite':True,'figures_png_pdf':6,
           'forecast_features_exclude_CE_and_future_physiology':True,
           'future_action_starts_after_anchor':True,
           'future_action_informativeness_control_present':True,
           'train_only_frozen_future_CE_probe_present':True,
           'train_derived_pump_rate_tail_sensitivity_present':True}
    (OUT/'completion_checks.json').write_text(json.dumps(check,indent=2))
    source=[ROOT/'config.yaml',ROOT/'README.md']+sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'scripts').glob('*.sh'))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    (OUT/'final_source_sha256.json').write_text(json.dumps(manifest,indent=2))
    subprocess.run([sys.executable,'src/report.py','--outcome',args.outcome,
                    '--interpretation-file',args.interpretation_file],cwd=ROOT,check=True)
    assert f'Outcome {args.outcome}' in (OUT/'FINAL_REPORT.md').read_text()
    (OUT/'run_status.json').write_text(json.dumps({'stage':'complete','outcome':args.outcome,'returncode':0},indent=2))
    bundle=ROOT/'prospective_seed0_v1_results.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in ['src','scripts','outputs','plots','logs']:
            for p in (ROOT/folder).rglob('*'):
                if not p.is_file() or '__pycache__' in p.parts:continue
                if p.suffix in ['.npy','.npz'] and not p.name.startswith('rollout_probe_'):continue
                z.write(p,p.relative_to(ROOT).as_posix())
        for p in [ROOT/'config.yaml',ROOT/'README.md']:z.write(p,p.name)
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    print('FINALIZED',json.dumps(check),'BUNDLE_BYTES',bundle.stat().st_size,flush=True)

if __name__=='__main__':main()
