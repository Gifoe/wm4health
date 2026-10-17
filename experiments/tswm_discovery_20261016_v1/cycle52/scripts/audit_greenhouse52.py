"""Real-data prevalence of the pinned native TSWM interpolation boundary issue."""
import copy
import csv
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import random
import sys
import traceback
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

R = pathlib.Path(__file__).resolve().parents[1]
UP = R.parent / 'upstream/TSWM-Official-Repo'
OUT = R / 'outputs/native_audit_v1'
assert not OUT.exists(), 'Version output rather than overwrite evidence'
OUT.mkdir(parents=True)
H, F, STRIDE = 256, 16, 80
TOL = 2e-7

def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save_status(**kwargs):
    (OUT / 'status.json').write_text(json.dumps({'utc':stamp(),'pid':os.getpid(),**kwargs},indent=2),encoding='utf-8')
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module

sources = {
    'dataset':(UP/'dataset_utils/greenhouse/dataset.py','f3f737d925df34e13da85eb81732e47ce9a2759f1dcd6a00d3d35295ceb9efc6'),
    'windowing':(UP/'rq2_latent_vs_ob/common/windowing.py','280f175e4bd4056ec0604a9f757f8bf67fc74b034fa2e694875a48bcc3eb8184'),
}
for path, expected in sources.values(): assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
save_status(status='RUNNING_NATIVE_DATA_AUDIT',phase='loading_source',completed_teams=0)

def summarize(rows):
    n=len(rows)
    return {
        'accepted_windows':n,
        'target_history_changed_windows':sum(r['target_max_abs_diff']>TOL for r in rows),
        'target_history_changed_fraction':sum(r['target_max_abs_diff']>TOL for r in rows)/n if n else None,
        'exog_history_changed_windows':sum(r['exog_max_abs_diff']>TOL for r in rows),
        'future_target_perturbation_changed_history_windows':sum(r['perturb_target_max_abs_diff']>TOL for r in rows),
        'max_target_history_change_normalized':max((r['target_max_abs_diff'] for r in rows),default=0.),
        'max_exog_history_change_normalized':max((r['exog_max_abs_diff'] for r in rows),default=0.),
        'max_future_target_perturbation_history_change':max((r['perturb_target_max_abs_diff'] for r in rows),default=0.),
        'target_history_rms_diff_across_windows':float(np.sqrt(np.mean([r['target_mse_diff'] for r in rows]))) if n else None,
        'null_boundary_windows':sum(r['target_boundary_nan_count']==0 for r in rows),
        'null_boundary_perturbation_failures':sum(r['target_boundary_nan_count']==0 and r['perturb_target_max_abs_diff']>TOL for r in rows),
    }

try:
    native=load_module('native_greenhouse_audit52',sources['dataset'][0])
    window=load_module('native_windowing_audit52',sources['windowing'][0])
    rows=[]; teams_out=[]; example_count=0
    for exp in [1,2]:
        data_root=R/f'data/greenhouse3_original_v1/experiment_{exp}/TimeSeries'
        team_names=sorted(p.name for p in data_root.iterdir() if p.is_dir() and (p/'GreenhouseClimate.xlsx').is_file())
        assert len(team_names)==6
        shuffled=team_names.copy();random.Random(0).shuffle(shuffled)
        n_val=max(1,round(len(shuffled)*.2));val_names=set(shuffled[:n_val])
        readme=data_root/'ReadMe.docx'
        if readme.is_file():
            with zipfile.ZipFile(readme) as z:
                xml=ET.fromstring(z.read('word/document.xml'))
            paragraphs=[''.join(p.itertext()) for p in xml.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
            (OUT/f'experiment_{exp}_readme.txt').write_text('\n'.join(paragraphs),encoding='utf-8')
        for team in team_names:
            print(json.dumps({'utc':stamp(),'phase':'native_load','experiment':exp,'team':team}),flush=True)
            ds=native.GreenhouseDataset(data_root,teams=[team],seq_len=H+F,stride=STRIDE,cache=False)
            df=ds._team_df[team]
            team_rows=[]
            schema=window.discover_schema(ds[0]) if len(ds) else None
            raw_target=df[ds._tgt_names].to_numpy(dtype='float32')
            raw_exog=df[ds._exo_names].to_numpy(dtype='float32')
            raw_action=df[ds._act_names].to_numpy(dtype='float32')
            np.savez_compressed(OUT/f'raw_exp{exp}_{team}.npz',target=raw_target,exog=raw_exog,action=raw_action,date_ns=df['Date'].astype('int64').to_numpy(),starts=np.array([s for t,s in ds._windows],dtype=np.int64))
            for idx,(native_team,start) in enumerate(ds._windows):
                assert native_team==team
                sample=ds[idx]
                u=window.unified_window(sample,H,F,schema)
                raw=raw_target[start:start+H+F]
                exo=raw_exog[start:start+H+F]
                safe_t=native.GreenhouseDataset._norm(native._interp_nan_columns(raw[:H]),ds.target_channels,-1.,1.)
                safe_e=native.GreenhouseDataset._norm(native._interp_nan_columns(exo[:H]),ds.exog_channels,-1.,1.)
                delta_t=u['target_history']-safe_t
                delta_e=u['exog_history']-safe_e
                # Reuse unchanged native __getitem__ on a retained copy. Only
                # future target values change; NaN mask and all inputs stay fixed.
                changed=copy.copy(ds)
                small=df.iloc[start:start+H+F].copy().reset_index(drop=True)
                changed._team_df={team:small};changed._windows=[(team,0)]
                for name,lo,hi in ds.target_channels:
                    small.loc[H:,name]=small.loc[H:,name]+.3*(hi-lo)
                pu=window.unified_window(changed[0],H,F,schema)
                perturb=pu['target_history']-u['target_history']
                assert np.array_equal(pu['continuous_history'],u['continuous_history'])
                assert np.array_equal(pu['continuous_future'],u['continuous_future'])
                assert np.array_equal(pu['exog_history'],u['exog_history'])
                assert np.array_equal(pu['exog_future'],u['exog_future'])
                record={'experiment':exp,'team':team,'split':'val' if team in val_names else 'train','start':int(start),'history_last_timestamp':str(df['Date'].iloc[start+H-1]),'target_boundary_nan_count':int(np.isnan(raw[H-1]).sum()),'exog_boundary_nan_count':int(np.isnan(exo[H-1]).sum()),'target_history_nan_count':int(np.isnan(raw[:H]).sum()),'target_max_abs_diff':float(np.abs(delta_t).max()),'target_mse_diff':float(np.mean(delta_t.astype(np.float64)**2)),'exog_max_abs_diff':float(np.abs(delta_e).max()),'perturb_target_max_abs_diff':float(np.abs(perturb).max()),'target_changed_elements':int((np.abs(delta_t)>TOL).sum())}
                rows.append(record);team_rows.append(record)
                if record['target_max_abs_diff']>TOL and example_count<5:
                    np.savez_compressed(OUT/f'example_{example_count}.npz',raw_target=raw,native_target_history=u['target_history'],safe_target_history=safe_t,perturbed_target_history=pu['target_history'],channel_names=np.array(ds._tgt_names),record_json=np.array(json.dumps(record)))
                    example_count+=1
            intervals=df['Date'].diff().dt.total_seconds().div(60).dropna().value_counts()
            stats={'experiment':exp,'team':team,'split':'val' if team in val_names else 'train','raw_rows':len(df),'sessions':len(native._split_sessions(df)),'interval_minutes_counts':{str(k):int(v) for k,v in intervals.items()},'target_nan_fraction':{n:float(np.isnan(raw_target[:,j]).mean()) for j,n in enumerate(ds._tgt_names)},'exog_nan_fraction':{n:float(np.isnan(raw_exog[:,j]).mean()) for j,n in enumerate(ds._exo_names)},**summarize(team_rows)}
            teams_out.append(stats)
            (OUT/'team_progress.json').write_text(json.dumps(teams_out,indent=2),encoding='utf-8')
            save_status(status='RUNNING_NATIVE_DATA_AUDIT',phase='team_complete',completed_teams=len(teams_out),completed_windows=len(rows))
            print(json.dumps(stats),flush=True)
    with (OUT/'window_audit.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'utc':stamp(),'status':'COMPLETE_REAL_DATA_INPUT_CONTRACT_AUDIT','protocol':{'context':H,'horizon':F,'stride':STRIDE,'max_nan_ratio':.1,'max_nan_gap':6,'split_seed':0,'val_team_ratio':.2,'normalized_difference_tolerance':TOL},'upstream_commit':'2e5e48540ff0da48e8192163969a6c58df1d5a3e','source_sha256':{k:v for k,(p,v) in sources.items()},'acquisition_receipt_sha256':hashlib.sha256((R/'outputs/acquisition_v2/receipt.json').read_bytes()).hexdigest(),'overall':summarize(rows),'by_experiment_and_split':[{ 'experiment':exp,'split':split,**summarize([r for r in rows if r['experiment']==exp and r['split']==split])} for exp in [1,2] for split in ['train','val']],'teams':teams_out,'limitations':['Overlapping accepted windows are not independent samples.','Original repository does not uniquely identify which of the two public experiment archives generated the paper Greenhouse table. No headline reproduction is claimed.','Target future dependence violates the forecast input contract. Exogenous-history dependence is separately descriptive, because known future exogenous covariates may be declared inputs.','This does not measure downstream predictive/ranking impact, validate counterfactual effects or establish research novelty.']}
    assert result['overall']['null_boundary_perturbation_failures']==0
    for path, expected in sources.values(): assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save_status(status='COMPLETE_REAL_DATA_INPUT_CONTRACT_AUDIT',completed_teams=len(teams_out),completed_windows=len(rows))
    print(json.dumps({'status':result['status'],'overall':result['overall'],'groups':result['by_experiment_and_split']}),flush=True)
except BaseException as error:
    save_status(status='FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    raise
