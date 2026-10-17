"""Independently check retained masks/rows and close the negative native audit."""
import csv
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

R=pathlib.Path(__file__).resolve().parents[1]
storage=json.loads((R.parent/'STORAGE_LOCATION_V1.json').read_text())
OUT=pathlib.Path(storage['artifact_root'])/'cycle52/native_audit_v3'
audit=json.loads((OUT/'audit.json').read_text())
assert audit['status']=='COMPLETE_REAL_DATA_INPUT_CONTRACT_AUDIT'
verification_path=R/'outputs/verification_v1.json'
assert not verification_path.exists()
rows=list(csv.DictReader((OUT/'window_audit.csv').open(encoding='utf-8')))
assert len(rows)==audit['overall']['accepted_windows']==1781
counts=[]
for path in sorted(OUT.glob('raw_exp*.npz')):
    with np.load(path,allow_pickle=False) as raw:
        starts=raw['starts'];target=raw['target'];exo=raw['exog']
        boundary=target[starts+255]
        exoboundary=exo[starts+255]
        assert np.isfinite(boundary).all() and np.isfinite(exoboundary).all()
        counts.append({'snapshot':path.name,'accepted_windows':len(starts),'target_missing_boundary_windows':int(np.isnan(boundary).any(axis=1).sum()),'exog_missing_boundary_windows':int(np.isnan(exoboundary).any(axis=1).sum()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
assert len(counts)==12 and sum(c['accepted_windows'] for c in counts)==len(rows)
for row in rows:
    assert float(row['target_max_abs_diff'])==float(row['perturb_target_max_abs_diff'])==float(row['exog_max_abs_diff'])==0
acq=json.loads((R/'outputs/acquisition_v2/receipt.json').read_text())
file_count=0
for archive in acq['archives']:
    for item in archive['files']:
        original=R/item['local_path']
        copied=pathlib.Path(storage['data_root'])/original.relative_to(R/'data')
        assert hashlib.sha256(original.read_bytes()).hexdigest()==hashlib.sha256(copied.read_bytes()).hexdigest()==item['sha256']
        file_count+=1

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
up=R.parent/'upstream/TSWM-Official-Repo'
native=load('greenhouse_verify52',up/'dataset_utils/greenhouse/dataset.py')
window=load('window_verify52',up/'rq2_latent_vs_ob/common/windowing.py')
frame=pd.DataFrame({'Date':pd.date_range('2022-01-01',periods=272,freq='5min')})
for channels in [native.TARGET_CHANNELS,native.ACTION_CHANNELS,native.EXOG_CHANNELS]:
    for name,lo,hi in channels:frame[name]=(lo+hi)/2
frame['Tair']=10.;frame.loc[255,'Tair']=np.nan
check=np.concatenate([frame[[c[0] for c in native.TARGET_CHANNELS]].to_numpy(dtype=np.float32),frame[[c[0] for c in native.EXOG_CHANNELS]].to_numpy(dtype=np.float32)],axis=1)
assert native._window_continuous_ok(check,.1,6)
ds=object.__new__(native.GreenhouseDataset)
ds.seq_len=272;ds.target_channels=native.TARGET_CHANNELS;ds.action_channels=native.ACTION_CHANNELS;ds.exog_channels=native.EXOG_CHANNELS
ds._tgt_names=[x[0] for x in ds.target_channels];ds._act_names=[x[0] for x in ds.action_channels];ds._exo_names=[x[0] for x in ds.exog_channels]
ds._windows=[('fixture',0)];ds._team_meta={'fixture':{}};ds._team_df={'fixture':frame}
schema=window.discover_schema(ds[0]);before=window.unified_window(ds[0],256,16,schema)['target_history']
frame.loc[256:,'Tair']=30.
after=window.unified_window(ds[0],256,16,schema)['target_history']
positive=float(np.abs(after-before).max())
assert abs(positive-2/3)<2e-7
frame.loc[255,'Tair']=10.;null_before=window.unified_window(ds[0],256,16,schema)['target_history']
frame.loc[256:,'Tair']=20.;null_after=window.unified_window(ds[0],256,16,schema)['target_history']
assert np.array_equal(null_before,null_after)
for key,path in [('dataset',up/'dataset_utils/greenhouse/dataset.py'),('windowing',up/'rq2_latent_vs_ob/common/windowing.py')]:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['source_sha256'][key]
verification={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_REAL_BOUNDARY_MASKS_AND_POSITIVE_CONTROL','independent_snapshot_checks':counts,'native_accepted_windows':len(rows),'boundary_missing_windows':0,'future_only_perturbation_changed_real_histories':0,'positive_control_normalized_history_change':positive,'positive_control_accepted_by_native_filter':True,'no_missing_boundary_control_change':0,'source_and_large_disk_files_reverified':file_count,'audit_sha256':hashlib.sha256((OUT/'audit.json').read_bytes()).hexdigest()}
verification_path.write_text(json.dumps(verification,indent=2),encoding='utf-8')
(OUT/'verification_v1.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')

lines=['| Public experiment | Split | Accepted windows | Histories using future targets |','|---|---|---:|---:|']
for group in audit['by_experiment_and_split']:
    lines.append(f"| {group['experiment']} | {group['split']} | {group['accepted_windows']} | {group['target_history_changed_windows']} |")
table='\n'.join(lines)
summary=f'''# C52 real-data input-contract audit

Decision: SOURCE_BOUNDARY_RISK_NOT_ACTIVE_UNDER_NATIVE_DEFAULT_WINDOWS.
Accepted final research directions: 0. Overall discovery remains active and incomplete.

The exact original third Autonomous Greenhouse Challenge data (DOI [10.4121/21960932.v1](https://doi.org/10.4121/21960932.v1), CC BY 4.0) were selectively acquired from the original archive byte ranges. Two full archives total 45,381,396,051 bytes; their directories required 4,983,156 transferred bytes. Time-series-only acquisition transferred 31,567,900 bytes and extracted 40 XLSX/DOCX files totaling 32,517,760 bytes. No image members were downloaded. Selected-member ZIP CRC32, HTTPS archive ETags, local SHA-256 and the copies on the large disk were checked; the full original archive MD5 values are declared source metadata, not independently verified full-download hashes.

The pinned unchanged TSWM loader and windowing source were run for both public experiment batches separately: context 256, horizon 16, stride 80, native NaN limits .1/6, split seed 0, held-out team ratio .2. Twelve team-batch groups produced 1,781 accepted windows.

{table}

Every accepted window has finite target and exogenous values at the final history index. Native full-window interpolation and history-only interpolation produce identical normalized histories; changing only future targets also changes no real accepted history. Independent retained-mask and row checks agree. A constructed valid positive control still changes normalized history by {positive:.9f}, confirming that the audit detects the source defect when its boundary condition exists; a finite-boundary control changes by zero.

The previous constructed-window result remains valid, but it does not explain behavior on these default real-data windows. Do not propose a missingness repair or claim reported Greenhouse performance was inflated from that fixture. No model training or predictive comparison was performed in C52.

The two public experiment archives must not be merged or treated as independent real domains. The paper repository does not uniquely identify which archive generated its reported Greenhouse table; this source-version ambiguity remains. Accepted windows overlap and are not independent statistical replicates. Exogenous future values may be declared model inputs, so exogenous-history dependence would require a separate information-availability interpretation.

Evidence: [machine-readable audit](outputs/native_audit_complete_v3/audit.json), [window rows](outputs/native_audit_complete_v3/window_audit.csv), [verification](outputs/verification_v1.json), [acquisition](outputs/acquisition_v2/receipt.json), [protocol](PROTOCOL.md).

Storage: user-authorized large-disk directory recorded in `outputs/storage_v1.json`; raw data and retained arrays stay on the server. No result synchronization to the local workstation after the user's instruction. Code, configuration, aggregate results and provenance are prepared for Gifoe/wm4health.
'''
reports={
 'experimental_results.md':summary,
 'negative_evidence.md':summary+'\nRejected explanation: the constructed interpolation defect is active on the native default accepted Greenhouse windows. Observation count 0/1781. This says nothing about other strides, datasets, future missingness patterns or downstream models.\n',
 'baseline_registry.md':'# C52 baseline registry\n\nPinned TSWM source 2e5e48540ff0da48e8192163969a6c58df1d5a3e. Native real loader/windowing executed unchanged on two original public Greenhouse experiment batches, 12 team-batch groups, 1781 accepted windows. No new model was trained; no original paper headline reproduced. Full original source data-version ambiguity remains.\n',
 'literature_review.md':'# C52 source and novelty boundary\n\nSource: [TSWM official repository](https://github.com/HaochenZhang717/TSWM-Official-Repo) at commit 2e5e48540ff0da48e8192163969a6c58df1d5a3e. Data: [third Autonomous Greenhouse Challenge](https://doi.org/10.4121/21960932.v1), Petropoulou et al., 2023, CC BY 4.0. This is a source/data qualification audit and makes no new-method novelty claim. C49 literature and negative conditional-score outcome are retained separately.\n',
 'hypothesis_space.md':'# C52 competing explanations\n\nH1: future-target interpolation leakage affects the native default real windows. Refuted for both acquired public batches under the checked protocol: 0/1781 target histories change; all boundary masks are finite.\n\nH2: an observed model failure comes from the data-version choice. Unresolved; no model was trained, and the source repository does not specify which public batch generated its headline.\n\nH3: a shared learned-dynamics deficiency survives competent baselines across systems. Still requires direct model evidence; this preprocessing audit does not establish it.\n',
 'research_ideas.md':'# C52 direction decision\n\nNo new direction is accepted. A source interpolation fixture cannot establish a world-model research gap and is inactive on the native default real windows. Stop this repair avenue. Return to source-grounded model failures; use existing ordinary controls and avoid duplicating completed negative cycles. The candidate must concern a reusable time-series world-model mechanism and survive independent held-out evaluation.\n',
 'final_research_direction.md':'# Current research status after C52\n\nAccepted final directions: 0. Overall goal ACTIVE_UNACHIEVED.\n\nC49 observation-marginalized conditional scoring has no stable quality/cost benefit over complete-clock ELBO and is deprioritized. C52 finds the native interpolation fixture inactive on 1781 original-data windows, removing it as an explanation under the checked protocol. A defensible novel world-model problem and useful independently validated minimal method remain missing. The next model experiment must first resolve or explicitly separate the two original Greenhouse batches and inspect relevant existing results in the user-selected wm4health repository.\n',
 'README.md':summary,
}
for name,text in reports.items():
    dest=R/name;assert not dest.exists();dest.write_text(text,encoding='utf-8')
env={'python':sys.version,'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','torch','openpyxl','et-xmlfile']},'upstream_commit':audit['upstream_commit'],'source_sha256':audit['source_sha256']}
(R/'outputs/environment_v1.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
manifest={'status':'C52_REPORTS_AND_VERIFICATION_COMPLETE','accepted_final_directions':0,'reports':[{'path':name,'sha256':hashlib.sha256((R/name).read_bytes()).hexdigest()} for name in reports],'verification_sha256':hashlib.sha256(verification_path.read_bytes()).hexdigest()}
(R/'FINAL_REPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'status':manifest['status'],'real_windows':1781,'affected':0,'positive_control':positive,'verified_data_files':file_count,'reports':len(reports)}),flush=True)
