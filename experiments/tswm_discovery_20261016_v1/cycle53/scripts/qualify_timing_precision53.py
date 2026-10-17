"""Distinguish subsecond source timestamps from material sampling irregularity."""
from native53 import *
import numpy as np
import pandas as pd
from dataset_utils.greenhouse.dataset import GreenhouseDataset

assert len(json.loads((R/'outputs/SELECTION_SEAL_V6.json').read_text())['selected_checkpoints'])==40
OUT=ART/'data_diagnostics_v2';assert not OUT.exists();OUT.mkdir()
teams=sorted(p.name for p in DATA.iterdir() if p.is_dir() and (p/'GreenhouseClimate.xlsx').exists() and p.name!='reference')
banks=[]
for name,root,subset,count in [('source_train',DATA,teams,815),('source_val',DATA,['reference'],163),
    ('new_crop_all',pathlib.Path(STORAGE['data_root'])/'greenhouse3_original_v1/experiment_2/TimeSeries',None,803)]:
    raw=GreenhouseDataset(root,teams=subset,seq_len=272,stride=80);assert len(raw)==count
    steps=[];forecast=[];bad_windows=0
    for team,start in raw._windows:
        dates=pd.to_datetime(raw._team_df[team].iloc[start:start+272].Date).to_numpy(dtype='datetime64[ns]')
        dt=np.diff(dates)/np.timedelta64(1,'s');error=np.abs(dt-300.)
        fe=(dates[256:]-dates[255])/np.timedelta64(1,'s')-300.*np.arange(1,17)
        steps.append(dt);forecast.append(fe);bad_windows+=bool(np.any(error>1.))
    steps=np.concatenate(steps);forecast=np.stack(forecast)
    error=np.abs(steps-300.)
    banks.append({'bank':name,'windows':len(raw),'window_weighted_steps':len(steps),
        'exactly_300_second_step_fraction':float(np.mean(steps==300.)),
        'within_one_second_step_fraction':float(np.mean(error<=1.)),
        'windows_with_step_deviation_over_one_second':int(bad_windows),
        'nonpositive_steps':int(np.sum(steps<=0)),
        'step_seconds_min':float(steps.min()),'step_seconds_max':float(steps.max()),
        'absolute_step_error_seconds_percentiles':{str(q):float(np.percentile(error,q)) for q in [0,50,95,99,100]},
        'max_absolute_forecast_elapsed_time_error_seconds':float(np.abs(forecast).max()),
        'horizon16_min_seconds':float((forecast[:,-1]+4800.).min()),
        'horizon16_max_seconds':float((forecast[:,-1]+4800.).max())})
result={'utc':now(),'status':'COMPLETE_TIMING_PRECISION_QUALIFICATION','one_second_tolerance':1.,'banks':banks,
    'all_accepted_windows_within_one_second_per_step':all(b['windows_with_step_deviation_over_one_second']==0 for b in banks),
    'scope':'Strict timestamp equality flags subsecond variation. Report actual extrema and a disclosed one-second numerical tolerance; this is not resampling, not a change to native input windows and not evidence of a dynamics-model gap.',
    'predecessor_summary_sha256':sha(ART/'data_diagnostics_v1/summary.json'),'source_files_rechecked':check_source(),'model_predictions_read':False}
dump(OUT/'summary.json',result);print(json.dumps(result),flush=True)
