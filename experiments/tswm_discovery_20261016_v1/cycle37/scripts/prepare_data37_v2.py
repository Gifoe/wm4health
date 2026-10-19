"""Native hour0..23 parser and explicit feature allowlists; preserve failed V1."""
import pathlib,json,hashlib,re
import pandas as pd
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];raw=R/'data';O=R/'data_v2'
assert not O.exists();O.mkdir()
def read(name):return pd.read_csv(raw/name)
w=read('wind_zone1_original.csv');sy=read('solar_train15_original.csv');sx=read('solar_predictors15_original.csv')
sy=sy[sy.ZONEID==1];sx=sx[sx.ZONEID==1]
sw=[x for x in sx if re.fullmatch('VAR[0-9]+',x)];assert len(sw)==12
assert not sy.duplicated(['ZONEID','TIMESTAMP']).any() and not sx.duplicated(['ZONEID','TIMESTAMP']).any()
# predictors15 includes POWER: it is never a feature or a source of labels.
s=sy.merge(sx[['ZONEID','TIMESTAMP']+sw],on=['ZONEID','TIMESTAMP'],how='left',validate='one_to_one')
audit={}
for name,df,target,weather in [('wind',w,'TARGETVAR',['U10','V10','U100','V100']),('solar',s,'POWER',sw)]:
    assert (df.ZONEID==1).all()
    parts=df.TIMESTAMP.str.strip().str.split(' ',n=1,expand=True)
    source_date=pd.to_datetime(parts[0],format='%Y%m%d')
    hour=parts[1].str.split(':').str[0].astype(int);minute=parts[1].str.split(':').str[1].astype(int)
    assert hour.between(0,23).all() and (minute==0).all()
    valid=source_date+pd.to_timedelta(hour,unit='h')
    d=df.assign(valid_time=valid).sort_values('valid_time').reset_index(drop=True)
    assert not d.valid_time.duplicated().any()
    assert np.isfinite(d[[target]+weather].values.astype(float)).all()
    delta=d.valid_time.diff().dropna().dt.total_seconds()
    assert (delta==3600).all()
    # Original daily+24h contract: use origin D00 and target stamps D01..D+1 00.
    d['forecast_origin_day']=(d.valid_time-pd.Timedelta(hours=1)).dt.floor('D')
    d['lead_hour']=(d.valid_time-d.forecast_origin_day).dt.total_seconds()/3600
    assert d.lead_hour.between(1,24).all()
    canonical=d[['valid_time','forecast_origin_day','lead_hour',target]+weather].rename(columns={target:'y'})
    dst=O/f'{name}_zone1_canonical.csv';canonical.to_csv(dst,index=False,float_format='%.17g')
    audit[name]={'rows':len(d),'zone':1,'weather_columns':weather,'feature_target_disjoint':True,
        'valid_min':str(d.valid_time.min()),'valid_max':str(d.valid_time.max()),
        'days':d.forecast_origin_day.nunique(),'complete_24h_days':int((d.groupby('forecast_origin_day').size()==24).sum()),
        'timestamp_hour_values':sorted(hour.unique().tolist()),'missing_cells':int(canonical.isna().sum().sum()),
        'canonical_sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),
        'weather_truth_labels_available':False,'timezone':'original benchmark clock, unspecified',
        'forecast_issue_contract':'organizer daily-midnight+24h; no per-row actual issuance field',
        'target_source':'only original Task15 train; no solution file or predictors POWER used'}
result={'datasets':audit,'solar_predictors_POWER_explicitly_excluded':True,
    'hour_parser_fix':'native source uses 0..23; rejected V1 1..24 assumption, no data moved or imputed',
    'raw_sources':[{'path':'data/'+p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(raw.glob('*original.csv'))],
    'daily_grouping':'valid_time minus1h floorD, target leads1..24 according to declared organizer contract',
    'not_independently_logged_historical_release_replay':True,'no_model_scores':True,
    'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
(O/'data_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
