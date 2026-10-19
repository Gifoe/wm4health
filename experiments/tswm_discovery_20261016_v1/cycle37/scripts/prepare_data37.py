"""Fixed zone1 original Task15 data, explicit hour-24 conversion, no model scores."""
import pathlib,zipfile,io,json,hashlib
import pandas as pd
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];D=R/'data_intake';O=R/'data'
assert not O.exists(),'Preserve prepared data';O.mkdir()
records=[];datasets={}
def extract(outer,chain,name):
    z=outer
    for member in chain[:-1]:z=zipfile.ZipFile(io.BytesIO(z.read(member)))
    b=z.read(chain[-1]);(O/name).write_bytes(b)
    records.append({'file':name,'member_chain':chain,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    return pd.read_csv(io.BytesIO(b))
def stamps(series):
    parts=series.str.strip().str.split(' ',n=1,expand=True)
    day=pd.to_datetime(parts[0],format='%Y%m%d')
    hour=parts[1].str.split(':').str[0].astype(int)
    minute=parts[1].str.split(':').str[1].astype(int)
    assert hour.between(1,24).all() and (minute==0).all()
    return day+pd.to_timedelta(hour,unit='h'),day,hour
with zipfile.ZipFile(D/'GEFCom2014_original.zip') as outer:
    wind=extract(outer,['GEFCom2014 Data/GEFCom2014-W_V2.zip','Wind/Task 15/Task15_W_Zone1_10.zip','Task15_W_Zone1_10/Task15_W_Zone1.csv'],'wind_zone1_original.csv')
    solar_y=extract(outer,['GEFCom2014 Data/GEFCom2014-S_V2.zip','Solar/Task 15/train15.csv'],'solar_train15_original.csv')
    solar_x=extract(outer,['GEFCom2014 Data/GEFCom2014-S_V2.zip','Solar/Task 15/predictors15.csv'],'solar_predictors15_original.csv')
    solar_y=solar_y[solar_y.ZONEID==1];solar_x=solar_x[solar_x.ZONEID==1]
    assert not solar_y.duplicated(['ZONEID','TIMESTAMP']).any()
    assert not solar_x.duplicated(['ZONEID','TIMESTAMP']).any()
    solar=solar_y.merge(solar_x,on=['ZONEID','TIMESTAMP'],how='left',validate='one_to_one')
for name,df,target in [('wind',wind,'TARGETVAR'),('solar',solar,'POWER')]:
    assert (df.ZONEID==1).all()
    t,days,hours=stamps(df.TIMESTAMP)
    df=df.assign(valid_time=t,forecast_day=days,source_hour=hours).sort_values('valid_time').reset_index(drop=True)
    assert not df.valid_time.duplicated().any()
    weather=[x for x in df.columns if x not in ['ZONEID','TIMESTAMP',target,'valid_time','forecast_day','source_hour']]
    assert np.isfinite(df[[target]+weather].values.astype(float)).all()
    diffs=df.valid_time.diff().dropna().dt.total_seconds().value_counts().to_dict()
    canonical=df[['valid_time','forecast_day','source_hour',target]+weather].rename(columns={target:'y'})
    canonical.to_csv(O/f'{name}_zone1_canonical.csv',index=False,float_format='%.17g')
    info={'zone':1,'rows':len(df),'columns':list(df.columns),'weather_columns':weather,
          'valid_min':str(df.valid_time.min()),'valid_max':str(df.valid_time.max()),
          'forecast_day_min':str(df.forecast_day.min()),'forecast_day_max':str(df.forecast_day.max()),
          'diff_seconds':{str(k):int(v) for k,v in diffs.items()},'missing_cells':int(df.isna().sum().sum()),
          'daily_rows':{str(k):int(v) for k,v in df.groupby('forecast_day').size().value_counts().to_dict().items()},
          'sha256':hashlib.sha256((O/f'{name}_zone1_canonical.csv').read_bytes()).hexdigest(),
          'target_semantics':'original normalized power; no true future weather measurements identified',
          'timezone':'unspecified original benchmark calendar; no UTC invented'}
    datasets[name]=info
out={'raw_members':records,'datasets':datasets,'no_model_scores':True,
     'hour24_conversion':'source date + source hour; 24:00 is next-day midnight',
     'future_predictors_excluded_without_target':'Solar weather rows without original train15 target are not used'}
(O/'data_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'datasets':datasets,'raw_members':[{k:v for k,v in x.items() if k!='member_chain'} for x in records]}))
