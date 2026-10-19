import pathlib,json,pandas as pd,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
for name,path,target in [('solar', 'solar_with_weather/15T/train-00000-of-00001.parquet','target'),
                          ('jena','jena_weather/10T/train-00000-of-00001.parquet','target_0')]:
    f=ROOT/'assets/fev'/path;df=pd.read_parquet(f)
    cols=list(df.columns);a=np.asarray(df.iloc[0][target])
    record={'name':name,'rows':len(df),'columns':cols,'target':target,'first_target_shape':a.shape,
        'first_target_dtype':str(a.dtype),'finite_count':int(np.isfinite(a).sum()),
        'row0_item_id':str(df.iloc[0].get('item_id','unknown'))}
    if 'timestamp' in cols:
        t=np.asarray(df.iloc[0]['timestamp']);record.update(timestamp_shape=t.shape,first_timestamp=str(t[0]),last_timestamp=str(t[-1]))
    print(json.dumps(record))
