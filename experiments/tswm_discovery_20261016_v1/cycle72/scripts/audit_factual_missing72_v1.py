"""Inspect only declared factual columns after the finite-value gate stopped fitting."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v3 import *
import pandas as pd
import numpy as np
source=PROJECT/'upstream'/'epi-cf-benchmark';sys.path.insert(0,str(source/'src'))
from epicf_benchmark import preprocessing as pre
acq=read(OUT/'DATA_ACQUISITION72_V1.json');rows=[]
for setting,name,target,policy,serm,inf in [
 ('single','single_policy_data.csv','0-F',['treatment'],pre.SINGLE_SERM_FACTUAL,'infected_sch=0-occ=F'),
 ('multi','multi_policy_data.csv','F-F',['school_treatment','occ_treatment'],pre.MULTI_SERM_FACTUAL,'infected_F-F')]:
    spec=next(r for r in acq['files'] if r['file']==name);p=storage('data')/spec['storage_relative'];assert sha(p)==spec['sha256']
    cols=['county','date','day','population_size',target,*policy,*pre.AGE_COLUMNS,*pre.MOBILITY_COLUMNS,*serm,inf]
    df=pd.read_csv(p,usecols=cols,dtype={'county':'string'})
    numeric=[x for x in cols if x not in ('county','date')]
    missing=[]
    for c in numeric:
        bad=~np.isfinite(df[c].to_numpy(dtype=np.float64))
        if bad.any():missing.append(dict(column=c,nonfinite_values=int(bad.sum()),affected_counties=int(df.loc[bad,'county'].nunique()),
            first_locations=df.loc[bad,['county','date','day']].head(5).to_dict(orient='records')))
    rows.append(dict(setting=setting,rows=len(df),counties=int(df['county'].nunique()),county_row_count_distribution={str(k):int(v) for k,v in df.groupby('county').size().value_counts().items()},missing=missing,
        target_nonfinite=int((~np.isfinite(df[target])).sum()),compartment_nonfinite=int((~np.isfinite(df[serm])).to_numpy().sum()),
        factual_columns_parsed=cols,counterfactual_columns_parsed=False))
obj=dict(utc=utc(),status='FACTUAL_MISSINGNESS_GATE_AUDITED_BEFORE_FITTING',settings=rows,forecasts_computed=0,
    test_targets_accessed=False,no_rows_or_counties_removed=True)
dump(OUT/'FACTUAL_MISSINGNESS72_V1.json',obj);print(json.dumps(obj),flush=True)
