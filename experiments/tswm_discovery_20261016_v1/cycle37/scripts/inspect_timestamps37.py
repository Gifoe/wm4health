"""Diagnose a timestamp-reader rejection without computing outcome statistics."""
import pathlib,pandas as pd,json
R=pathlib.Path(__file__).resolve().parents[1];out={}
for f in (R/'data').glob('*original.csv'):
    d=pd.read_csv(f);s=d.TIMESTAMP.str.strip().str.split(' ',n=1,expand=True)
    h=s[1].str.split(':').str[0].astype(int);m=s[1].str.split(':').str[1].astype(int)
    out[f.name]={'rows':len(d),'columns':list(d),'hour_counts':{str(k):int(v) for k,v in h.value_counts().sort_index().items()},
                 'minute_counts':{str(k):int(v) for k,v in m.value_counts().sort_index().items()},
                 'first_timestamp':d.TIMESTAMP.iloc[0],'last_timestamp':d.TIMESTAMP.iloc[-1]}
(R/'data_intake/timestamp_diagnosis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
