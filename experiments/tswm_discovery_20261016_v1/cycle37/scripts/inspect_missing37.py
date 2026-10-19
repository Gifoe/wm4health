"""Missingness metadata only, no predictive outcomes or imputation."""
import pathlib,pandas as pd,numpy as np,json
R=pathlib.Path(__file__).resolve().parents[1];out={}
for f in (R/'data').glob('*original.csv'):
    d=pd.read_csv(f);d=d[d.ZONEID==1]
    num=d.select_dtypes(include='number')
    bad=~np.isfinite(num.to_numpy(dtype=float));mask=bad.any(axis=1)
    out[f.name]={'zone1_rows':len(d),'nonfinite_columns':dict(zip(num.columns,bad.sum(axis=0).astype(int).tolist())),
        'bad_row_count':int(mask.sum()),'first_bad_timestamps':d.loc[mask,'TIMESTAMP'].head(8).tolist(),
        'last_bad_timestamps':d.loc[mask,'TIMESTAMP'].tail(8).tolist()}
(R/'data_intake/missing_diagnosis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
