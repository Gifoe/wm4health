"""Finalize numeric-download manifest and observed cohort description."""
import hashlib
import json
from collections import Counter
import numpy as np
import pandas as pd
from core import ROOT

def main():
    out=ROOT/'outputs'; raw=ROOT/'data/raw'
    records=json.loads((out/'case_preprocessing_log.json').read_text())
    included=[r for r in records if r['status']=='included']
    summary=json.loads((out/'data_summary.json').read_text())
    summary['actual_action_track_pairs']={', '.join(k):v for k,v in Counter(tuple(r['action_tracks']) for r in included).items()}
    summary['actual_map_sources']=dict(Counter(r['map_track'] for r in included))
    summary['exclusion_reasons']=dict(Counter(r['reason'] for r in records if r['status']=='excluded'))
    summary['mean_full_case_state_coverage']=np.mean([r['state_coverage'] for r in included],0).tolist()
    summary['mean_full_case_action_coverage']=np.mean([r['action_coverage'] for r in included],0).tolist()
    tr=pd.read_csv(raw/'trks.csv').set_index('tid')
    dictionary=pd.read_csv(raw/'track_names.csv').set_index('Parameter')
    manifest=[]
    for p in sorted(raw.glob('*.csv')):
        row={'filename':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        if len(p.stem)==40:
            rr=tr.loc[p.stem]; name=rr.tname
            assert dictionary.loc[name,'Type/Hz']=='N',f'Unexpected nonnumeric download: {name}'
            row.update({'kind':'numeric','caseid':int(rr.caseid),'track':name,'url':'https://api.vitaldb.net/'+p.stem})
        else: row['kind']='metadata'
        manifest.append(row)
    pd.DataFrame(manifest).to_csv(out/'download_manifest.csv',index=False)
    summary['numeric_track_files']=sum(r['kind']=='numeric' for r in manifest)
    summary['downloaded_csv_bytes']=sum(r['bytes'] for r in manifest)
    (out/'data_summary.json').write_text(json.dumps(summary,indent=2))
    ca=pd.read_csv(raw/'cases.csv').set_index('caseid'); splits=json.loads((out/'split_caseids.json').read_text())
    rows=[]
    for split,ids in splits.items():
        d=ca.loc[ids]
        for col in ['age','weight','height','caseend']:
            x=d[col]; rows.append({'split':split,'variable':col,'mean':x.mean(),'std':x.std(),'median':x.median(),'q25':x.quantile(.25),'q75':x.quantile(.75),'n':len(x)})
        rows.append({'split':split,'variable':'female_fraction','mean':(d.sex=='F').mean(),'n':len(d)})
    pd.DataFrame(rows).to_csv(out/'cohort_descriptive_statistics.csv',index=False)
    print(json.dumps({k:summary[k] for k in ['actual_action_track_pairs','actual_map_sources','exclusion_reasons','numeric_track_files','downloaded_csv_bytes']}),flush=True)

if __name__=='__main__': main()
