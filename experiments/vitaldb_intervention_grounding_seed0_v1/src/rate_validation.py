"""Small prespecified RATE-vs-VOL semantics audit, no physiology/model outcomes."""
import concurrent.futures as cf
import json
import numpy as np
import pandas as pd
from prepare import RAW,OUT,load_numeric

def one(cid,drug,index):
    row={'caseid':int(cid),'drug':drug}
    try:
        rate=load_numeric(index[(cid,f'Orchestra/{drug}_RATE')])
        volume=load_numeric(index[(cid,f'Orchestra/{drug}_VOL')])
        start=max(rate[0,0],volume[0,0]); end=min(rate[-1,0],volume[-1,0])
        rr=rate[(rate[:,0]>=start)&(rate[:,0]<=end)]
        vv=volume[(volume[:,0]>=start)&(volume[:,0]<=end)]
        dt=np.diff(rr[:,0]); rv=rr[:-1,1]
        good=(dt>0)&(dt<=60)&(rv>=0)&(rv<=3600)&np.isfinite(rv)
        integrated=float(np.sum(dt[good]*rv[good]/3600))
        dv=np.diff(vv[:,1]); dvdt=np.diff(vv[:,0])
        valid=np.isfinite(dv)&(dv>=0)&(dv<=10)&(dvdt<=60)&(dvdt>0)
        total=float(dv[valid].sum())
        net_valid=np.isfinite(dv)&(dv>=-1)&(dv<=10)&(dvdt<=60)&(dvdt>0)
        net=float(dv[net_valid].sum())
        row.update({'rate_integral_ml':integrated,'positive_volume_increments_ml':total,
                    'rate_to_volume_ratio':integrated/total if total>0 else None,
                    'net_volume_excluding_large_resets_ml':net,
                    'rate_to_net_volume_ratio':integrated/net if net>0 else None,
                    'duration_seconds':float(end-start),'rate_interval_valid_fraction':float(good.mean()),
                    'negative_volume_difference_count':int((dv<0).sum()),
                    'large_volume_reset_count':int((dv < -1).sum())})
    except Exception as exc: row['error']=str(exc)
    return row

def main():
    tr=pd.read_csv(RAW/'trks.csv'); index=dict(zip(zip(tr.caseid,tr.tname),tr.tid))
    candidates=json.loads((OUT/'cohort_preregistered.json').read_text())['candidate_caseids'][:5]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(lambda item:one(*item,index),[(cid,d) for cid in candidates for d in ['PPF20','RFTN20']]))
    pd.DataFrame(result).to_csv(OUT/'rate_volume_semantics_check.csv',index=False)
    print(pd.DataFrame(result).to_string(index=False),flush=True)

if __name__=='__main__': main()
