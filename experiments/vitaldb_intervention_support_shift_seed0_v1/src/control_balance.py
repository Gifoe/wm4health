"""Auditable Z-versus-R removal balance, including temporal concentration."""
import numpy as np,pandas as pd
from common import *

def main():
    f=pd.read_csv(OUT/'support_removal_manifest.csv')
    z=f[~f.keep_Z];r=f[~f.keep_R]
    blocks=f.groupby('block_id').size()
    rows=[]
    for name,x in [('Z',z),('R',r)]:
        removed=x.groupby('block_id').size()
        frac=(removed/blocks.loc[removed.index]).to_numpy()
        per_patient=x.groupby('subjectid').size().to_numpy()
        rows.append(dict(condition=name,removed_windows=len(x),touched_blocks=len(removed),
          removed_cases=x.caseid.nunique(),removed_patients=x.subjectid.nunique(),
          fraction_removed_per_touched_block_q10=float(np.quantile(frac,.1)),
          fraction_removed_per_touched_block_median=float(np.median(frac)),
          fraction_removed_per_touched_block_q90=float(np.quantile(frac,.9)),
          removed_windows_per_patient_q10=float(np.quantile(per_patient,.1)),
          removed_windows_per_patient_median=float(np.median(per_patient)),
          removed_windows_per_patient_q90=float(np.quantile(per_patient,.9))))
    pd.DataFrame(rows).to_csv(OUT/'random_control_balance.csv',index=False)
    print('CONTROL_BALANCE_COMPLETE',rows,flush=True)

if __name__=='__main__':main()
