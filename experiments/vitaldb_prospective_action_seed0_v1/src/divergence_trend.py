"""Quantitative patient-cluster trend of prospective action value versus future schedule change."""
import numpy as np
import pandas as pd
from core import ROOT
from evaluate import bootstrap,abs_error,scalar_error

def main():
    out=ROOT/'outputs';ref=np.load(out/'test_reference.npz')
    target=ref['target'];mask=ref['mask'];subjects=ref['subject'];divergence=ref['divergence'];bins=ref['quartile']
    hold=np.load(out/'prediction_hold.npz')['prediction'];true=np.load(out/'prediction_true.npz')['prediction']
    fav=scalar_error(abs_error(hold,target,mask),0)-scalar_error(abs_error(true,target,mask),0)
    slopes=[]
    for person in np.unique(subjects):
        k=(subjects==person)&np.isfinite(fav)&np.isfinite(divergence)
        if k.sum()<20 or np.std(divergence[k])<1e-6:continue
        x=np.log1p(divergence[k]);y=fav[k]
        slopes.append(float(np.cov(x,y,bias=True)[0,1]/np.var(x)))
    slope=bootstrap(np.array(slopes),np.arange(len(slopes))) if slopes else (np.nan,np.nan,np.nan,0)
    df=pd.DataFrame({'subject':subjects,'bin':bins,'fav':fav}).dropna()
    wide=df.groupby(['subject','bin']).fav.mean().unstack()
    q1=wide[1] if 1 in wide else pd.Series(dtype=float)
    q4=wide[4] if 4 in wide else pd.Series(dtype=float)
    contrast=(q4-q1).dropna()
    qcontrast=bootstrap(contrast.to_numpy(),contrast.index.to_numpy())
    pd.DataFrame([{'test':'within_patient_log1p_divergence_slope','estimate':slope[0],
                   'ci_low':slope[1],'ci_high':slope[2],'n_patients':slope[3]},
                  {'test':'paired_patient_Q4_minus_Q1_FAV','estimate':qcontrast[0],
                   'ci_low':qcontrast[1],'ci_high':qcontrast[2],'n_patients':qcontrast[3]}]).to_csv(
                   out/'divergence_trend.csv',index=False)
    print('DIVERGENCE_TREND_COMPLETE',flush=True)

if __name__=='__main__':main()
