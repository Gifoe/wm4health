"""Outcome-free interpretation of TRAIN-derived intervention regimes."""
import json
import numpy as np
import pandas as pd
from common import *

def main():
    store=Store();a=np.asarray(arr('train','action'))
    physical=np.expm1(a*store.norm['action_std'][None,None,:]+store.norm['action_mean'][None,None,:])/10
    start=np.median(physical[:,:6],axis=1);end=np.median(physical[:,-6:],axis=1)
    positive=physical[physical>0]
    on=float(np.quantile(positive,.1))
    delta=end-start;change=float(np.quantile(np.abs(delta[np.abs(delta)>0]),.75))
    labels=np.zeros((len(a),2),np.int8)
    labels[(start<=on)&(end>on)]=1
    labels[(start>on)&(end<=on)]=4
    labels[(start>on)&(end>on)&(delta>=change)]=2
    labels[(start>on)&(end>on)&(delta<=-change)]=3
    clusters=np.load(ARRAYS/'train_action_cluster.npy')
    frame=pd.read_csv(OUT/'action_cluster_summary.csv')
    for c in frame.action_cluster:
        m=clusters==c;dose=physical[m]
        for j,drug in enumerate(('ppf','rft')):
            for code,name in ((1,'initiation'),(2,'increase'),(3,'decrease'),(4,'stop')):
                frame.loc[frame.action_cluster==c,f'{drug}_{name}_fraction']=float(np.mean(labels[m,j]==code))
            frame.loc[frame.action_cluster==c,f'{drug}_q10_total']=float(np.quantile(dose[:,:,j].sum(1),.1))
    frame.to_csv(OUT/'action_cluster_summary.csv',index=False)
    states=pd.read_csv(OUT/'state_cluster_summary.csv');raw=np.load(ARRAYS/'train_state_raw.npy')
    state_cluster=np.load(ARRAYS/'train_state_cluster.npy')
    for s in states.state_cluster:
        m=state_cluster==s
        states.loc[states.state_cluster==s,'female_fraction']=float(np.mean(raw[m,37]))
        states.loc[states.state_cluster==s,'median_weight_kg']=float(np.median(raw[m,38]))
        states.loc[states.state_cluster==s,'median_height_cm']=float(np.median(raw[m,39]))
        states.loc[states.state_cluster==s,'median_BIS_change_5min']=float(np.nanmedian(raw[m,33]))
        states.loc[states.state_cluster==s,'median_HR_change_5min']=float(np.nanmedian(raw[m,34]))
        states.loc[states.state_cluster==s,'median_MAP_change_5min']=float(np.nanmedian(raw[m,35]))
    states.to_csv(OUT/'state_cluster_summary.csv',index=False)
    selected=[tuple(x) for x in json.loads((OUT/'selection_protocol.json').read_text())['selected_cells']]
    cutoff=np.array(json.loads((SOURCE/'outputs/pump_rate_tail_protocol.json').read_text())['0.99']['maximum_ml_per_10s'])
    tail_rows=[]
    for split in ('train','val','test'):
        s=np.load(ARRAYS/f'{split}_state_cluster.npy');ac=np.load(ARRAYS/f'{split}_action_cluster.npy')
        norm_a=np.asarray(arr(split,'action'))
        dose=np.expm1(norm_a*store.norm['action_std'][None,None,:]+store.norm['action_mean'][None,None,:])/10
        tail=(dose>cutoff).any((1,2))
        for sn,an in selected:
            m=(s==sn)&(ac==an)
            tail_rows.append(dict(split=split,state_cluster=sn,action_cluster=an,
                 windows=int(m.sum()),high_rate_tail_fraction=float(tail[m].mean())))
    pd.DataFrame(tail_rows).to_csv(OUT/'cell_tail_prevalence.csv',index=False)
    jwrite('action_cluster_description_protocol.json',{'positive_dose_p10_on_ml_per_10s':on,
      'abs_first_last_minute_change_p75_ml_per_10s':change,
      'labels':'initiation/increase/decrease/stop from first and last 1-min median future rate for each drug; TRAIN-only thresholds',
      'no_future_physiology_or_CE':True})
    print('REGIME_DESCRIPTIONS_COMPLETE',flush=True)

if __name__=='__main__':main()
