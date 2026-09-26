"""Numeric-only VitalDB API acquisition and strictly causal resampling."""
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yaml
import vitaldb

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'config.yaml').read_text())
RAW = ROOT / 'data/raw'
CASES = ROOT / 'data/cases'
OUT = ROOT / 'outputs'
for p in (RAW, CASES, OUT): p.mkdir(parents=True, exist_ok=True)
TRACKS = ['BIS/BIS', 'Solar8000/HR', 'Solar8000/ART_MBP',
          'Solar8000/FEM_MBP', 'Solar8000/NIBP_MBP',
          'Orchestra/PPF20_RATE', 'Orchestra/RFTN20_RATE',
          'Orchestra/PPF20_VOL', 'Orchestra/RFTN20_VOL',
          'Orchestra/PPF20_CE', 'Orchestra/RFTN20_CE']
PREPROCESS_VERSION = 2

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))

def fetch(url, path, timeout=90):
    if path.exists() and path.stat().st_size > 10: return path
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=(15,timeout))
            r.raise_for_status()
            tmp = path.with_suffix(path.suffix + '.part')
            tmp.write_bytes(r.content)
            tmp.replace(path)
            return path
        except Exception:
            if attempt == 3: raise
            time.sleep(1 + attempt)

def load_numeric(tid):
    path = fetch('https://api.vitaldb.net/' + tid, RAW / (tid + '.csv'))
    tab = pd.read_csv(path, na_values=['-nan(ind)'])
    a = tab.iloc[:, :2].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    # Waveform CSVs have NaN timestamps; whitelist plus this guard forbids them.
    if not np.isfinite(a[:,0]).all(): raise ValueError('non-numeric timestamp schema')
    a = a[np.argsort(a[:,0],kind='stable')]
    return a

def resample(a, grid, lower, upper, gap=60):
    """Latest valid observation at or before t, never backfill/interpolate."""
    a = a[np.isfinite(a[:,1]) & (a[:,1] >= lower) & (a[:,1] <= upper)]
    if len(a) == 0: return np.full(len(grid), np.nan), np.zeros(len(grid),bool)
    ii = np.searchsorted(a[:,0], grid, side='right') - 1
    age = grid - a[np.maximum(ii,0),0]
    ok = (ii >= 0) & (age <= gap)
    vals = np.where(ok,a[np.maximum(ii,0),1],np.nan)
    fresh = ok & (age < 10)
    return vals, fresh

def valid_anchors(state, action, raw_state):
    n = len(state); h=CFG['history_steps']; f=CFG['horizon_steps']
    def count(x, left, right):
        cs = np.r_[0,np.cumsum(x)]
        return cs[right] - cs[left]
    t = np.arange(h-1,n-f,dtype=np.int32)
    hist = count(np.isfinite(state[:,0]),t-h+1,t+1)/h
    # Evaluate observed-bin targets, not values carried through missing future bins.
    fut = count(raw_state[:,0],t+1,t+f+1)/f
    ok = (hist >= CFG['minimum_history_bis_coverage']) & (fut >= CFG['minimum_target_bis_coverage'])
    ok &= np.isfinite(state[t,0])
    for j in range(2): ok &= count(np.isfinite(action[:,j]),t-h+1,t+1)/h >= CFG['minimum_action_coverage']
    for k in (3,6,18,30): ok &= raw_state[t+k,0]
    return t[ok]

def process_case(rec, index, rate_allowed):
    cid = int(rec['caseid']); target = CASES / f'{cid}.npz'; meta = CASES / f'{cid}.json'
    if target.exists() and meta.exists():
        previous=json.loads(meta.read_text())
        if previous.get('preprocess_version')==PREPROCESS_VERSION: return previous
    available = index[cid]
    loaded = {}
    def get(name):
        if name not in loaded: loaded[name] = load_numeric(available[name])
        return loaded[name]
    try:
        bis = get('BIS/BIS')
        grid = np.arange(0, np.floor(bis[-1,0]/10)*10+1, 10,dtype=float)
        if len(grid) < 210: raise ValueError('recording_shorter_than_35_minutes')
        s = np.full((len(grid),3),np.nan)
        fresh = np.zeros_like(s,dtype=bool)
        s[:,0],fresh[:,0] = resample(bis,grid,1,100)
        hr_name = 'Solar8000/HR'
        if hr_name in available: s[:,1],fresh[:,1] = resample(get(hr_name),grid,20,250)
        map_track = None
        for name in ['Solar8000/ART_MBP','Solar8000/FEM_MBP','Solar8000/NIBP_MBP']:
            if name in available:
                vals, obs = resample(get(name),grid,20,200)
                if map_track is None or np.isfinite(vals).sum() > np.isfinite(s[:,2]).sum():
                    s[:,2], fresh[:,2], map_track = vals,obs,name
                if name!='Solar8000/NIBP_MBP' and np.isfinite(vals[np.isfinite(s[:,0])]).mean()>=.5:
                    s[:,2], fresh[:,2], map_track = vals,obs,name
                    break
        # Repeated NIBP publications cannot be assumed to be new cuff measurements.
        if map_track=='Solar8000/NIBP_MBP': fresh[:,2]=False
        action = np.full((len(grid),2),np.nan)
        ce = np.full_like(action,np.nan)
        choices = []
        quality = []
        support = np.isfinite(s[:,0])
        for j,drug in enumerate(['PPF20','RFTN20']):
            rn,vn,cn = [f'Orchestra/{drug}_{suffix}' for suffix in ['RATE','VOL','CE']]
            mode='volume_difference'; cov=0.0
            if rate_allowed[j] and rn in available:
                rate, _ = resample(get(rn),grid,0,3600)
                cov = float(np.isfinite(rate[support]).mean()) if support.any() else 0
                if cov >= .8:
                    action[:,j] = rate/360.0 # mL/h -> mL per 10-second interval
                    mode='direct_rate'
            if mode == 'volume_difference':
                if vn not in available: raise ValueError(f'{drug}_missing_valid_administration')
                vol, _ = resample(get(vn),grid,0,10000)
                delta = np.r_[np.nan,np.diff(vol)]
                # Official example sets resets and >10mL/10s jumps to zero.
                # We mark them missing so artifact zeros cannot become true stops.
                bad = (delta < 0) | (delta > 10)
                delta[bad] = np.nan
                action[:,j] = delta
            ce[:,j], _ = resample(get(cn),grid,0,100)
            if np.nansum(action[:,j]) <= 1: raise ValueError(f'{drug}_no_substantial_infusion')
            if not np.any(ce[:,j] > 0): raise ValueError(f'{drug}_no_positive_ce')
            choices.append(rn if mode=='direct_rate' else vn)
            quality.append({'mode':mode,'rate_coverage_on_bis':cov})
        anchors = valid_anchors(s,action,fresh)
        if len(anchors) < 30: raise ValueError('fewer_than_30_valid_windows')
        demo = np.array([rec['age'],float(rec['sex']=='F'),rec['weight'],rec['height']],np.float32)
        if not np.isfinite(demo).all(): raise ValueError('missing_demographics')
        np.savez_compressed(target,state=s.astype(np.float32),state_fresh=fresh,
                            action=action.astype(np.float32),ce=ce.astype(np.float32),
                            demographics=demo,time=grid.astype(np.float32),anchors=anchors,
                            caseid=cid,subjectid=int(rec['subjectid']))
        result={'caseid':cid,'subjectid':int(rec['subjectid']),'status':'included','steps':len(grid),
                'preprocess_version':PREPROCESS_VERSION,
                'windows':len(anchors),'action_tracks':choices,'rate_quality':quality,
                'hr_track':hr_name if hr_name in available else None,'map_track':map_track,
                'state_coverage':np.isfinite(s).mean(0).tolist(),
                'action_coverage':np.isfinite(action).mean(0).tolist(),
                'ce_coverage':np.isfinite(ce).mean(0).tolist(),
                'track_ids':{k:available[k] for k in loaded},
                'first_last_bis':[float(s[np.isfinite(s[:,0]),0][0]),float(s[np.isfinite(s[:,0]),0][-1])]}
    except Exception as exc:
        result={'caseid':cid,'subjectid':int(rec['subjectid']),'status':'excluded','reason':str(exc),'preprocess_version':PREPROCESS_VERSION}
    dump(meta,result)
    return result

def main():
    fetch('https://api.vitaldb.net/cases', RAW/'cases.csv')
    fetch('https://api.vitaldb.net/trks', RAW/'trks.csv',180)
    fetch('https://physionet.org/files/vitaldb/1.0.0/track_names.csv',RAW/'track_names.csv')
    fetch('https://raw.githubusercontent.com/vitaldb/examples/master/ppf_bis.ipynb',OUT/'official_ppf_bis.ipynb')
    cases=pd.read_csv(RAW/'cases.csv'); trks=pd.read_csv(RAW/'trks.csv')
    dictionary=pd.read_csv(RAW/'track_names.csv').set_index('Parameter')
    assert all(dictionary.loc[x,'Type/Hz']=='N' for x in TRACKS)
    vitaldb.dataset.dfci=cases; vitaldb.dataset.dftrks=trks
    required=['BIS/BIS','Orchestra/PPF20_CE','Orchestra/RFTN20_CE']
    track_eligible=set(vitaldb.find_cases(required))
    stage={'all_cases':len(cases)}
    use=cases[(cases.age>18)&(cases.weight>35)&(cases.caseend>7200)&(cases.ane_type=='General')].copy()
    stage['adult_general_weight_duration']=len(use)
    use=use[use.caseid.isin(track_eligible)]
    stage['bis_both_ce']=len(use)
    for drug in ['PPF20','RFTN20']:
        ids=set(trks.loc[trks.tname.isin([f'Orchestra/{drug}_RATE',f'Orchestra/{drug}_VOL']),'caseid'])
        use=use[use.caseid.isin(ids)]
    stage['both_drug_administration_available']=len(use)
    use=use.dropna(subset=['age','sex','weight','height','subjectid'])
    use=use[(use.height>=100)&(use.height<=230)&(use.weight<=300)]
    stage['complete_plausible_demographics']=len(use)
    availability={name:int(trks[(trks.tname==name)&trks.caseid.isin(use.caseid)].caseid.nunique()) for name in TRACKS}
    rate_allowed=[availability[f'Orchestra/{x}_RATE']/len(use)>=.9 for x in ['PPF20','RFTN20']]
    rng=np.random.default_rng(0)
    use=use.sort_values('caseid').iloc[rng.permutation(len(use))[:CFG['max_cases']]]
    use.to_csv(OUT/'candidate_cases.csv',index=False)
    dump(OUT/'cohort_preregistered.json',{'stages':stage,'candidate_caseids':use.caseid.tolist(),
                                       'track_availability':availability,'rate_allowed':rate_allowed})
    index={int(cid): dict(zip(g.tname,g.tid)) for cid,g in trks[trks.caseid.isin(use.caseid)&trks.tname.isin(TRACKS)].groupby('caseid')}
    results=[]
    with cf.ThreadPoolExecutor(max_workers=CFG['download_workers']) as pool:
        futs={pool.submit(process_case,rec,index,rate_allowed):rec['caseid'] for rec in use.to_dict('records')}
        for future in cf.as_completed(futs):
            result=future.result(); results.append(result)
            print(f"DATA {len(results)}/{len(use)} case={result['caseid']} {result['status']} {result.get('reason','')} windows={result.get('windows',0)}",flush=True)
    results.sort(key=lambda x:x['caseid']); included=[r for r in results if r['status']=='included']
    dump(OUT/'case_preprocessing_log.json',results)
    if len(included)<50: raise RuntimeError(f'Insufficient cohort: {len(included)}; inspect exclusions')
    subjects=np.array(sorted(set(r['subjectid'] for r in included)))
    subjects=np.random.default_rng(0).permutation(subjects)
    n=len(subjects); ntr=int(.7*n); nv=int(.15*n)
    split_subjects={'train':subjects[:ntr].tolist(),'val':subjects[ntr:ntr+nv].tolist(),'test':subjects[ntr+nv:].tolist()}
    splits={key:[r['caseid'] for r in included if r['subjectid'] in ids] for key,ids in split_subjects.items()}
    dump(OUT/'split_caseids.json',splits); dump(OUT/'split_subjectids.json',split_subjects)
    counts={k:{'cases':len(v),'patients':len(split_subjects[k]),'windows':sum(r['windows'] for r in included if r['caseid'] in v)} for k,v in splits.items()}
    summary={'cohort_stages':stage,'selected_before_qc':len(use),'included_cases':len(included),
             'included_patients':len(subjects),'excluded_cases':len(use)-len(included),'splits':counts,
             'track_availability':availability,'rate_allowed':rate_allowed,
             'sampling_seconds':10,'history_steps':180,'horizon_steps':30,'waveform_downloads':0,
             'endpoint_bis_filters':'not applied: these select clinical outcomes rather than data quality',
             'resampling':'latest past numeric observation; max age 60s; leading/long gaps remain NaN; explicit masks',
             'drug_units':'mL per 10 seconds; direct RATE / 360 preferred, otherwise VOL differences',
             'ce_role':'device-computed TCI reference only; excluded from model feature construction and training',
             'api':'official numeric per-track CSV API; no full-case .vital/parquet requests',
             'map_policy':'prefer arterial MAP with >=50% coverage; cuff MAP may enter history but never the MAP target',
             'timeline':'common CSV case-relative time; metadata times used only for duration eligibility',
             'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [RAW/'cases.csv',RAW/'trks.csv',RAW/'track_names.csv']}}
    dump(OUT/'data_summary.json',summary)
    print('DATA_READY '+json.dumps(counts),flush=True)

if __name__=='__main__': main()
