"""Reconstruct every window from raw CSVs and independently verify availability layout."""
import runtime_profile
import hashlib,json,pathlib,time
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/real_data'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    tick=time.perf_counter();dest=DATA/'independent_verification.json';assert not dest.exists();reg=json.loads((DATA/'registry.json').read_text())
    paths={p.name:p for p in (ROOT.parent/'cycle8/data/wt_walks_v1').rglob('*.csv')};raw={}
    cols=reg['target_columns']+reg['condition_columns'];assert len(cols)==16 and len(set(cols))==16
    for name,digest in reg['raw_csv_sha256'].items():
        file=paths[name];assert sha(file)==digest;raw[name]=pd.read_csv(file)[cols].to_numpy(dtype='float64')
    train_names=next(v['recordings'] for v in reg['splits'] if v['split']=='train');assert train_names==['actuators_random_walk_'+str(i)+'.csv' for i in range(1,11)]
    train=np.concatenate([raw[n] for n in train_names]);mean=train.mean(0);scale=np.maximum(train.std(0),1e-6)
    assert np.array_equal(mean,np.asarray(reg['mean'])) and np.array_equal(scale,np.asarray(reg['scale']))
    sets=[set(v['recordings']) for v in reg['splits']]
    for i in range(len(sets)):
        for j in range(i):assert not sets[i]&sets[j]
    count=0;worst=0.;checks=[]
    for entry in reg['splits']:
        path=DATA/entry['file'];assert sha(path)==entry['sha256']
        with np.load(path,allow_pickle=False) as archive:s={key:archive[key] for key in archive.files}
        assert list(s['recording_names'])==entry['recordings'];assert np.array_equal(s['mean'],mean) and np.array_equal(s['scale'],scale)
        assert s['history'].shape==(entry['windows'],64,16) and s['future_conditions'].shape==(entry['windows'],16,5) and s['target'].shape==(entry['windows'],16,11)
        for rid,name in enumerate(entry['recordings']):
            values=((raw[name]-mean)/scale).astype('float32');use=np.flatnonzero(s['recording_id']==rid);expected=np.arange(64,len(values)-16+1,16)
            assert np.array_equal(s['origin'][use],expected)
            for index,origin in zip(use,expected):
                # Exact source positions; past cannot extend into the outcome window.
                assert origin-64>=0 and origin+16<=len(values)
                for saved,truth in [(s['history'][index],values[origin-64:origin]),(s['future_conditions'][index],values[origin:origin+16,11:]),(s['target'][index],values[origin:origin+16,:11])]:
                    error=float(np.max(np.abs(saved-truth)));worst=max(worst,error);assert np.isfinite(saved).all()
                count+=1
        checks.append(dict(split=entry['split'],windows=entry['windows'],sha256=sha(path)))
    assert worst==0 and count==sum(v['windows'] for v in reg['splits'])
    report=dict(pass_=True,raw_csv_hashes=len(raw),window_files_checked=len(checks),every_window_reconstructed=count,raw_reconstruction_max_error=worst,
        split_recording_sets_disjoint=True,normalization_train_only=True,future_input_columns=reg['condition_columns'],future_sensor_or_clock_inputs=False,
        checks=checks,seconds=time.perf_counter()-tick,script_sha256=sha(pathlib.Path(__file__)),
        scope='All window values/layout/hashes checked against raw recordings and frozen roles. Command availability is supported by the archived protocol,not inferred from data correlations.')
    report['pass']=report.pop('pass_');dest.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
