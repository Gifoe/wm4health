"""TRAIN-only numerical check for independently recomputed normalization."""
import pathlib,json,numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
for ds in ['NP','PJM']:
    meta=json.loads((ROOT/'outputs/native_cache_v1'/ds/'manifest.json').read_text())['data']
    frame=pd.read_csv(pathlib.Path('D:/baseline_reproduction_seed2021_v1/data')/(ds+'.csv'))
    names=frame['cols'].drop_duplicates().tolist()
    raw=frame.pivot(index='date',columns='cols',values='data')[['OT']+names[:-1]].to_numpy(dtype=np.float64)
    tr=int(int(len(raw)*.8)*.875)
    mu=raw[:tr].mean(0);sd=raw[:tr].std(0)
    expectedmu=np.array(meta['y_mean']+meta['cov_mean']);expectedsd=np.array(meta['y_scale']+meta['cov_scale'])
    print(json.dumps({'dataset':ds,'train_only':True,'mean_max_abs':float(np.abs(mu-expectedmu).max()),
        'scale_max_abs':float(np.abs(sd-expectedsd).max()),'mean':mu.tolist(),'scale':sd.tolist()}))
