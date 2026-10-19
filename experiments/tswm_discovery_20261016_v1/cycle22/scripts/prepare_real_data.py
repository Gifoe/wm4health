"""Frozen whole-recording windows; known command paths, train-only normalization."""
import runtime_profile
import hashlib,json,pathlib
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/real_data'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not DATA.exists();DATA.mkdir(parents=True)
    audit=json.loads((ROOT/'outputs/interface_audit_v1/inventory.json').read_text());target=audit['target_columns'];conds=audit['condition_columns'];columns=target+conds
    files={p.name:p for p in (ROOT.parent/'cycle8/data/wt_walks_v1').rglob('*.csv')};raw={}
    for row in audit['records']:
        file=files[row['file']];assert sha(file)==row['sha256'];raw[file.name]=pd.read_csv(file)[columns].to_numpy(dtype='float64')
    split=audit['proposed_split'];training=np.concatenate([raw[n] for n in split['train']]);mean=training.mean(0);scale=np.maximum(training.std(0,ddof=0),1e-6)
    settings=next(v['settings'] for v in audit['records'] if v['file']=='actuators_random_walk_2.csv')
    waves=[v for v in audit['records'] if v['file'].startswith('loads_hatch_mix_')]
    split.pop('wave_test');split['wave_matched']=[v['file'] for v in waves if v['settings']==settings];split['wave_acquisition']=[v['file'] for v in waves if v['settings']!=settings]
    assert len(split['wave_matched'])==8 and len(split['wave_acquisition'])==2
    registry=[]
    for name,names in split.items():
        hist=[];future=[];ys=[];origins=[];ids=[]
        for rid,filename in enumerate(names):
            a=((raw[filename]-mean)/scale).astype('float32')
            for origin in range(64,len(a)-16+1,16):
                hist.append(a[origin-64:origin]);future.append(a[origin:origin+16,11:]);ys.append(a[origin:origin+16,:11]);origins.append(origin);ids.append(rid)
        file=DATA/(name+'.npz');np.savez_compressed(file,history=np.stack(hist),future_conditions=np.stack(future),target=np.stack(ys),
            origin=np.asarray(origins,dtype='int64'),recording_id=np.asarray(ids,dtype='int64'),recording_names=np.asarray(names),mean=mean,scale=scale)
        entry=dict(split=name,file=file.name,windows=len(hist),recordings=names,sha256=sha(file));registry.append(entry);print(json.dumps(entry),flush=True)
    manifest=dict(splits=registry,target_columns=target,condition_columns=conds,mean=mean.tolist(),scale=scale.tolist(),
        raw_csv_sha256={v['file']:v['sha256'] for v in audit['records']},script_sha256=sha(pathlib.Path(__file__)),plan_sha256=sha(ROOT/'EXPERIMENT_PLAN.md'),
        source_pin_sha256=sha(ROOT/'native_source_pin.json'),scope='New public real-data migration, not paper benchmark reproduction or causal-effect identification.')
    (DATA/'registry.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
