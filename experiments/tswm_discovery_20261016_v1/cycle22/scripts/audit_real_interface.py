"""New-task role/clock/split qualification of already verified public recordings."""
import runtime_profile
import hashlib,json,pathlib,platform,time
import numpy as np
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];OLD=ROOT.parent/'cycle8';OUT=ROOT/'outputs/interface_audit_v1'
TARGETS=['current_in','current_out','rpm_in','rpm_out','pressure_upwind','pressure_downwind','pressure_ambient','pressure_intake','mic','signal_1','signal_2']
CONDITIONS=['load_in','load_out','hatch','pot_1','pot_2']
SETTINGS=['v_1','v_2','v_mic','v_in','v_out','osr_1','osr_2','osr_mic','osr_in','osr_out','osr_upwind','osr_downwind','osr_ambient','osr_intake','res_in','res_out']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);prior=json.loads((OLD/'outputs/data_audit/inventory.json').read_text())
    hashes={pathlib.PurePosixPath(v['file']).name:v['sha256'] for v in prior['rows']};rows=[]
    root=OLD/'data/wt_walks_v1'
    for file in sorted(root.rglob('*.csv')):
        if not (file.stem.startswith('actuators_random_walk_') or file.stem.startswith('loads_hatch_mix_')):continue
        assert sha(file)==hashes[file.name];df=pd.read_csv(file);assert set(TARGETS+CONDITIONS+['timestamp']).issubset(df)
        ts=df['timestamp'].to_numpy(dtype='float64');dt=np.diff(ts);numeric=df[TARGETS+CONDITIONS].to_numpy(dtype='float64')
        row=dict(file=file.name,rows=len(df),sha256=sha(file),nonfinite_model_values=int((~np.isfinite(numeric)).sum()),
            nonpositive_clock_steps=int(np.sum(dt<=0)),timestamp_precision='float64; no future elapsed time supplied',
            dt_quantiles=np.quantile(dt,[0,.1,.5,.9,1]).tolist(),settings={k:df[k].unique().tolist() for k in SETTINGS},
            target_std={k:float(df[k].std()) for k in TARGETS},condition_ranges={k:[float(df[k].min()),float(df[k].max())] for k in CONDITIONS},
            condition_changed_fraction={k:float(np.mean(np.diff(df[k])!=0)) for k in CONDITIONS})
        rows.append(row)
    assert len(rows)==26
    source=OLD/'sources/chamber/datasets/wt_walks_v1/generators/actuators_random_walk.py'
    protocol=source.read_text(encoding='utf-8');assert 'MSR,1,0' in protocol and 'for actuator, walk in walks.items()' in protocol
    inventory=dict(records=rows,target_columns=TARGETS,condition_columns=CONDITIONS,
        source_repo='https://github.com/juangamella/causal-chamber',source_revision='0c8a77ea14bd4437b1fdd79adc1aa91ac2342569',
        generator_sha256=sha(source),data_version='wt_walks_v1; cached publisher-MD5/SHA-verified archive from cycle8',
        availability='All5 command paths are preset in experiment protocols before measurements; future realizations of11 sensor values and future timing are unavailable inputs.',
        proposed_split=dict(train=[f'actuators_random_walk_{i}.csv' for i in range(1,11)],val=[f'actuators_random_walk_{i}.csv' for i in range(11,13)],
            test=[f'actuators_random_walk_{i}.csv' for i in range(13,17)],wave_test=[v['file'] for v in rows if v['file'].startswith('loads_hatch_mix_')]),
        scope='Metadata/availability qualification, not causal-response truth or a new gap. Do not use earlier measurement artifacts as a model failure.',
        script_sha256=sha(pathlib.Path(__file__)))
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print(json.dumps(dict(recordings=len(rows),rows=sum(r['rows'] for r in rows),nonfinite=sum(r['nonfinite_model_values'] for r in rows),
        nonpositive_steps=sum(r['nonpositive_clock_steps'] for r in rows),files=[dict(file=r['file'],rows=r['rows'],dt_median=r['dt_quantiles'][2]) for r in rows])),flush=True)
if __name__=='__main__':main()
