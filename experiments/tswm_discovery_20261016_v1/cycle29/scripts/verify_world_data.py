"""Independent SciPy physics replay; no model scores or test-driven selection."""
import runtime_profile
import hashlib,json,pathlib,time
import numpy as np
from scipy.signal import dlsim
ROOT=pathlib.Path(__file__).resolve().parents[1];DATA=ROOT/'outputs/world_data';OUT=ROOT/'outputs/world_verification'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def linear(p,prefix,u):return dlsim(tuple(p[prefix+k] for k in 'ABCD')+(1.,),u)[1]
def nonlin(p,v):return np.tanh(v@p['w1'].T+p['b1'])@p['w2'].T+p['b2']
def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();meta=json.loads((DATA/'systems.json').read_text());manifest=json.loads((DATA/'manifest.json').read_text())
    assert manifest['plan_sha256']==sha(ROOT/'MODEL_COMPARISON_PLAN.md')
    for row in manifest['files']:assert sha(DATA/row['path'])==row['sha256'],row['path']
    sets={split:set() for split in ['train','val','test']}
    for r in meta:sets[r['split']].add(hashlib.sha256(json.dumps(r['parameters'],sort_keys=True).encode()).hexdigest())
    assert len(sets['train'])==1024 and len(sets['val'])==128 and len(sets['test'])==256
    assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    replay=[];prefix_max=0.;observed=0
    for family in ['lti','wh']:
        with np.load(DATA/f'test_{family}.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        for key in ['u','y','clean']:
            assert np.isfinite(arrays[key]).all();prefix_max=max(prefix_max,float(np.max(np.abs(arrays[key][:,:,:128]-arrays[key][:,0:1,:128]))))
        observed+=len(arrays['y'])
        for i in [0,1,2,3,31,63,95,127]:
            row=next(r for r in meta if r['split']=='test' and r['family']==family and r['index']==i);p={k:np.asarray(v,dtype='float64') for k,v in row['parameters'].items()};ref=arrays['reference_u'][i,:,None]
            if family=='wh':
                v=linear(p,'g1_',ref);im=v[200:].mean(0);isc=v[200:].std(0)+1e-6;refy=linear(p,'g2_',nonlin(p,(v-im)/isc))
            else:refy=linear(p,'',ref)
            um=ref[200:].mean(0);usc=ref[200:].std(0)+1e-6;ym=refy[200:].mean(0);ysc=refy[200:].std(0)+1e-6
            expected_reference=dict(u_mean=um,u_scale=usc,y_mean=ym,y_scale=ysc)
            if family=='wh':expected_reference.update(internal_mean=im,internal_scale=isc)
            calibration_error=max(float(np.max(np.abs(np.asarray(row['reference'][k])-v))) for k,v in expected_reference.items())
            assert calibration_error<1e-8,(family,i,calibration_error)
            for plan in range(4):
                u=arrays['raw_u'][i,:,None].copy()
                if plan==1:u[328:]=0.
                elif plan==2:u[328:]=1.5
                elif plan==3:
                    for t in range(128):u[328+t]=1.5 if (t//16)%2==0 else -1.5
                if family=='wh':out=linear(p,'g2_',nonlin(p,(linear(p,'g1_',u)-im)/isc))
                else:out=linear(p,'',u)
                yy=((out[200:]-ym)/ysc)[:,0];uu=((u[200:]-um)/usc)[:,0];stored=arrays['clean'][i,plan].astype('float64')
                normalized_error=float(np.max(np.abs(yy-stored)/(1+np.abs(yy))));u_error=float(np.max(np.abs(uu-arrays['u'][i,plan])/(1+np.abs(uu))))
                noise_error=float(np.max(np.abs((yy+arrays['noise'][i])-arrays['y'][i,plan])/(1+np.abs(yy+arrays['noise'][i]))))
                assert max(normalized_error,u_error,noise_error)<2e-7,(family,i,plan,normalized_error,u_error,noise_error)
                replay.append(dict(family=family,system=i,plan=plan,clean_relative_max=normalized_error,u_relative_max=u_error,noisy_relative_max=noise_error,reference_max_abs=calibration_error))
    assert prefix_max==0
    result=dict(status='passed',data_manifest_sha256=sha(DATA/'manifest.json'),script_sha256=sha(pathlib.Path(__file__)),plan_sha256=sha(ROOT/'MODEL_COMPARISON_PLAN.md'),independent_simulator='scipy.signal.dlsim + NumPy tanh; no world_data functions used',unique_worlds_by_split={k:len(v) for k,v in sets.items()},shared_prefix_worlds=observed,shared_prefix_max_abs=prefix_max,replay_branches=len(replay),replay=replay,elapsed_s=time.monotonic()-started,not_model_performance=True)
    (OUT/'data_replay.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='replay'}),flush=True)
if __name__=='__main__':main()
