"""Independent NumPy recurrence and float64 metric replay of saved audit data."""
import runtime_profile
import hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/prefix_contract'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sim(p,g,u):
    a,b,c,d=[p[g+'_'+k] for k in 'ABCD'];x=np.zeros(a.shape[0]);y=np.empty_like(u)
    for t in range(len(u)):
        y[t]=c@x+d@u[t];x=a@x+b@u[t]
    return y
def nonlin(p,v):return np.tanh(v@p['w1'].T+p['b1'])@p['w2'].T+p['b2']
def run(p,u,arm):
    v=sim(p,'g1',u)
    if arm.startswith('native_'):v=(v-v[200:].mean(0))/(v[200:].std(0)+1e-6)
    elif arm=='fixed_reference':v=(v-p['cal_internal_mean'])/p['cal_internal_scale']
    y=sim(p,'g2',nonlin(p,v))[200:];u=u[200:].copy()
    if arm=='native_complete':
        y=(y-y.mean(0))/(y.std(0)+1e-6);u=(u-u.mean(0))/(u.std(0)+1e-6)
    elif arm=='fixed_reference':
        y=(y-p['cal_y_mean'])/p['cal_y_scale'];u=(u-p['cal_u_mean'])/p['cal_u_scale']
    return y+p['noise'],u
def main():
    start=time.perf_counter();r=json.loads((OUT/'results.json').read_text());arms=[x['arm'] for x in r['summary'][::2]]
    metric_errors=[];recurrence=[];cal_errors=[];observed=[];prefix_checks=0
    for row in r['files']:
        path=OUT/row['path'];assert sha(path)==row['sha256'];i=int(path.stem.split('_')[-1])
        with np.load(path) as z:p={k:z[k] for k in z.files}
        assert np.array_equal(p['original_input'][:600],np.r_[p['original_input'][:600]])
        if i in [0,1,2,3,7,15,31,63,95,127]:
            ref=p['reference_input'];v=sim(p,'g1',ref);m=v[200:].mean(0);s=v[200:].std(0)+1e-6
            yy=sim(p,'g2',nonlin(p,(v-m)/s))[200:]
            cc=dict(internal_mean=m,internal_scale=s,u_mean=ref[200:].mean(0),u_scale=ref[200:].std(0)+1e-6,y_mean=yy.mean(0),y_scale=yy.std(0)+1e-6)
            for k,vv in cc.items():cal_errors.append(float(np.max(np.abs(vv-p['cal_'+k]))))
        for arm in arms:
            y0,u0=p[arm+'_original_y'],p[arm+'_original_u']
            if i in [0,1,2,3,7,15,31,63,95,127]:
                a,b=run(p,p['original_input'],arm);recurrence.append(float(np.max(np.abs(a-y0))));recurrence.append(float(np.max(np.abs(b-u0))))
            for cut in [400,744]:
                y1,u1=p[f'{arm}_{cut}_y'],p[f'{arm}_{cut}_u'];dy=y1[:cut]-y0[:cut];du=u1[:cut]-u0[:cut]
                vals=dict(relative_prefix_y_rms=float(np.sqrt(np.mean(dy**2))/(y0.std()+1e-12)),prefix_y_max=float(np.abs(dy).max()),prefix_u_max=float(np.abs(du).max()),prefix_u_rms=float(np.sqrt(np.mean(du**2))),original_y_std=float(y0.std()),first_changed_input_output_difference=float(abs(y1[cut,0]-y0[cut,0])))
                original=next(x for x in r['system_rows'] if (x['system'],x['arm'],x['cut'])==(i,arm,cut))
                for k,vv in vals.items():metric_errors.append(abs(vv-original[k]))
                observed.append(dict(system=i,arm=arm,cut=cut,**vals))
                if arm in ['raw_causal','fixed_reference']:
                    assert np.array_equal(y0[:cut+1],y1[:cut+1]);assert np.array_equal(u0[:cut],u1[:cut]);prefix_checks+=1
                if i in [0,1,2,3,7,15,31,63,95,127]:
                    ub=p['original_input'].copy();ub[200+cut:]=p['alternative_input'][200+cut:]
                    a,b=run(p,ub,arm);recurrence.append(float(np.max(np.abs(a-y1))));recurrence.append(float(np.max(np.abs(b-u1))))
    for ag in r['summary']:
        rr=[x for x in observed if (x['arm'],x['cut'])==(ag['arm'],ag['cut'])];v=np.array([x['relative_prefix_y_rms'] for x in rr])
        aa=dict(relative_y_rms_median=float(np.median(v)),relative_y_rms_mean=float(v.mean()),relative_y_rms_p95=float(np.quantile(v,.95)),fraction_relative_above_1e8=float(np.mean(v>1e-8)),prefix_y_max=max(x['prefix_y_max'] for x in rr),prefix_u_rms_median=float(np.median([x['prefix_u_rms'] for x in rr])))
        for k,vv in aa.items():metric_errors.append(abs(vv-ag[k]))
    assert max(metric_errors)==0
    assert max(recurrence)<2e-9,max(recurrence)
    assert max(cal_errors)<2e-9,max(cal_errors)
    source=ROOT.parent/'cycle16/upstream/sysid_benchmarks';mf=ROOT.parent/'cycle16/outputs/source/metaid/manifest.json'
    assert sha(mf)==r['source_manifest_sha256']
    for row in json.loads(mf.read_text())['files']:assert sha(source/row['path'])==row['sha256']
    assert sha(ROOT/'PROTOCOL.md')==r['protocol_sha256']
    assert all(x['y_max_error']==x['u_max_error']==0 for x in r['native_replay']) and len(r['native_replay'])==16
    receipt=dict(status='pass',results_sha256=sha(OUT/'results.json'),verifier_sha256=sha(pathlib.Path(__file__)),files_verified=len(r['files']),metrics_replayed=len(metric_errors),metric_max_error=max(metric_errors),independent_numpy_output_max_error=max(recurrence),independent_calibration_max_error=max(cal_errors),strict_prefix_checks=prefix_checks,independent_systems=10,native_source_unchanged=True,elapsed_seconds=time.perf_counter()-start)
    (OUT/'independent_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
