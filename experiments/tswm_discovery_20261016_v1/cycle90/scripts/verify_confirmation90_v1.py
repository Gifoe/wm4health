from common90_v1 import *
import numpy as np,time

def dist(a,b):return np.sqrt(np.mean((a[:,None]-b[None,:])**2,axis=(2,3)))

def main():
    assert not (O/'CONFIRMATION_VERIFICATION90_V1.json').exists();started=time.perf_counter();r=read(O/'ORDINARY_CONFIRMATION90_V2.json');assert sha(r['payload_path'])==r['payload_sha256']
    ordinary=read(O/'ORDINARY_CONTROL90_V1.json');library=read(O/'LIBRARY90_V1.json');pi0=np.array([.4,.2,.1,.08,.07,.05,.04,.035,.025]);maximum=0.;values=0;arrays={}
    with np.load(r['payload_path']) as raw:
        for arm,receipt in [('unsupervised_pruned',ordinary),('supervised_library',library)]:
            assert sha(receipt['checkpoint_path'])==receipt['checkpoint_sha256']
            with np.load(receipt['checkpoint_path']) as cp:
                scale=cp['state_scale'] if arm=='unsupervised_pruned' else cp['scale'];center=cp['state_center'] if arm=='unsupervised_pruned' else cp['center']
                for family in [0,1]:
                    pred=raw[f'f{family}_{arm}_predictions'];bank=raw[f'f{family}_reference'];dense=raw[f'f{family}_dense_reference'];pi=raw[f'f{family}_{arm}_pi']
                    assert np.isfinite(pred).all() and abs(pi.sum()-1)<1e-12
                    pn=(pred-center)/scale;bn=(bank-center)/scale;dn=(dense-center)/scale;cost=[];labels=[]
                    rows=[z for z in r['queries'] if z['arm']==arm and z['family']==family]
                    for q,z in enumerate(rows):
                        ab=dist(pn[q],bn[q]);aa=dist(pn[q],pn[q]);bb=dist(bn[q],bn[q]);mean=np.einsum('i,itd->td',pi0,bn[q])[None]
                        energy=float(pi@ab@pi0-.5*pi@aa@pi);det=float(dist(mean,bn[q])[0]@pi0);spread=float(np.sqrt(np.sum(pi0*dist(bn[q],mean)[:,0]**2)));ratio=float(pi@ab.min(1)/spread)
                        measured=dict(energy_score=energy,exact_population_energy_score=float(.5*pi0@bb@pi0),energy_minus_deterministic=energy-det,off_manifold_ratio=ratio)
                        for key,v in measured.items():maximum=max(maximum,abs(v-z[key]));values+=1
                        pe=pi@np.stack([np.max(abs(pred[q,:,:,0]),1)>.75,pred[q,:,-1,0]>.25],1);te=pi0@np.stack([np.max(abs(bank[q,:,:,0]),1)>.75,bank[q,:,-1,0]>.25],1)
                        maximum=max(maximum,float(abs(pe-z['event_probability']).max()),float(abs(te-z['true_event_probability']).max()));values+=4
                        labels.append(ab.argmin(1));cost.append(dist(pn[q],dn[q])**2)
                    cost=np.stack(cost);labels=np.stack(labels)
                    for z in [v for v in r['joint'] if v['arm']==arm and v['family']==family]:
                        ix=z['group']*4+np.arange(z['K']);common=np.min(cost[ix].mean(0),1);separate=np.min(cost[ix],axis=2).mean(0)
                        expected=float(pi@(common-separate));maximum=max(maximum,abs(expected-z['common_minus_separate_mse']));values+=1
                        same=float(pi@np.all(labels[ix]==labels[ix[:1]],axis=0));maximum=max(maximum,abs(same-z['nearest_reference_bin_shared_agreement']));values+=1
    assert maximum<1e-10,maximum
    for summary in r['summary']:
        assert summary['competence_screen_passed'] and summary['energy_minus_deterministic']<0 and summary['off_manifold_ratio']<=.5
    assert r['exact_shared_physics_grid_gap_max_abs']<1e-12
    result=dict(utc=utc(),status='INDEPENDENT_BASELINE_CONFIRMATION_RAW_METRICS_REPLAYED',values_recomputed=values,max_abs_error=maximum,raw_payload_sha256=r['payload_sha256'],all_four_arm_family_screens_passed=True,native_unqualified_evaluation_not_opened=True,continuous_grid_evaluator_not_a_theorem=True,accepted_directions=0,full_goal_achieved=False,elapsed_seconds=time.perf_counter()-started)
    write(O/'CONFIRMATION_VERIFICATION90_V1.json',result);log(dict(stage='confirmation_replay_closed',**{k:v for k,v in result.items() if k!='utc'}))

if __name__=='__main__':main()
