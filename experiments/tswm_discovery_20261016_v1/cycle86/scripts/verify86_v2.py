from common86_v1 import *
from scipy.linalg import expm
import urllib.request

def independently_describe(rates,mask):
    rr=rates[mask];bins=np.linspace(-.8,.8,9);a=np.histogramdd(rr,bins=(bins,bins))[0];b=np.histogramdd(rates,bins=(bins,bins))[0]
    return dict(retained=int(mask.sum()),fraction=float(mask.mean()),positive_growth_fraction=float((rr.max(axis=1)>0).mean()),mean_max_rate=float(rr.max(axis=1).mean()),mean_rate1=float(rr[:,0].mean()),mean_rate2=float(rr[:,1].mean()),joint_histogram_total_variation=float(np.abs(a/a.sum()-b/b.sum()).sum()/2))
def main():
    result=json.loads((O/'RESULT86_V1.json').read_text());assert result['status']=='SOURCE_DERIVED_SAMPLER_DIAGNOSTIC_COMPLETE';source=result['source'];url='https://raw.githubusercontent.com/'+source['repository']+'/'+source['commit']+'/'+source['relative_file']
    receipt=json.loads((O/'SOURCE_PUBLIC_HTTPS86_V1.json').read_text());assert receipt['source_url']==url and receipt['normal_verified_https'] and receipt['public_source_equals_pinned_server_cache'];raw=SOURCE.read_bytes();assert sha(SOURCE)==receipt['source_sha256']
    assert hashlib.sha256(raw).hexdigest()==source['sha256']==sha(SOURCE)
    tree=ast.parse(raw.decode());node=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==source['class_name'] for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='solve_equations');assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==source['method_ast_sha256']
    checks=[];total=0
    for entry in result['populations']:
        assert sha(entry['array_path'])==entry['array_sha256'];d=np.load(entry['array_path'],allow_pickle=False);rates=d['rates'];angle=d['angles'];q=np.array([[[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]] for t in angle]);coordinates=np.einsum('nki,npk->npi',q,d['initial']);expected=np.einsum('nij,nptj->npti',q,coordinates[:,:,None]*np.exp(rates[:,None,None]*d['times'][None,None,:,None]));paths=d['paths'];physical_error=float(np.max(np.abs(expected-paths)));assert physical_error<1e-10 and np.isfinite(paths).all();total+=paths.size
        scipy_error=0.
        for j in range(0,4096,64):
            for k in [0,1,7,23,80]:
                p=d['initial'][j]@expm(d['times'][k]*d['a'][j]).T;scipy_error=max(scipy_error,float(np.max(np.abs(p-paths[j,:,k]))))
        assert scipy_error<1e-10
        probe_error=0.
        for record in entry['conditions']:
            p,threshold,tag=record['paths'],record['threshold'],record['tag'];mask=np.ones(4096,dtype=bool) if threshold is None else np.all(np.abs(paths[:,:p])<threshold,axis=(1,2,3));assert np.array_equal(mask,d[tag+'_mask']);summary=independently_describe(rates,mask)
            for key,value in summary.items():assert np.isclose(value,record['distribution'][key],atol=1e-13,rtol=1e-13)
            x=np.c_[np.ones(4096),np.einsum('ni,ni->n',d['initial'][:,0],d['initial'][:,0])];y=rates.max(axis=1);fit=mask&(d['split']==0);qq,rr=np.linalg.qr(x[fit]);w=np.linalg.solve(rr,qq.T@y[fit]);assert np.allclose(w,d[tag+'_probe_coef'],atol=1e-12,rtol=1e-12);prediction=x@w;probe_error=max(probe_error,float(np.max(np.abs(prediction-d[tag+'_probe_prediction']))));assert probe_error<1e-12
            constant=float(y[fit].mean())
            for name,m in [('selected_test',mask&(d['split']==2)),('unselected_test',d['split']==2)]:
                mse=float(np.mean((prediction[m]-y[m])**2));baseline=float(np.mean((constant-y[m])**2));r=record['snapshot_probe'][name];assert r['n']==int(m.sum()) and abs(mse-r['mse'])<1e-12 and abs(baseline-r['training_constant_mse'])<1e-12 and abs(1-mse/baseline-r['relative_mse_improvement'])<1e-12
        for threshold in ['3','6']:
            for p1,p2 in [(1,4),(4,16)]:assert not np.any(d[f'p{p2}_{threshold}_mask']&~d[f'p{p1}_{threshold}_mask'])
        checks.append(dict(seed=entry['seed'],trajectory_scalars=paths.size,max_rotation_reconstruction_difference=physical_error,max_scipy_expm_difference=scipy_error,max_snapshot_probe_replay_difference=probe_error))
    out=dict(status='PASS',utc=utc(),result_sha256=sha(O/'RESULT86_V1.json'),source_public_https_sha256=hashlib.sha256(raw).hexdigest(),source_url=url,verified_trajectory_scalars=total,checks=checks,verification_protocol_sha256=sha(C/'VERIFICATION86_V1.md'),released_checkpoint_provenance_verified=False,new_method_validated=False,local_result_payload_files_written=0);write(O/'VERIFICATION86_V1.json',out);print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
