"""Independent polynomial replay and arithmetic for own sensitivity controls."""
import runtime_profile
import hashlib,json,pathlib,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/innovation_pilot';OUT=ROOT/'outputs/active_subspace'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    tick=time.perf_counter();receipt=OUT/'independent_verification.json';assert not receipt.exists()
    result=json.loads((OUT/'results.json').read_text());assert len(result['rows'])==18
    assert sha(ROOT/'scripts/active_subspace_control.py')==result['script_sha256']
    assert sha(ROOT/'ACTIVE_SUBSPACE_CONTROL.md')==result['plan_sha256']
    teachers={r['checkpoint_sha256']:r for r in json.loads((BASE/'fit_records.json').read_text()) if r['method']=='fullobs'}
    replay=[];metrics=[];whiten=[];orthogonal=[];validation=[];checks=[]
    for row in result['rows']:
        file=OUT/row['model_file'];assert sha(file)==row['model_sha256']
        teacher=BASE/'checkpoints'/teachers[row['teacher_sha256']]['checkpoint'];assert sha(teacher)==row['teacher_sha256']
        m=np.load(file,allow_pickle=False)
        sets={n:np.load(BASE/'data'/(row['family']+'_'+n+'.npz'),allow_pickle=False) for n in ['train','val','test']}
        def observed(raw,s):return ((raw-s['mean'])/s['scale']@s['rotation'].T).astype('float32')
        def encode(raw,s):return (observed(raw,s)-m['input_mean'])@m['input_transform']
        def forward(raw,s):
            z=encode(raw,s);v=np.column_stack([z,s['action']])
            # Direct monomial products, independent of sklearn PolynomialFeatures.
            phi=np.prod(v[:,None,:]**m['feature_powers'][None,:,:],axis=-1)/m['feature_scales']
            zp=phi@m['transition_coefficients']
            return zp,np.column_stack([zp,np.ones(len(zp))])@m['diagnostic_decoder']
        for split in ['test','val']:
            s=sets[split];zp,pred=forward(s['current'],s)
            if split=='test':replay.append(float(np.max(np.abs(pred-m['prediction']))))
            else:
                score=float(np.mean((zp-encode(s['future'],s))**2));validation.append(abs(score-row['selected']['validation_latent_MSE']))
                best=min(row['candidates'],key=lambda r:r['validation_latent_MSE']);assert best==row['selected']
        tr=sets['train'];te=sets['test'];z=encode(tr['current'],tr)
        whiten.append(float(np.max(np.abs(np.cov(z,rowvar=False)-np.eye(4)))))
        U=m['output_basis'];orthogonal.append(float(np.max(np.abs(U.T@U-np.eye(U.shape[1])))))
        zt=encode(te['current'],te);current=np.column_stack([zt,np.ones(len(zt))])@m['diagnostic_decoder']
        replay.append(float(np.max(np.abs(current-m['current_prediction']))))
        lo=te['current'].copy();hi=lo.copy();lo[:,2]=.4;hi[:,2]=1.6
        modulation=forward(hi,te)[1][:,:2]-forward(lo,te)[1][:,:2]
        replay.append(float(np.max(np.abs(modulation-m['modulation']))))
        # Analytical physical response, no original physical_step call.
        truth=np.zeros_like(modulation)
        if row['family']=='rotating':truth[:,1]=.36*te['action'][:,1]/te['scale'][1]
        elif row['family']=='coupled_linear':truth[:,0]=.42*te['action'][:,0]/te['scale'][0]
        else:truth[:,0]=.48*np.tanh(te['action'][:,0])/te['scale'][0]
        # Original differences of float32 state updates carry rounding error.
        assert np.max(np.abs(truth-m['true_modulation']))<3e-6
        gain=float(np.mean((current[:,2]*te['scale'][2]+te['mean'][2]>1)==(te['current'][:,2]>1)))
        future=((te['future'][:,:2]-te['mean'][:2])/te['scale'][:2]).astype('float32')
        cur=((te['current'][:,:2]-te['mean'][:2])/te['scale'][:2]).astype('float32')
        recomputed={'gain_accuracy':gain,'native_physical_MSE':float(np.mean((m['prediction'][:,:2]-future)**2)),
            'current_physical_decoding_MSE':float(np.mean((current[:,:2]-cur)**2)),
            'modulation_relative_RMSE':float(np.sqrt(np.mean((modulation-m['true_modulation'])**2)/np.mean(m['true_modulation']**2)))}
        metrics.extend(abs(recomputed[k]-row[k]) for k in recomputed)
        checks.append(dict(key=row['key'],model_sha256=sha(file),teacher_sha256=sha(teacher),recomputed=recomputed))
    assert max(replay)<1e-9 and max(metrics)<1e-9 and max(validation)<1e-9
    assert max(whiten)<1e-9 and max(orthogonal)<1e-9
    r=dict(pass_=True,model_hashes=18,teacher_hashes=18,metrics_recomputed=len(metrics),
        manual_monomial_replay_max_error=max(replay),metric_max_error=max(metrics),validation_MSE_max_error=max(validation),
        whitened_covariance_max_error=max(whiten),output_basis_orthogonality_max_error=max(orthogonal),
        rows=checks,seconds=time.perf_counter()-tick,script_sha256=sha(pathlib.Path(__file__)),
        scope='Own saved models, independent monomials, analytical response and test arithmetic; not novelty or independent-cohort confirmation.')
    r['pass']=r.pop('pass_');receipt.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
if __name__=='__main__':main()
