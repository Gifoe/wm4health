"""Executable API/data/numeric qualification before the 12 registered fits."""
from common37_v2 import *
import time
O=ROOT/'outputs/preflight_v2';assert not O.exists();O.mkdir(parents=True)
source=json.loads((BASE/'cycle36/source_audit/manifest.json').read_text(encoding='utf-8'))['files']
for item in source:
    if item['path'].startswith('KITE/'):
        assert sha(BASE/'cycle36/source_audit'/item['path'])==item['sha256']
results=[]
for name in ['wind','solar']:
    groups,meta=dataset(name)
    (O/f'{name}_data.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    g=groups['fit'];small={k:v[:2] for k,v in g.items()};t=tensors(small)
    for kind in ['quantile_mlp','kite_l1','kite_l2']:
        seed(SEEDS[0]);m=make_model(kind,g);initial=state_sha(m)
        optim=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=.0001,foreach=False)
        times=[];none=[]
        for i in range(3):
            m.train();seed(123+i);optim.zero_grad(set_to_none=True);torch.cuda.synchronize();tic=time.perf_counter()
            value=loss(m,kind,t,torch.arange(2,device='cuda'));assert torch.isfinite(value)
            value.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            none=[n for n,p in m.named_parameters() if p.grad is None];torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            optim.step();torch.cuda.synchronize();times.append(time.perf_counter()-tic)
        m.eval();before=state_sha(m);paths=predict(m,kind,small,meta,n=4,rng_seed=71)
        assert paths.shape==(2,4,24) and np.isfinite(paths).all() and state_sha(m)==before
        diff=None
        if kind!='quantile_mlp':
            seed(71)
            with torch.no_grad():v=m.inference(t['hy'],t['hx'],t['fx'],num_samples=4).squeeze(-1).cpu().numpy()
            v=v.astype('float64')*meta['scaler']['y_std']+meta['scaler']['y_mean']
            diff=float(np.max(np.abs(v-paths.mean(1))));assert diff<2e-5
        s=scores(paths,small['y']);assert all(np.isfinite(x).all() for x in s.values())
        results.append({'data':name,'model':kind,'parameter_count':sum(p.numel() for p in m.parameters()),
            'initial_sha256':initial,'three_step_seconds':times,'none_gradient_parameters':none,
            'native_mean_vs_path_mean_max_abs':diff,'inference_state_unchanged':True})
        del m,optim;t=torch.cuda.empty_cache() or tensors(small)
    # Score identities checked on observed FIT-only fixture, not VAL/TEST.
    a=np.random.default_rng(21).normal(size=(2,8,24));b=a.copy()
    for h in range(24):b[:,:,h]=a[:,np.random.default_rng(h).permutation(8),h]
    assert np.allclose(scores(a,small['y'])['crps'],scores(b,small['y'])['crps'],atol=1e-12)
    z=np.array([[[0.],[2.]]]);assert abs(float(fair_crps(z,np.array([[1.]]))[0,0]))<1e-12
    del groups,g,t
    torch.cuda.empty_cache()
out={'passed':True,'results':results,'no_val_or_test_scores':True,'qualification_updates':18,
    'protocol_sha256':sha(ROOT/'LEARNING_PROTOCOL.md'),'common_sha256':sha(ROOT/'scripts/common37_v2.py'),
    'data_script_sha256':sha(ROOT/'scripts/prepare_data37_v3.py')}
(O/'result.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'cells':len(results),'params':[(x['data'],x['model'],x['parameter_count']) for x in results],
                  'mean_api_max_abs':max(x['native_mean_vs_path_mean_max_abs'] or 0 for x in results),
                  'median_step_seconds':float(np.median([t for x in results for t in x['three_step_seconds']]))}))
