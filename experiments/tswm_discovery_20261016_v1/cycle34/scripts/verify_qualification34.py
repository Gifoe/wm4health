"""Fresh native API/flow replay and raw-file audit, separate from producer."""
from sundial_common34 import *
import time,pandas as pd
OUT=ROOT/'outputs/qualification_v1'
def official(model,hist,bank,flow=None):
    old=torch.randn;count=0
    def draw(*shape,**kwargs):
        nonlocal count
        assert tuple(shape)==tuple(bank.shape) and not kwargs
        count+=1;return bank.clone()
    torch.randn=draw
    try:
        if flow is None:answer=model.generate(hist,max_new_tokens=96,num_samples=len(bank)//len(hist))
        else:answer=model.flow_loss.sample(flow,len(bank)//len(hist))[:,:,:96]
    finally:torch.randn=old
    assert count==1
    return answer
def main():
    start=time.perf_counter();assert not (OUT/'verification.json').exists()
    r=json.loads((OUT/'result.json').read_text());locks=validate_sources();assert r['locks']==locks
    assert r['script_sha256']==sha(ROOT/'scripts/qualify_sundial34.py')
    model=load_model();assert sum(p.numel() for p in model.parameters())==r['parameters']==128329680
    before=state_sha(model);assert before==r['model_state_before']==r['model_state_after']
    maximum=0.;checks=0
    for record in r['records']:
        path=OUT/record['fixture_path'];assert sha(path)==record['fixture_sha256'];a=np.load(path)
        if 'data_path' in record:
            rawpath=ROOT/record['data_path'];assert sha(rawpath)==record['data_sha256']
            row=pd.read_parquet(rawpath).iloc[0];raw=np.asarray(row[record['target_field']],dtype=np.float32)
            run=0;first=None
            for i,val in enumerate(raw):
                run=run+1 if np.isfinite(val) else 0
                if run==672:first=i-671;break
            assert first==record['history_start_index'] and str(row['id'])==record['id']
            assert np.array_equal(raw[first:first+672][None],a['history'])
            times=np.asarray(row['timestamp']);assert str(times[first])==record['history_start_timestamp']
            assert str(times[first+671])==record['history_end_timestamp']
        else:
            t=np.arange(672,dtype=np.float32);expected=np.stack([np.sin(t/13)+t/500,2*np.cos(t/31)-t/1000+5])
            assert np.array_equal(expected,a['history'])
        hist=torch.tensor(a['history'],device='cuda');bank=torch.tensor(a['noise'])
        with torch.inference_mode():answer=official(model,hist,bank)
        assert tuple(answer.shape)==tuple(record['shape']) and torch.isfinite(answer).all()
        for name in ['native','cached_native','explicit']:
            val=a[name];error=float(np.abs(answer.cpu().numpy()-val).max());maximum=max(maximum,error)
            assert error/(1+np.abs(val).max())<2e-5;checks+=1
        assert np.array_equal(a['prefix'],a['explicit'][:,:2])
        # Independently calculate normalization and native conditioning, without
        # invoking the producer's cache/from_noise implementation.
        center=hist.mean(-1,keepdim=True);scale=hist.std(-1,unbiased=False,keepdim=True)+1e-5
        assert np.array_equal(center.cpu().numpy(),a['mean']) and np.array_equal(scale.cpu().numpy(),a['scale'])
        with torch.inference_mode():z=model.model(input_ids=(hist-center)/scale,use_cache=False,return_dict=True).last_hidden_state[:,-1]
        assert torch.equal(z,torch.tensor(a['condition'],device='cuda'));checks+=1
    # Independently differentiate the ORIGINAL source FlowLoss.sample instead
    # of the explicit adapter, including its native CPU->GPU noise operation.
    gf=OUT/'gradient_fixture.npz';assert sha(gf)==r['gradient']['fixture_sha256'];g=np.load(gf)
    e=torch.tensor(g['noise'].reshape(1,720),requires_grad=True);direction=torch.tensor(g['direction'].reshape(1,720))
    original=official(model,hist,e,flow=z)*scale[:,None,:]+center[:,None,:]
    gradient=torch.autograd.grad(original.mean(),e)[0]
    assert np.allclose(gradient.detach().numpy(),g['gradient'].reshape(1,720),rtol=2e-5,atol=2e-6)
    eps=r['gradient']['step']
    with torch.no_grad():
        plus=(official(model,hist,e.detach()+eps*direction,flow=z)*scale[:,None,:]+center[:,None,:]).mean()
        minus=(official(model,hist,e.detach()-eps*direction,flow=z)*scale[:,None,:]+center[:,None,:]).mean()
    analytic=float((gradient*direction).sum());fd=float((plus-minus)/(2*eps))
    assert abs(analytic-r['gradient']['analytic'])<1e-5 and abs(fd-r['gradient']['finite_difference'])<1e-5
    rel=abs(analytic-fd)/(1e-6+abs(analytic)+abs(fd));assert rel<.03
    assert state_sha(model)==before
    result={'status':'passed','qualification_sha256':sha(OUT/'result.json'),'script_sha256':sha(__file__),
      'locks':locks,'native_replay_and_condition_checks':checks,'maximum_native_output_abs_error':maximum,
      'gradient_original_source_symmetric_error':rel,'raw_histories_and_earliest_finite_rule_checked':True,
      'model_state_unchanged':True,'seconds':time.perf_counter()-start,
      'scope':'native generator qualification only; no novel method, forecast calibration or research direction accepted'}
    dump(OUT/'verification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
