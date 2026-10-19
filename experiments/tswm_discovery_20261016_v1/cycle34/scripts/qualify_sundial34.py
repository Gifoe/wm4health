"""Execute source-native sampling and exact-noise/cache qualification only."""
from sundial_common34 import *
import time,pandas as pd,traceback
OUT=ROOT/'outputs/qualification_v1'
def native_with_noise(model,hist,noise,horizon=96,flow_cache=None):
    original=torch.randn;calls=[]
    def controlled(*shape,**kwargs):
        dims=tuple(shape[0]) if len(shape)==1 and isinstance(shape[0],(list,tuple)) else tuple(shape)
        assert dims==tuple(noise.shape),(dims,noise.shape)
        assert not kwargs;calls.append(dims);return noise.clone()
    torch.randn=controlled
    try:
        if flow_cache is None:out=model.generate(hist,max_new_tokens=horizon,num_samples=noise.shape[0]//len(hist))
        else:
            out=model.flow_loss.sample(flow_cache['z'],noise.shape[0]//len(hist))[:,:,:horizon]
            out=out*flow_cache['scale'][:,None,:]+flow_cache['mean'][:,None,:]
    finally:torch.randn=original
    assert len(calls)==1
    return out,calls
def relative(a,b):return float((a-b).abs().max())/(1+float(a.abs().max()))
def real_fixture(label,relative_path,target):
    path=ROOT/'assets/fev'/relative_path;df=pd.read_parquet(path);assert len(df)==1
    row=df.iloc[0];values=np.asarray(row[target],dtype=np.float32);times=np.asarray(row['timestamp'])
    assert len(times)==len(values) and (np.diff(times)>np.timedelta64(0,'ms')).all()
    good=np.convolve(np.isfinite(values).astype(np.int32),np.ones(672,dtype=np.int32),mode='valid')
    eligible=np.flatnonzero(good==672);assert len(eligible);start=int(eligible[0])
    assert (np.diff(times[start:start+672])==np.diff(times[start:start+672])[0]).all()
    meta={'name':label,'data_path':str(path.relative_to(ROOT)),'data_sha256':sha(path),'id':str(row['id']),
        'target_field':target,'history_start_index':start,'history_start_timestamp':str(times[start]),
        'history_end_timestamp':str(times[start+671]),'full_points':len(values),'full_nonfinite':int((~np.isfinite(values)).sum()),
        'source_row_count':len(df),'history_finite':True,'no_data_future_used':True}
    return values[start:start+672][None],meta
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);begin=time.perf_counter();locks=validate_sources()
    seed(2026340);model=load_model();before=state_sha(model);torch.cuda.reset_peak_memory_stats()
    t=np.arange(672,dtype=np.float32)
    examples=[(np.stack([np.sin(t/13)+t/500,2*np.cos(t/31)-t/1000+5]),{'name':'deterministic_api_fixture','not_a_scientific_dataset':True})]
    examples += [real_fixture('solar','solar_with_weather/15T/train-00000-of-00001.parquet','target'),
                 real_fixture('jena','jena_weather/10T/train-00000-of-00001.parquet','target_0')]
    records=[]
    for index,(raw,meta) in enumerate(examples):
        hist=torch.tensor(raw,device='cuda');cache=cache_history(model,hist);s=4;b=len(hist)
        seed(20263400+index);noise=torch.randn(s*b,720)
        with torch.inference_mode():
            native,calls=native_with_noise(model,hist,noise)
            direct,flowcalls=native_with_noise(model,hist,noise,flow_cache=cache)
            explicit=from_noise(model,cache,noise.cuda().reshape(s,b,720))
            prefix=from_noise(model,cache,noise[:2*b].cuda().reshape(2,b,720))
            nativeprefix,_=native_with_noise(model,hist,noise[:2*b])
        assert native.shape==(b,4,96) and torch.isfinite(native).all()
        errors={'api_vs_cached_native_flow':relative(native,direct),'api_vs_explicit_noise':relative(native,explicit),
                'explicit_prefix_vs_full':relative(explicit[:,:2],prefix),'native_prefix_vs_full':relative(native[:,:2],nativeprefix)}
        assert max(errors.values())<2e-5,(meta['name'],errors)
        path=OUT/(meta['name']+'.npz')
        np.savez_compressed(path,history=raw,noise=noise.numpy(),condition=cache['z'].cpu().numpy(),
            mean=cache['mean'].cpu().numpy(),scale=cache['scale'].cpu().numpy(),native=native.cpu().numpy(),
            cached_native=direct.cpu().numpy(),explicit=explicit.cpu().numpy(),prefix=prefix.cpu().numpy())
        records.append({**meta,'shape':list(native.shape),'controlled_native_rng_calls':calls,'native_flow_rng_calls':flowcalls,
             'relative_errors':errors,'fixture_path':path.name,'fixture_sha256':sha(path)})
        print(json.dumps({'fixture':meta['name'],'relative_errors':errors}),flush=True)
    # Smooth probe in the last real context, along its normalized gradient.
    e=noise[:1].cuda().reshape(1,1,720).detach().requires_grad_(True)
    objective=from_noise(model,cache,e).mean();gradient=torch.autograd.grad(objective,e)[0]
    norm=float(gradient.norm());assert np.isfinite(norm) and norm>1e-8
    direction=gradient.detach()/norm;eps=.01
    with torch.no_grad():plus=from_noise(model,cache,e.detach()+eps*direction).mean();minus=from_noise(model,cache,e.detach()-eps*direction).mean()
    fd=float((plus-minus)/(2*eps));analytic=float((gradient*direction).sum());gradient_error=abs(fd-analytic)/(1e-6+abs(fd)+abs(analytic))
    assert gradient_error<.03,(analytic,fd,gradient_error)
    np.savez_compressed(OUT/'gradient_fixture.npz',noise=e.detach().cpu().numpy(),direction=direction.cpu().numpy(),gradient=gradient.cpu().numpy(),
                        plus=plus.cpu().numpy(),minus=minus.cpu().numpy())
    after=state_sha(model);assert before==after
    result={'status':'qualified_pending_independent_replay','locks':locks,'records':records,'runtime':runtime(),
        'script_sha256':sha(__file__),'model_state_before':before,'model_state_after':after,
        'parameters':sum(p.numel() for p in model.parameters()),'native_steps':50,'noise_dimension':720,'qualified_horizon':96,
        'gradient':{'analytic':analytic,'finite_difference':fd,'step':eps,'symmetric_relative_error':gradient_error,
                    'fixture_sha256':sha(OUT/'gradient_fixture.npz')},
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'seconds':time.perf_counter()-begin,
        'not_a_method_validation':True,'accepted_research_directions':0}
    dump(OUT/'result.json',result);print(json.dumps({'status':result['status'],'parameters':result['parameters'],'seconds':result['seconds'],'gradient':result['gradient']}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        if OUT.exists():dump(OUT/'failure.json',{'traceback':traceback.format_exc(),'script_sha256':sha(__file__)})
        raise
