"""Measure exact fixed sampler cost; no parameter or estimator selection."""
from sundial_common34 import *
import time
OUT=ROOT/'outputs/sampler_profile_v1'
def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    proof=json.loads((ROOT/'outputs/qualification_v1/verification.json').read_text());assert proof['status']=='passed'
    locks=validate_sources();assert proof['locks']==locks
    model=load_model();a=np.load(ROOT/'outputs/qualification_v1/jena.npz')
    hist=torch.tensor(a['history'],device='cuda');cache=cache_history(model,hist);seed(2026347)
    records=[]
    with torch.inference_mode():
        for n in [32,64,256,1024]:
            noise=torch.randn(n,1,720,device='cuda')
            from_noise(model,cache,noise);torch.cuda.synchronize();times=[]
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(3):
                start=time.perf_counter();y=from_noise(model,cache,noise);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                assert torch.isfinite(y).all()
            records.append({'paths':n,'times':times,'median_seconds':float(np.median(times)),
                'path_seconds':float(np.median(times))/n,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'flow_network_row_evaluations':50*n})
    result={'status':'measured','records':records,'locks':locks,'script_sha256':sha(__file__),
        'qualification_proof_sha256':sha(ROOT/'outputs/qualification_v1/verification.json'),
        'noise_dimension':720,'steps':50,'fp32':True,'not_a_model_performance_or_novelty_result':True}
    dump(OUT/'profile.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
