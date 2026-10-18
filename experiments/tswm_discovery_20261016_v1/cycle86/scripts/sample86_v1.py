from common86_v1 import *
import platform,time

class ExactIntegrator:
    def solve(self,initial,times,system,drift,diffusion):
        a=drift.numpy().reshape(-1,2,2);rates,vectors=np.linalg.eigh(a);coordinates=np.einsum('nji,nj->ni',vectors,initial.numpy());powers=np.exp(rates[:,None,:]*times.numpy()[None,:,None].astype(np.float64));paths=np.einsum('nij,ntj->nti',vectors,coordinates[:,None,:]*powers)
        return torch.from_numpy(paths)
def native_method():
    text=SOURCE.read_text(encoding='utf-8');tree=ast.parse(text);matches=[(c,n) for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='solve_equations'];assert len(matches)==1
    cls,node=matches[0];env={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),env)
    return env['solve_equations'],dict(repository='FIM4Science/OpenFIM',commit='cee2bb53be1e8e92b73f98dfdbe3605efc9119cc',relative_file=SOURCE.relative_to(CACHE).as_posix(),sha256=sha(SOURCE),class_name=cls.name,start_line=node.lineno,end_line=node.end_lineno,method_ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),integrator_substituted='Exact symmetric linear continuous-time flow',released_checkpoint_training_provenance_verified=False)
def population(seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,1000]));rates=rng.uniform(-.8,.8,(4096,2));angles=rng.uniform(-np.pi,np.pi,4096);q=np.stack([np.stack([np.cos(angles),-np.sin(angles)],axis=-1),np.stack([np.sin(angles),np.cos(angles)],axis=-1)],axis=1);a=np.einsum('nij,nj,nkj->nik',q,rates,q);initial=rng.uniform(-1,1,(4096,16,2));return rates,angles,a,initial
def descriptor(rates,mask):
    values=rates[mask];hist=np.histogram2d(values[:,0],values[:,1],bins=8,range=[[-.8,.8],[-.8,.8]])[0];hist/=hist.sum();full=np.histogram2d(rates[:,0],rates[:,1],bins=8,range=[[-.8,.8],[-.8,.8]])[0];full/=full.sum()
    return dict(retained=int(mask.sum()),fraction=float(mask.mean()),positive_growth_fraction=float(np.mean(np.max(values,axis=1)>0)),mean_max_rate=float(np.mean(np.max(values,axis=1))),mean_rate1=float(values[:,0].mean()),mean_rate2=float(values[:,1].mean()),joint_histogram_total_variation=float(np.sum(np.abs(hist-full))/2))
def snapshot_probe(rates,initial,mask,split):
    target=rates.max(axis=1);energy=np.sum(initial[:,0]**2,axis=1);design=np.column_stack([np.ones(len(energy)),energy]);fit=mask&(split==0);selected=mask&(split==2);unselected=split==2;w=np.linalg.lstsq(design[fit],target[fit],rcond=None)[0];pred=design@w;constant=float(target[fit].mean())
    out={}
    for name,inds in [('selected_test',selected),('unselected_test',unselected)]:
        err=float(np.mean((pred[inds]-target[inds])**2));base=float(np.mean((constant-target[inds])**2));out[name]=dict(n=int(inds.sum()),mse=err,training_constant_mse=base,relative_mse_improvement=1-err/base)
    out.update(slope=float(w[1]),train_count=int(fit.sum()),initial_energy_rate_correlation=float(np.corrcoef(energy[mask],target[mask])[0,1]));return out,w,pred
def main():
    started=time.perf_counter();method,source=native_method();write(O/'SOURCE86_V1.json',source);log(dict(kind='sampling_start',pid=os.getpid(),source=source,protocol_sha256=sha(C/'PROTOCOL86_V1.md')));results=[]
    for seed in SEEDS:
        rates,angles,a,initial=population(seed);n=len(a);drift=torch.from_numpy(a.reshape(n,4));diff=torch.zeros(n,2,dtype=torch.float64);rng=np.random.default_rng(np.random.SeedSequence([seed,2000]));order=rng.permutation(n);split=np.zeros(n,dtype=np.int64);split[order[int(.6*n):int(.8*n)]]=1;split[order[int(.8*n):]]=2
        saved=dict(rates=rates,angles=angles,a=a,initial=initial,split=split);records=[]
        for paths in [1,4,16]:
            for threshold in [None,3.,6.]:
                owner=types.SimpleNamespace(chunk_size=n,num_all_paths=paths,num_steps=80,dt=.05,sampling_device='cpu',reject_threshold=threshold,integrator=ExactIntegrator(),system=None)
                hp,ht,dp,df=method(owner,torch.from_numpy(initial[:,:paths].reshape(n*paths,2)),drift,diff,n*paths)
                times=torch.linspace(0.,4.,81).numpy().astype(np.float64);coords=np.einsum('nji,npj->npi',np.stack([np.stack([np.cos(angles),-np.sin(angles)],axis=-1),np.stack([np.sin(angles),np.cos(angles)],axis=-1)],axis=1),initial[:,:paths]);q=np.stack([np.stack([np.cos(angles),-np.sin(angles)],axis=-1),np.stack([np.sin(angles),np.cos(angles)],axis=-1)],axis=1);reference=np.einsum('nij,nptj->npti',q,coords[:,:,None,:]*np.exp(rates[:,None,None,:]*times[None,None,:,None]));assert np.isfinite(reference).all()
                mask=np.ones(n,dtype=bool) if threshold is None else np.max(np.abs(reference),axis=(1,2,3))<threshold
                assert np.array_equal(dp.numpy(),a[mask].reshape(-1,4));max_error=float(np.max(np.abs(hp.numpy()-reference[mask])));assert max_error<1e-10
                tag=f'p{paths}_'+('none' if threshold is None else str(int(threshold)));probe,w,pred=snapshot_probe(rates,initial,mask,split);saved[tag+'_mask']=mask;saved[tag+'_probe_coef']=w;saved[tag+'_probe_prediction']=pred
                if paths==16 and threshold is None:saved['times']=times;saved['paths']=reference
                record=dict(seed=seed,paths=paths,threshold=threshold,tag=tag,distribution=descriptor(rates,mask),snapshot_probe=probe,max_native_vs_analytic_path_difference=max_error);records.append(record);log(dict(kind='sampling_condition',**record))
        path=RAW/f'population_{seed}_v1.npz';save(path,**saved);record=dict(seed=seed,conditions=records,array_path=str(path),array_sha256=sha(path));write(O/f'POPULATION86_{seed}_V1.json',record);results.append(record)
    out=dict(status='SOURCE_DERIVED_SAMPLER_DIAGNOSTIC_COMPLETE',utc=utc(),elapsed_seconds=time.perf_counter()-started,seeds=SEEDS,mechanisms_per_seed=4096,conditions_per_seed=9,source=source,populations=results,runtime=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),cuda_used=False),released_checkpoint_tested=False,new_method_validated=False,accepted_research_directions=0,protocol_sha256=sha(C/'PROTOCOL86_V1.md'),local_result_payload_files_written=0);write(O/'RESULT86_V1.json',out);print(json.dumps({k:v for k,v in out.items() if k!='populations'},indent=2),flush=True)
if __name__=='__main__':main()
