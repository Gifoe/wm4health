"""Execute untouched official split definitions with an offline file-I/O adapter."""
import runtime_profile
import ast,hashlib,json,numbers,os,pathlib
import numpy as np
from scipy.io import loadmat
ROOT=pathlib.Path(__file__).resolve().parents[1];RAW=ROOT/'outputs/real_data';OUT=ROOT/'outputs/real_splits'
SOURCE=ROOT.parent/'cycle28/upstream/nonlinear_benchmarks/nonlinear_benchmarks'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def source_defs(path,names,env):
    tree=ast.parse(path.read_text(encoding='utf-8'));nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
def official_functions(paths):
    env=dict(np=np,Number=numbers.Number,os=os)
    source_defs(SOURCE/'utilities.py',['Input_output_data','atleast_2d_fun','always_return_tuples_of_datasets_fun'],env)
    seen=[]
    def offline_download(url,name_dir,**kwargs):seen.append(dict(name=name_dir,url=url));return str(RAW/'offline_native_paths'/name_dir)
    def offline_loadmat(path):
        basename=pathlib.Path(path).name.lower();assert basename in paths,(basename,paths);return loadmat(paths[basename])
    env.update(cashed_download=offline_download,loadmat=offline_loadmat)
    source_defs(SOURCE/'benchmarks.py',['Silverbox','Cascaded_Tanks','WienerHammerBenchMark'],env)
    return env,seen
def main():
    assert not OUT.exists();OUT.mkdir(parents=True);paths={};receipts=[]
    for family,filename,required in [('silverbox','SNLS80mV.mat',{'V1','V2'}),('cascaded_tanks','dataBenchmark.mat',{'uEst','uVal','yEst','yVal','Ts'}),('wiener_hammerstein','WienerHammerBenchMark.mat',{'uBenchMark','yBenchMark','fs'})]:
        path=RAW/family/filename
        if not path.exists():continue
        acquisition=RAW/family/'acquisition.json';fallback=RAW/family/'fallback_acquisition.json'
        receipt=json.loads((acquisition if acquisition.exists() else fallback).read_text(encoding='utf-8'));assert sha(path)==receipt['member_sha256']
        data=loadmat(path);assert required<=set(data)
        schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in data.items() if not k.startswith('__')};paths[filename.lower()]=path
        receipts.append(dict(family=family,filename=filename,bytes=path.stat().st_size,sha256=sha(path),schema=schema,acquisition_route=receipt.get('route','direct remote'),acquisition_receipt_sha256=sha(acquisition if acquisition.exists() else fallback)))
    assert 'snls80mv.mat' in paths and 'databenchmark.mat' in paths
    env,seen=official_functions(paths);records=[];windows=[];nativeproof=[]
    def record(family,role,train,test,direct_u,direct_y,aggregate=True):
        assert np.array_equal(test.u,direct_u) and np.array_equal(test.y,direct_y)
        assert np.isfinite(train.u).all() and np.isfinite(train.y).all() and np.isfinite(test.u).all() and np.isfinite(test.y).all()
        assert test.state_initialization_window_length==50
        p=OUT/f'{family}_{role}.npz';np.savez_compressed(p,train_u=train.u,train_y=train.y,test_u=test.u,test_y=test.y)
        mu=np.asarray([train.u.mean(),train.y.mean()]);std=np.asarray([train.u.std(),train.y.std()]);assert np.min(std)>1e-12
        candidates=np.arange(128,len(test.u)-128+1,128,dtype=int)
        indices=np.linspace(0,len(candidates)-1,min(32,len(candidates)),dtype=int);origins=candidates[indices] if aggregate else []
        for origin in origins:
            windows.append(dict(family=family,record=role,origin=int(origin),context_start=int(origin-128),context_stop=int(origin),target_start=int(origin),target_stop=int(origin+128),sampling_time=float(test.sampling_time),source_npz=p.name))
        item=dict(family=family,record=role,official_train_name=train.name,official_test_name=test.name,train_samples=len(train),test_samples=len(test),state_initialization_window_length=test.state_initialization_window_length,sampling_time=float(test.sampling_time),train_means=mu.tolist(),train_stds=std.tolist(),source_npz=p.name,sha256=sha(p),include_aggregate=aggregate,selected_origins=[int(v) for v in origins],official_split_vs_independent_mat_max_abs=0.)
        records.append(item)
    raw=loadmat(paths['snls80mv.mat']);u=raw['V1'][0];y=raw['V2'][0];train,tests=env['Silverbox'](train_test_split=True);s=int((127400-40650)*.75)
    assert np.array_equal(train.u,u[40650:40650+s]) and np.array_equal(train.y,y[40650:40650+s])
    record('silverbox','multisine',train,tests[0],u[40650+s:127400],y[40650+s:127400])
    record('silverbox','arrow_full',train,tests[1],u[100:40575],y[100:40575])
    record('silverbox','arrow_no_extrapolation',train,tests[2],u[100:32100],y[100:32100],aggregate=False)
    raw=loadmat(paths['databenchmark.mat']);train,test=env['Cascaded_Tanks'](train_test_split=True)
    assert np.array_equal(train.u,raw['uEst'][:,0]) and np.array_equal(train.y,raw['yEst'][:,0])
    record('cascaded_tanks','test',train,test,raw['uVal'][:,0],raw['yVal'][:,0])
    if 'wienerhammerbenchmark.mat' in paths:
        raw=loadmat(paths['wienerhammerbenchmark.mat']);train,test=env['WienerHammerBenchMark'](train_test_split=True);u=raw['uBenchMark'][:,0];y=raw['yBenchMark'][:,0]
        assert np.array_equal(train.u,u[5200:105200]) and np.array_equal(train.y,y[5200:105200]);record('wiener_hammerstein','test',train,test,u[105200:184000],y[105200:184000])
    result=dict(status='passed',protocol_sha256=sha(ROOT/'REAL_TRANSFER_PLAN.md'),script_sha256=sha(pathlib.Path(__file__)),official_source_files={p.name:sha(p) for p in [SOURCE/'benchmarks.py',SOURCE/'utilities.py']},route='AST original functions/types + only download/loadmat path adapter; native split math unchanged',receipts=receipts,records=records,windows=windows,n_forecast_windows=len(windows),physical_systems=len({r['family'] for r in records}),no_model_performance=True,official_full_open_loop_is_different_protocol=True,offline_download_calls=seen)
    dump(OUT/'manifest.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['records','windows','offline_download_calls','receipts']}),flush=True)
    print(json.dumps(dict(records=[{k:r[k] for k in ['family','record','train_samples','test_samples','sampling_time','include_aggregate']} for r in records])),flush=True)
if __name__=='__main__':main()
