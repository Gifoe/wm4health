import sys,pathlib,ast,types,os,numbers
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common69_v1 import *
from scipy.io import loadmat
receipt=read(OUT/'DATA_SOURCE69_V2.json') if (OUT/'DATA_SOURCE69_V2.json').exists() else read(OUT/'ACQUISITION69_V1.json');assert len(receipt['sources'])==20 and len(receipt['data'])==3
for row in receipt['sources']+receipt['data']:assert sha(row.get('server_path',row.get('path')))==row['sha256']
sources={r['path']:pathlib.Path(r['server_path']) for r in receipt['sources']};scope=dict(np=np,os=os,loadmat=loadmat,Number=numbers.Number)
def exact_nodes(path,names):
 tree=ast.parse(path.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names];assert {n.name for n in nodes}==set(names);return ast.Module(body=nodes,type_ignores=[])
ut=exact_nodes(sources['official_loader/nonlinear_benchmarks/utilities.py'],['Input_output_data','atleast_2d_fun','always_return_tuples_of_datasets_fun']);exec(compile(ut,'<exact_safe_official_data_containers69>','exec'),scope)
root=storage('data');paths={r['name']:pathlib.Path(r['server_path']) for r in receipt['data']}
def cache(url,name,**kwargs):
 assert name in paths
 return str(root/name)
scope['cashed_download']=cache;bt=exact_nodes(sources['official_loader/nonlinear_benchmarks/benchmarks.py'],['CED','Cascaded_Tanks','WienerHammerBenchMark']);exec(compile(bt,'<exact_official_splits_no_downloader69>','exec'),scope)
rows=[];audits=[];artifact=storage('artifact')/'datasets';artifact.mkdir(exist_ok=True)
for benchmark,func in [('WH','WienerHammerBenchMark'),('Tanks','Cascaded_Tanks'),('CED','CED')]:
 train,test=scope[func]();train=list(train) if isinstance(train,(list,tuple)) else [train];test=list(test) if isinstance(test,(list,tuple)) else [test]
 for i,(tr,te) in enumerate(zip(train,test)):
  arrays={k:np.asarray(v,dtype='float64') for k,v in dict(train_u=tr.u,train_y=tr.y,test_u=te.u,test_y=te.y).items()};assert all(v.ndim==1 and np.isfinite(v).all() for v in arrays.values());case=benchmark+str(i);p=artifact/(case+'.npz');assert not p.exists();np.savez_compressed(p,**arrays);r=dict(case=case,benchmark=benchmark,index=i,path=str(p),sha256=sha(p),train_length=len(tr),test_length=len(te),sample_time=float(tr.sampling_time),official_initial=int(te.state_initialization_window_length),paper_initial=50 if benchmark=='WH' else 5 if benchmark=='Tanks' else 4,context_length=100000 if benchmark=='WH' else 1000 if benchmark=='Tanks' else 400,context_stride=400 if benchmark=='WH' else 10 if benchmark=='Tanks' else 400,source_path=str(paths[func]),source_sha256=sha(paths[func]),released_units='As released; no magnitude-based physical-unit inference');rows.append(r);audits.append(dict(case=case,arrays={k:dict(shape=list(v.shape),min=float(v.min()),max=float(v.max()),mean=float(v.mean()),std=float(v.std())) for k,v in arrays.items()}))
dump(OUT/'DATA_MANIFEST69_V1.json',dict(utc=utc(),status='OFFICIAL_SPLITS_EXECUTED_WITH_SAFE_AST_AND_VERIFIED_NUMERIC_INPUTS',cases=rows,descriptive_audit=audits,official_loader_AST_sha256=hashlib.sha256(ast.unparse(bt).encode()).hexdigest(),data_container_AST_sha256=hashlib.sha256(ast.unparse(ut).encode()).hexdigest(),download_or_TLS_fallback_executed=False,protocol_sha256=sha(ROOT/'PROTOCOL69_V1.md'),source_receipt=str(OUT/('DATA_SOURCE69_V2.json' if (OUT/'DATA_SOURCE69_V2.json').exists() else 'ACQUISITION69_V1.json')),source_receipt_sha256=sha(OUT/('DATA_SOURCE69_V2.json' if (OUT/'DATA_SOURCE69_V2.json').exists() else 'ACQUISITION69_V1.json'))));print(json.dumps(dict(status='DATA_READY',cases=[{k:r[k] for k in ['case','train_length','test_length','official_initial','paper_initial']} for r in rows])),flush=True)
