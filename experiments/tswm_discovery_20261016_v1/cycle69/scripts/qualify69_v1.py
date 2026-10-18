import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native69_v1 import *
manifest=read(OUT/'DATA_MANIFEST69_V1.json');items=[]
for case in manifest['cases']:
 assert sha(case['path'])==case['sha256'];items.append(dict(case=case,arrays={k:v for k,v in np.load(case['path'],allow_pickle=False).items()}))
meta=Meta();dyno=Dyno();rows=[]
for item in items:
 case=item['case'];data=item['arrays'];meta.set_context(data,case)
 for n in sorted({case['paper_initial'],case['official_initial']}):
  tu,iu,iy=meta.init_state(data,n,case['benchmark']);fu=tu[n:n+100]
  if len(fu)<100:fu=torch.cat([fu,fu[-1:].expand(100-len(fu))])
  m,s=meta.query(iu,iy,fu);unc,ucs=meta.query(iu,iy,fu,cached=False);altered,alts=meta.query(iu,iy,fu,hidden_fill=111.);em=float((m-unc).abs().max().cpu());es=float((s-ucs).abs().max().cpu());assert em<=1e-6 and es<=1e-6,(em,es);assert torch.equal(m,altered) and torch.equal(s,alts);rows.append(dict(case=case['case'],initial=n,contexts=meta.contexts,query_shape=list(m.shape),cached_uncached_mean_max_abs=em,cached_uncached_std_max_abs=es,hidden_future_replacement_mean_max_abs=0,hidden_future_replacement_std_max_abs=0))
 short=dict(data,test_u=data['test_u'][:256],test_y=data['test_y'][:256]);pred,scaler=dyno.forecast(short,case,items);assert pred.shape==short['test_y'].shape and np.isfinite(pred).all();rows[-1]['dyno_short_shape']=list(pred.shape);rows[-1]['dyno_scaler']=scaler
dump(OUT/'QUALIFICATION69_V1.json',dict(utc=utc(),status='PASS',native_meta_parameters=meta.parameters,native_meta_state_tensors=meta.state_tensors,cached_author_class='TSTransformerBenchmark',neural_fits=0,meta_source_receipt_sha256=sha(PROJECT/'cycle68'/'outputs'/'MODEL_FILES68_V1.json'),dyno_metadata=dyno.metadata,dyno_AST_sha256=dyno.AST_sha256,qualifications=rows,task_target_scores_computed=False,manifest_sha256=sha(OUT/'DATA_MANIFEST69_V1.json')));print(json.dumps(dict(status='PASS',qualifications=rows,dyno_metadata=dyno.metadata)),flush=True)
