"""TRAIN-only numeric qualification with explicit source axes and record boundaries."""
import sys,pathlib,time,ast,zipfile,re
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common74_v1 import *
import numpy as np
from scipy.io import loadmat,whosmat
def main():
 started=time.perf_counter();a=read(OUT/'DATA_SOURCE74_V2.json');assert a['status']=='ALL_REQUESTED_ARCHIVES_AND_TRAIN_MEMBERS_READY';records=[];arrays={};proof=[]
 for data in a['datasets']:
  assert sha(data['archive_path'])==data['archive_sha256']
  with zipfile.ZipFile(data['archive_path']) as z:
   assert z.testzip() is None
   for source in data['training_members']:
    assert sha(source['path'])==source['sha256'];assert hashlib.sha256(z.read(source['member'])).hexdigest()==source['sha256']
    headers=[dict(name=k,shape=list(s),dtype=t) for k,s,t in whosmat(source['path'])]
    if data['name']=='EMPS':
     assert pathlib.PurePosixPath(source['member']).name=='DATA_EMPS.mat'
     m=loadmat(source['path'],variable_names=['qm','vir','t']);assert m['qm'].ndim==m['vir'].ndim==m['t'].ndim==2 and all(m[k].shape[1]==1 for k in ['qm','vir','t'])
     u=m['vir'][:,0].copy();y=m['qm'][:,0].copy();t=m['t'][:,0];dt=float(t[1]-t[0]);assert dt>0 and np.allclose(np.diff(t),dt,rtol=1e-7,atol=1e-9);case='EMPS_DATA_EMPS';channels=dict(input='vir',output='qm',unit_status='Raw released source units; no magnitude-based inference',input_role='recorded feedback motor-force path, provided conditionally')
    else:
     assert 'Validation' not in source['member'] and 'SpecialOddMSine' not in source['member']
     m=loadmat(source['path'],variable_names=['Force','Acceleration','Fs']);assert m['Force'].ndim==2 and m['Force'].shape[0]==1 and m['Acceleration'].shape==(3,m['Force'].shape[1]) and m['Fs'].shape==(1,1)
     u=m['Force'][0].copy();y=m['Acceleration'][0].copy();dt=1./float(m['Fs'][0,0]);assert dt>0;case=pathlib.PurePosixPath(source['member']).stem;channels=dict(input='Force',output='Acceleration[0,:]',unit_status='Raw released source units; no magnitude-based inference',input_role='measured shaker-force path, provided conditionally')
    assert u.ndim==y.ndim==1 and u.shape==y.shape and len(u)>2000 and np.isfinite(u).all() and np.isfinite(y).all()
    cut=int(.8*len(u));origins=list(range(cut+256,len(u)-128+1,256));assert origins
    path=storage('artifact')/'canonical_training_v1'/(case+'.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,u=u.astype(np.float64),y=y.astype(np.float64))
    records.append(dict(dataset=data['name'],case=case,length=len(u),train_end_exclusive=cut,dev_gap=256,context=128,horizon=128,stride=256,dev_origins=origins,sampling_time=dt,source_member=source,channels=channels,array_path=str(path),array_sha256=sha(path),mat_variables=headers))
    arrays[case]=(u,y,cut);proof.append(dict(case=case,min_clock_step=dt,modeled_values_finite=True,train_dev_separated=True,official_test_values_read=False))
 scales={}
 for name in ['EMPS','F16']:
  rs=[r for r in records if r['dataset']==name];assert len(rs)==(1 if name=='EMPS' else 8)
  count=sum(r['train_end_exclusive'] for r in rs);s1=np.zeros(2);s2=np.zeros(2)
  for r in rs:
   u,y,cut=arrays[r['case']];v=np.column_stack([u[:cut],y[:cut]]);s1+=v.sum(0);s2+=(v*v).sum(0)
  mean=s1/count;std=np.sqrt(np.maximum(s2/count-mean*mean,1e-24));assert np.isfinite(mean).all() and (std>1e-12).all();scales[name]=dict(mean=mean.tolist(),std=std.tolist(),train_points=count,scope='Pooled first80% of all original TRAIN records in this system only')
 history=[];checked=0
 for d in sorted(PROJECT.glob('cycle*')):
  if not d.is_dir() or d==ROOT:continue
  for n in ['baseline_registry.md','experimental_results.md']:
   p=d/n
   if not p.exists():continue
   raw=p.read_bytes();checked+=1;text=raw.decode('utf-16' if raw[:2] in [b'\xff\xfe',b'\xfe\xff'] else 'utf-8-sig',errors='replace')
   lines=[dict(line=i+1,text=s) for i,s in enumerate(text.splitlines()) if re.search(r'Silverbox|\bF16\b|F-16|\bEMPS\b',s,re.I)]
   if lines:history.append(dict(path=p.relative_to(PROJECT).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),matches=lines))
 assert any(r['path'].startswith('cycle29/') for r in history)
 assert not any(re.search(r'\bF16\b|F-16|\bEMPS\b',x['text'],re.I) for r in history for x in r['matches']),'Inspect prior exposure before calling this a new score cohort'
 obj=dict(utc=utc(),status='NINE_ORIGINAL_TRAIN_RECORDS_QUALIFIED_OFFICIAL_TEST_NUMERICS_CLOSED',data_receipt_sha256=sha(OUT/'DATA_SOURCE74_V2.json'),records=records,scaling=scales,qualification=proof,history_review=dict(files_checked=checked,hits=history,scope='Prior baseline/result ledgers only, not a universal no-exposure proof',Silverbox_excluded_due_to_prior_scores=True),new_model_fits=0,official_test_numeric_values_parsed=False,local_result_files_written=0,protocol_sha256=sha(ROOT/'DATA_PROTOCOL74_V1.md'),scaling_clarification_sha256=sha(ROOT/'DATA_SCALING_CLARIFICATION74_V1.md'),wall_seconds=time.perf_counter()-started)
 dump(OUT/'TRAIN_DATA74_V1.json',obj);print(json.dumps(dict(status=obj['status'],records=len(records),train_points=sum(r['train_end_exclusive'] for r in records),dev_origins=sum(len(r['dev_origins']) for r in records),sampling_times={n:sorted({r['sampling_time'] for r in records if r['dataset']==n}) for n in scales},history_files_checked=checked,wall_seconds=obj['wall_seconds'])),flush=True)
if __name__=='__main__':main()
