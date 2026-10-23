"""Preserve four completed evaluations; execute only two missing cases and numerical control."""
import pathlib,sys,json,hashlib,datetime,subprocess,traceback,ast
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'pendulum_native_completion_v1';OUT.mkdir(exist_ok=False)
OLD=ROOT/'pendulum_native_recipe_v1'
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\pendulum_native_recipe_v1')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 assert read(ROOT/'pendulum_native_execution_v1/exit.json')['exit_code']==1
 failure=read(OLD/'refined_test/FAILURE_V1.json');diffs=ast.literal_eval(failure['exception']);assert diffs['1']==diffs['2']==0 and 0<diffs['0']<1e-6
 data=read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json');acq=read(ROOT/'pendulum_acquisition_recovery_v2/RESULTS_V1.json');src=pathlib.Path(acq['source_root']);ckpt=src/'pendulum/checkpoints/meta_ilad_s1000.pt'
 state=torch.load(ckpt,weights_only=True,map_location='cpu');h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
 expected=h.hexdigest();cells=[]
 for split in ['test','ood','trainenv']:
  p=OLD/('paper_'+split)/'RESULTS_V1.json';row=read(p);assert row['loaded_state_digest']==row['final_state_digest']==expected
  for f,item in row['payloads'].items():assert sha_file(ART/('paper_'+split)/f)==item['sha256']
  cells.append(row)
 native=read(OLD/'refined_test/NATIVE_METRICS_V1.json');folder=ART/'refined_test';raw=next(x for x in data['sets'] if x['split']=='test')
 for f in ['predictions.npy','reconstructions.npy']:assert np.load(folder/f,mmap_mode='r').shape==(3000,40,32,32)
 recovered=dict(status='COMPLETED_BENCHMARK_RECOVERED_AFTER_POST_EVAL_SHADOW_FAILURE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),variant='refined',split='test',seed=1000,sequences=3000,native_metrics=native,checkpoint_sha256=sha_file(ckpt),loaded_state_digest=expected,final_state_digest=expected,digest_provenance='Reconstructed from strict-loaded checkpoint; original live unchanged-state assertion reached before saved metrics, live digest not separately persisted',source_revision=acq['source_revision'],data_sha256=raw['file_sha256'],payloads={p.name:dict(bytes=p.stat().st_size,sha256=sha_file(p)) for p in folder.iterdir() if p.is_file()},preserved_original_shadow_failure=failure,new_benchmark_calls=0,new_fits=0,local_payload_sync=False)
 dump('RECOVERED_REFINED_TEST_V1.json',recovered);cells.append(recovered)
 for split in ['ood','trainenv']:
  print(json.dumps(dict(stage='previously_unstarted_case',variant='refined',split=split)),flush=True)
  cp=subprocess.run([sys.executable,'-I','-B','-X','utf8','-u',str(ROOT/'scripts/native_pendulum_eval_worker_v2.py'),'--variant','refined','--split',split],cwd=ROOT);assert cp.returncode==0,split
  cells.append(read(ROOT/'pendulum_native_remaining_v1'/('refined_'+split)/'RESULTS_V1.json'))
 cp=subprocess.run([sys.executable,'-I','-B','-X','utf8','-u',str(ROOT/'scripts/qualify_pendulum_numeric_boundary_v1.py')],cwd=ROOT);assert cp.returncode==0,'numeric qualification'
 q=read(ROOT/'pendulum_numeric_qualification_v1/RESULTS_V1.json')
 result=dict(status='SIX_UNIQUE_NATIVE_EVALUATIONS_COMPLETE_WITH_PRESERVED_QUALIFICATION_FAILURE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cells=cells,complete_evaluator_invocations=6,actual_complete_evaluator_invocations=6,original_queue_complete_benchmarks=4,recovery_new_benchmarks=2,additional_native_boundary_calls=4,additional_decoder_only_calls=2,numerical_qualification=q,new_fits=0,independent_metric_verification_pending=True,local_payload_sync=False)
 dump('RESULTS_V1.json',result);print(json.dumps(dict(status=result['status'],actual_complete_evaluator_invocations=6,new_fits=0)),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
