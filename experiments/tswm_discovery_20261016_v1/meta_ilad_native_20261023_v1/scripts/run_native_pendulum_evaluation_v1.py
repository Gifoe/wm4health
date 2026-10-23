import pathlib,sys,json,datetime,subprocess,hashlib,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'pendulum_native_recipe_v1';OUT.mkdir(exist_ok=False)
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 data=json.loads((ROOT/'pendulum_data_generation_v1/RESULTS_V1.json').read_text());assert data['total_sequences']==19000 and all(x['functional_sanity_passed'] for x in data['sets'])
 dump('PLAN_V1.json',dict(protocol_sha256=hashlib.sha256((ROOT/'PENDULUM_NATIVE_EVALUATION_PROTOCOL_V1.md').read_bytes()).hexdigest(),cases=[v+'_'+s for v in ['paper','refined'] for s in ['test','ood','trainenv']],source_revision=data['source_revision'],new_fits=0))
 cells=[]
 for variant in ['paper','refined']:
  for split in ['test','ood','trainenv']:
   print(json.dumps(dict(stage='case_started',variant=variant,split=split,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
   cp=subprocess.run([sys.executable,'-I','-B','-X','utf8','-u',str(ROOT/'scripts/native_pendulum_eval_worker_v1.py'),'--variant',variant,'--split',split],cwd=ROOT)
   assert cp.returncode==0,(variant,split,cp.returncode)
   cells.append(json.loads((OUT/(variant+'_'+split)/'RESULTS_V1.json').read_text()))
 result=dict(status='SIX_COMPLETE_NATIVE_PHYSICAL_EVALUATOR_INVOCATIONS_FINISHED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cells=cells,complete_evaluator_invocations=6,additional_native_boundary_calls=2,source_recipe_versions=['paper_version','refined'],trained_checkpoints=1,new_fits=0,independent_metric_verification_pending=True,local_payload_sync=False)
 dump('RESULTS_V1.json',result);print(json.dumps(dict(status=result['status'],complete_evaluator_invocations=6,new_fits=0)),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
