"""Durable verification queue: observe the actual fit; never restart/kill training."""
import pathlib,sys,os,json,datetime,subprocess,hashlib,time,ctypes
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'postfit_queue_recovery_v1';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def event(stage,**kw):
 if 'utc' in kw:kw['source_utc']=kw.pop('utc')
 x=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**kw)
 with (OUT/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
 print(json.dumps(x),flush=True)
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def live(pid):
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.restype=ctypes.c_void_p;handle=k.OpenProcess(0x1000,False,int(pid))
 if not handle:return False
 code=ctypes.c_ulong();ok=k.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code));k.CloseHandle(ctypes.c_void_p(handle));return bool(ok and code.value==259)
def run(name,python,script,args=(),timeout=None):
 folder=OUT/name;folder.mkdir(exist_ok=False);log=folder/'stdout.log'
 with log.open('xb') as f:
  child=subprocess.Popen([str(python),'-I','-B','-X','faulthandler','-u',str(script),*map(str,args)],cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
  with (folder/'launch.json').open('x') as g:json.dump(dict(pid=child.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_seconds=timeout),g)
  expired=False
  try:code=child.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   expired=True;subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=f,stderr=subprocess.STDOUT,check=False);code=child.wait()
 receipt=dict(pid=child.pid,exit_code=code,timeout=expired,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=sha(log))
 with (folder/'exit.json').open('x') as g:json.dump(receipt,g,indent=2)
 event('verification_child_terminal',name=name,**receipt);return receipt
try:
 launch=read(ROOT/'factual_execution_recovery_v1/launch.json');pid=launch['pid'];receipt=ROOT/'factual_execution_recovery_v1/exit.json';event('observe_specific_factual_child',pid=pid,live_now=live(pid),restarts_permitted=False)
 while not receipt.exists():
  active=live(pid)
  if not active:
   # Parent writes the terminal receipt after waiting for this exact child.
   event('factual_child_missing_wait_for_capture_receipt',pid=pid)
   for _ in range(6):
    if receipt.exists():break
    time.sleep(5)
   if not receipt.exists():raise RuntimeError('Actual child is missing and capture terminal receipt is absent; no restart authorized by this queue.')
   break
  time.sleep(15)
 terminal=read(receipt);event('actual_factual_child_terminal',**terminal)
 if terminal['exit_code']!=0:
  dump('RESULTS_V1.json',dict(status='FACTUAL_RECOVERY_TERMINAL_INCOMPLETE_NO_AUTORESTART',fit_terminal=terminal,full_research_goal_achieved=False));sys.exit(0)
 result=read(ROOT/'factual_fit_recovery_v1/RESULTS_V1.json');assert result['optimizer_updates']==2000 and result['total_actual_optimizer_calls']==2500
 root_python=ROOT.parent/'.venv/Scripts/python.exe';cpu_python=ROOT/'.venv_editor_cpu/Scripts/python.exe';executions=[]
 for step in [0,500,1000,2000]:
  verified=ROOT/'factual_metric_verification_recovery_v1'/f'VERIFIED_STEP{step:04d}_V1.json'
  if verified.exists():
   v=read(verified);row=ROOT/'factual_fit_recovery_v1'/f'VALIDATION_STEP{step:04d}_V1.json';assert v['receipt_sha256']==sha(row);event('reuse_exact_metric_verification',step=step,receipt_sha256=sha(verified))
  else:
   r=run(f'metrics_step{step:04d}',root_python,ROOT/'scripts/verify_factual_metrics_v2.py',['--step',step]);executions.append(r);assert r['exit_code']==0
 for step in [500,1000,2000]:
  r=run(f'projection_step{step:04d}',root_python,ROOT/'scripts/compare_crossmodal_checkpoints_v1.py',['--step',step]);executions.append(r);assert r['exit_code']==0
 chosen=int(result['selected']['step']);r=run(f'clock_step{chosen:04d}',cpu_python,ROOT/'scripts/qualify_factual_clock_v3.py',['--step',chosen],timeout=600);executions.append(r)
 clock_path=ROOT/f'factual_clock_step{chosen:04d}_diagnostic_v1/RESULTS_V1.json';clock=read(clock_path) if clock_path.exists() else None
 qualified=bool(r['exit_code']==0 and not r['timeout'] and clock and clock['status']=='NATIVE_FILTER_AND_IMAGINATION_CLOCK_QUALIFIED')
 dump('RESULTS_V1.json',dict(status='POSTFIT_VERIFICATION_CLOSED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),factual_selected_step=chosen,competence_gate=result['competence_gate'],clock_qualified=qualified,clock_result=clock,children=executions,fit_terminal=terminal,mechanism_experiments_allowed_by_this_queue=False,physical_parameter_edit_evaluated=False,accepted_final_directions=0,full_research_goal_achieved=False));event('postfit_verification_closed',selected_step=chosen,competence_gate=result['competence_gate'],clock_qualified=qualified)
except Exception as e:
 import traceback
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc(),no_training_restarted=True));raise
