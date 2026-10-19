"""Bounded driver; logs every worker and stops on engineering failures."""
import json,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    plan_name='lirpa_plan_v2.json';plan=json.loads((ROOT/'outputs'/plan_name).read_text());out=ROOT/'outputs/lirpa';out.mkdir(exist_ok=True)
    python=ROOT/'.venv_verifier/Scripts/python.exe';records=[]
    for H in plan['confirmation_horizons']:
      for rep in plan['confirmation_reps']:
       for query in plan['confirmation_queries']:
        path=out/f'rep{rep}_q{query}_H{H}.json';log=path.with_suffix('.log')
        if path.exists():records.append({'rep':rep,'query':query,'horizon':H,'status':'verified_existing'});continue
        start=time.perf_counter()
        with log.open('w',encoding='utf-8') as stream:
          try:
            r=subprocess.run([str(python),'-X','faulthandler','-u',str(ROOT/'scripts/probe_lirpa.py'),'--rep',str(rep),'--query',str(query),'--horizon',str(H),'--plan',plan_name],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=plan['worker_timeout_seconds'])
            status='complete' if r.returncode==0 and path.exists() else 'engineering_failure'
            row={'rep':rep,'query':query,'horizon':H,'status':status,'exit_code':r.returncode,'seconds':time.perf_counter()-start}
          except subprocess.TimeoutExpired:
            row={'rep':rep,'query':query,'horizon':H,'status':'timeout','seconds':time.perf_counter()-start}
        records.append(row);(out/'registry.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
        if row['status']!='complete':
            print('STOP_FOR_DIAGNOSIS: runtime/unsupported failure is NOT scientific evidence',flush=True);return
    (out/'registry.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print('LIRPA_CONFIRMATION_COMPLETE',flush=True)
if __name__=='__main__':main()
