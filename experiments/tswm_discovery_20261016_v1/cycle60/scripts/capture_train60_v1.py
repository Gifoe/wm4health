"""Execute all six predeclared attempts; keep instability and ordinary errors distinct."""
from common60 import *
import subprocess,sys,traceback
statefile=OUT/'training_state60_v1.json';completed=[];failed=[]
def state(status,**kw):
    p=statefile.with_suffix('.tmp');p.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,'completed':completed,'failed':failed,**kw},indent=2));p.replace(statefile)
try:
    assert read(OUT/'preflight_state60_v2.json')['status']=='DATA_QUALIFIED_AND_SYNTHETIC_DEVICE_SELECTION_COMPLETE'
    q=read(OUT/'QUALIFICATION60_V1.json');assert q['chosen_device']=='cpu'
    jobs=[(system,seed) for system in ['N87','pick_place'] for seed in [2021,2022,2023]]
    dump(OUT/'TRAINING_PLAN60_V1.json',{'utc':utc(),'jobs':jobs,'updates_per_fit':5000,'device':q['chosen_device'],'device_qualification_sha256':sha(OUT/'QUALIFICATION60_V1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL60_V1.md'),'data_manifest_sha256':sha(OUT/'DATA60_V2.json'),'instability_rule':'Stop and retain a nonfinite training-loss fit; continue other predeclared seeds without retry. Any other implementation error stops the controller for inspection.','new_neural_test_scores_computed':False})
    for system,seed in jobs:
        tag=f'{system}_seed{seed}'
        with (OUT/(tag+'.stdout.log')).open('xb') as out,(OUT/(tag+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/train60_v1.py'),system,str(seed)],stdout=out,stderr=err,cwd=ROOT)
            state('RUNNING',active={'tag':tag,'child_pid':child.pid});code=child.wait()
        rec=OUT/'runs'/tag/'result.json'
        assert rec.exists(),(tag,'No terminal fit receipt',code)
        r=read(rec)
        if code==3:
            assert r['status']=='FAILED_NONFINITE_TRAINING_LOSS_PRESERVED';failed.append({'tag':tag,'result_sha256':sha(rec),'status':r['status']})
        else:
            assert code==0 and r['status']=='COMPLETE_FIXED_BUDGET_NSSM_VALIDATION_SELECTED',(tag,code,r['status'])
            assert sha(r['selected']['checkpoint'])==r['selected']['checkpoint_sha256'];completed.append({'tag':tag,'result_sha256':sha(rec),'selected':r['selected']})
        state('RUNNING',last_completed=tag)
    dump(OUT/'MODEL_SEAL60_V1.json',{'utc':utc(),'status':'ALL_SIX_PREDECLARED_ATTEMPTS_TERMINAL_SELECTED_MODELS_SEALED','models':completed,'failed':failed,'training_plan_sha256':sha(OUT/'TRAINING_PLAN60_V1.json'),'test_scored':False,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0})
    state('ALL_SIX_ATTEMPTS_TERMINAL_MODEL_SEAL_READY_FOR_SCORING')
except Exception:
    state('FAILED_IMPLEMENTATION_PRESERVED',traceback=traceback.format_exc());raise
