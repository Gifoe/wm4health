"""Retain prior valid fits; execute the explicitly centered existing controls."""
from common57 import *
import subprocess,traceback
FINAL_METHODS=['native','independent','running09c','running01c','full','proper']
state_path=OUT/'training_state57_v2.json';completed=[]
def state(status,**kw):
    obj={'utc':utc(),'pid':os.getpid(),'status':status,'completed':completed,**kw}
    tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(state_path)
try:
    failure=read(OUT/'FAILED_ATTEMPT57_V1.json');assert failure['heldout_open_receipt_absent']
    assert not (OUT/'HELDOUT_OPEN_RECEIPT57_V1.json').exists()
    q=read(OUT/'CENTERED_QUALIFICATION57_V1.json');cq=read(OUT/'CENTERED_CUDA_QUALIFICATION57_V1.json')
    assert q['status']=='PASS_ANALYTIC_POSITIVE_RELATIVE_KERNEL_AND_GAMMA1_GRADIENT'
    assert cq['status']=='PASS_CENTERED_CUDA_NATIVE_LOSS_AND_GRADIENT'
    assert q['relative_math_sha256']==cq['relative_math_sha256']==sha(ROOT/'scripts/relative57_v1.py')
    jobs=[]
    for sensor in SENSORS:
        jobs.append({'sensor':sensor,'method':'proper','seed':2021})
        for seed in SEEDS:
            for method in FINAL_METHODS[:-1]:jobs.append({'sensor':sensor,'method':method,'seed':seed})
    assert len(jobs)==32
    prior={x['tag']:x for x in failure['valid_previous_fits']}
    plan={'utc':utc(),'status':'FIXED_CENTERED_CONTROLS_BEFORE_ANY_HELDOUT','jobs':jobs,'reuse_only_validated_terminal_tags':list(prior),'retained_failed_updates':10000,'neural_updates_per_fit':STEPS,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'clarification_sha256':sha(ROOT/'EXECUTION_CLARIFICATION_V1.md'),'normalizer_clarification_sha256':sha(ROOT/'NORMALIZER_CLARIFICATION_V2.md'),'decision_rule_sha256':sha(ROOT/'DECISION_RULE_V1.md'),'train_script_sha256':sha(ROOT/'scripts/train57_v2.py'),'core_math_sha256':sha(ROOT/'scripts/math57.py'),'relative_math_sha256':sha(ROOT/'scripts/relative57_v1.py'),'model_sha256':sha(ROOT/'scripts/models57.py'),'qualification_sha256':sha(OUT/'QUALIFICATION57_V1.json'),'centered_qualification_sha256':sha(OUT/'CENTERED_QUALIFICATION57_V1.json'),'centered_cuda_qualification_sha256':sha(OUT/'CENTERED_CUDA_QUALIFICATION57_V1.json'),'data_manifest_sha256':sha(OUT/'DATA57_V1.json'),'heldout_opened':False}
    dump(OUT/'TRAINING_PLAN57_V2.json',plan)
    for job in jobs:
        tag=f"{job['method']}_{job['sensor']}_seed{job['seed']}"
        if tag in prior:
            old=prior[tag];assert sha(old['checkpoint'])==old['checkpoint_sha256']
            assert sha(OUT/'runs'/tag/'result.json')==old['result_sha256']
            result=read(OUT/'runs'/tag/'result.json');assert result['status']=='COMPLETE_FIXED_TERMINAL_FIT'
            completed.append({**old,'reused_verified_terminal':True});state('RUNNING',last_completed=tag);continue
        with (OUT/(tag+'.stdout.log')).open('xb') as out,(OUT/(tag+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/train57_v2.py'),job['sensor'],job['method'],str(job['seed'])],stdout=out,stderr=err,cwd=ROOT)
            state('RUNNING',active={'tag':tag,'pid':child.pid});code=child.wait()
        path=OUT/'runs'/tag/'result.json'
        if code or not path.exists():
            state('FAILED_FIT_PRESERVED',failed_tag=tag,exit_code=code);sys.exit(1)
        result=read(path)
        assert result['status']=='COMPLETE_FIXED_TERMINAL_FIT' and sha(result['checkpoint'])==result['checkpoint_sha256']
        completed.append({'tag':tag,'result_sha256':sha(path),'checkpoint':result['checkpoint'],'checkpoint_sha256':result['checkpoint_sha256'],'reused_verified_terminal':False})
        state('RUNNING',last_completed=tag)
    for sensor in SENSORS:
        for seed in SEEDS:
            digests={read(OUT/'runs'/f'{method}_{sensor}_seed{seed}'/'result.json')['initial_tensor_sha256'] for method in FINAL_METHODS[:-1]}
            assert len(digests)==1,'Matched initialization violated'
    seal={'utc':utc(),'status':'ALL_32_TERMINAL_MODELS_SEALED_HELDOUT_UNOPENED','models':completed,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'plan_sha256':sha(OUT/'TRAINING_PLAN57_V2.json'),'normalizer_clarification_sha256':sha(ROOT/'NORMALIZER_CLARIFICATION_V2.md'),'failed_attempt_sha256':sha(OUT/'FAILED_ATTEMPT57_V1.json'),'retained_failed_updates':10000,'heldout_opened':False,'source_files_sha256':source_check()}
    dump(OUT/'MODEL_SEAL57_V1.json',seal)
    state('ALL_32_FIXED_TERMINAL_FITS_COMPLETE_HELDOUT_UNOPENED')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
