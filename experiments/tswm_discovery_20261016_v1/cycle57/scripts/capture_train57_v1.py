"""Single GPU owner; seal every terminal model before downstream heldout work."""
from common57 import *
import subprocess,traceback
state_path=OUT/'training_state57_v1.json';completed=[]
def state(status,**kw):
    state_path.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,'completed':completed,**kw},indent=2),encoding='utf-8')
try:
    qual=read(OUT/'QUALIFICATION57_V1.json');assert qual['status']=='PASS_NATIVE_LOSS_MOMENT_GRADIENT_CUDA_AND_PROPER_LIKELIHOOD'
    assert read(OUT/'preflight_state57_v1.json')['status']=='PREFLIGHT_AND_DATA_COMPLETE_ZERO_FORMAL_UPDATES'
    jobs=[]
    for sensor in SENSORS:
        jobs.append({'sensor':sensor,'method':'proper','seed':2021})
        for seed in SEEDS:
            for method in METHODS[:-1]:jobs.append({'sensor':sensor,'method':method,'seed':seed})
    assert len(jobs)==32
    plan={'utc':utc(),'status':'FIXED_BEFORE_FORMAL_FITS','jobs':jobs,'neural_updates_per_fit':STEPS,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'clarification_sha256':sha(ROOT/'EXECUTION_CLARIFICATION_V1.md'),'train_script_sha256':sha(ROOT/'scripts/train57_v1.py'),'math_sha256':sha(ROOT/'scripts/math57.py'),'model_sha256':sha(ROOT/'scripts/models57.py'),'qualification_sha256':sha(OUT/'QUALIFICATION57_V1.json'),'data_manifest_sha256':sha(OUT/'DATA57_V1.json'),'heldout_opened':False}
    dump(OUT/'TRAINING_PLAN57_V1.json',plan)
    for job in jobs:
        tag=f"{job['method']}_{job['sensor']}_seed{job['seed']}"
        with (OUT/(tag+'.stdout.log')).open('xb') as stdout,(OUT/(tag+'.stderr.log')).open('xb') as stderr:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/train57_v1.py'),job['sensor'],job['method'],str(job['seed'])],stdout=stdout,stderr=stderr,cwd=ROOT)
            state('RUNNING',active={'tag':tag,'pid':child.pid})
            code=child.wait()
        result=read(OUT/'runs'/tag/'result.json') if (OUT/'runs'/tag/'result.json').exists() else None
        if code or result is None:
            state('FAILED_FIT_PRESERVED',failed_tag=tag,exit_code=code);sys.exit(1)
        assert result['status']=='COMPLETE_FIXED_TERMINAL_FIT' and sha(result['checkpoint'])==result['checkpoint_sha256']
        completed.append({'tag':tag,'result_sha256':sha(OUT/'runs'/tag/'result.json'),'checkpoint':result['checkpoint'],'checkpoint_sha256':result['checkpoint_sha256']})
        state('RUNNING',last_completed=tag)
    for sensor in SENSORS:
        for seed in SEEDS:
            digests={read(OUT/'runs'/f'{method}_{sensor}_seed{seed}'/'result.json')['initial_tensor_sha256'] for method in METHODS[:-1]}
            assert len(digests)==1,'Neural initialized tensors differ'
    seal={'utc':utc(),'status':'ALL_32_TERMINAL_MODELS_SEALED_HELDOUT_UNOPENED','models':completed,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'plan_sha256':sha(OUT/'TRAINING_PLAN57_V1.json'),'heldout_opened':False,'source_files_sha256':source_check()}
    dump(OUT/'MODEL_SEAL57_V1.json',seal)
    state('ALL_32_FIXED_TERMINAL_FITS_COMPLETE_HELDOUT_UNOPENED')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
