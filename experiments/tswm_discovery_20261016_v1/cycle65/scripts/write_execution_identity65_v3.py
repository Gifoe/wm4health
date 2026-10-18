from common65 import *
assert read(OUT/'recovery_state65_v1.json')['status']=='TRAIN_EVALUATE_REPLAY_SUMMARY_TERMINAL_SUCCESS'
assert read(OUT/'INDEPENDENT_REPLAY65_V1.json')['status']=='PASS'
files=['capture_evaluation_resume65_v1.py','finish_npz65_v1.py','recover_campaign65_v2.py','recover_cell65_v2.py','recover_validate65_v2.py','checkpoint_io65_v1.py','evaluate65_v3.py','replay65_v2.py','summarize65_v1.py','isolated_entry65_v1.py','close65_v3.py']
dump(OUT/'EXECUTION_IDENTITY65_V3.json',dict(utc=utc(),final_evaluation_replay_coordinator='capture_evaluation_resume65_v1.py',final_training_coordinator='finish_npz65_v1.py',prior_training_coordinator='recover_campaign65_v2.py',source_sha256={n:sha(ROOT/'scripts'/n) for n in files},supersedes='V1/V2 attribution of final successful evaluation coordinator',sealed_models=18,final_laws=240,independent_replay='PASS',original_evaluation_exit_code=2147483651,hash_verified_laws_reused=read(OUT/'EVALUATION_RESUME65_V1.json')['verified_laws'],scientific_settings_changed_on_resume=False,native_failure_root_cause='unproven'))
print(json.dumps(dict(status='EXECUTION_IDENTITY_V3_RECORDED')))
