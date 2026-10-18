from common87_v1 import *
def main():
 checks=[]
 for stem in ['check_rollout87_v1','check_rollout87_v2','check_rollout87_v3']:
  exit_path=O/stem/'EXIT.json';receipt=read(exit_path);assert receipt['exit_code']!=0;log_path=O/stem/'worker.log';checks.append(dict(stem=stem,exit_receipt=receipt,exit_sha256=sha(exit_path),log_path=str(log_path),log_sha256=sha(log_path),log_bytes=log_path.stat().st_size))
 write(O/'NUMERICAL_QUALIFICATION87_V1.json',dict(utc=utc(),status='INDEPENDENT_LEARNED_ROLLOUT_QUALIFICATION_NOT_PASSED',criterion_not_relaxed=True,previously_failed_relative_errors=[.021744366934861474,.015544012911164733],threshold=.01,checks=checks,original_batch_reference_verified=True,batch_precision_receipt_sha256=sha(O/'BATCH_PRECISION87_V2.json'),model_weights_and_selected_settings_unchanged=True,primary_adapted_model_supplemental_checks_not_reached=True,full_population_continuous_time_error_certified=False,note_sha256=sha(C/'NUMERICAL_OUTCOME87_V1.md')))
if __name__=='__main__':main()
