from common62 import *
c=read(OUT/'campaign_state62_v3.json');assert c['status']=='FAILED_INSPECT_EVIDENCE' and len(c['completed'])==8
eq=read(OUT/'PRO_DYN_EQUIVALENCE62_V1.json');assert eq['status']=='EXACT_EQUIVALENCE_AT_L_EQUALS_H'
fits=[read(OUT/(key+'_fit62_v1.json')) for key in c['completed']]
assert set(c['completed'])=={a+'_'+str(s) for s in [2021,2022,2023] for a in ['Informer','Informer_PRO','Informer_DYN']} - {'Informer_DYN_2023'}
for r in fits:assert sha(r['selected_checkpoint'])==r['selected_sha256'] and sha(r['terminal_checkpoint'])==r['terminal_sha256']
dump(OUT/'MODEL_SEAL62_V2.json',{'utc':utc(),'status':'EIGHT_SEALED_FITS_PARTIAL_REPRODUCTION','completed':c['completed'],'completed_count':8,'planned_count':9,'failed_formal_attempts':3,'formal_attempts_total':11,'known_aborted_updates_lower_bound':3200,'missing_fit':'Informer_DYN_2023','distinct_modified_architectures_at_this_L_H':1,'canonical_three_seed_modified_arm':'Informer_PRO','auxiliary_duplicate_arm':'Informer_DYN','failed_native_root_cause':'unknown; cached loading does not eliminate all faults','test_opened':False,'equivalence_sha256':sha(OUT/'PRO_DYN_EQUIVALENCE62_V1.json'),'amendment_sha256':sha(ROOT/'PARTIAL_CLOSURE_AMENDMENT62_V1.md')})
print(json.dumps({'status':'EIGHT_SEALED_FITS_PARTIAL_REPRODUCTION','completed':8,'planned':9,'failed':3}))
