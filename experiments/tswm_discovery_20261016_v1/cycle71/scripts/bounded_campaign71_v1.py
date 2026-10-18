"""Qualify segmented native execution, then close the original three-fit budget."""
import sys,pathlib,copy
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from bounded_capture71_v1 import *

def qualify():
    assert read(OUT/'RUNTIME_REPLAY_QUALIFICATION71_V2.json')['status']=='BOUNDED_RUNTIME_REPLAY_AND_AFFINITY_NUMERICAL_EQUIVALENCE_PASS'
    original=read(OUT/'resume_full_control71_v1_result.json');full=read(OUT/'resume_full_control71_v1_checkpoints.json')
    prefix=[r for r in full['checkpoints'] if r['epoch']==0];assert len(prefix)==1
    p=OUT/'bounded_qual71_v1_initial_manifest.json';dump(p,dict(checkpoints=prefix))
    # The exact initial temperature is already qualified by the original continuation.
    from resume_native71_v2 import temperature_from_log,nested_compare,torch
    temperature=temperature_from_log(2026731,'resume_full_control71_v1',0)
    common=dict(qualification=True,seed=2026731,total_epochs=3,train_tasks=128)
    a=run_request(dict(common,tag='bounded_qual71_v1_part1',stop=2,parent_manifest=str(p),parent_checkpoint_sha256=prefix[-1]['sha256'],temperature_float64_hex=temperature.hex()))
    b=run_request(dict(common,tag='bounded_qual71_v1_part2',stop=3,parent_manifest=a['checkpoint_manifest'],parent_checkpoint_sha256=a['last']['sha256'],temperature_float64_hex=a['temperature_float64_hex']))
    reference=torch.load(original['last']['path'],map_location='cpu',weights_only=False)
    candidate=torch.load(b['last']['path'],map_location='cpu',weights_only=False)
    keys=['module_state','proposal_state','module_optimizer','proposal_optimizer','scheduler','torch_rng','cuda_rng','numpy_rng','python_rng','train_shuffle','val_shuffle','train_structures','val_structures','SA_running_acc_rate','SA_running_factor','metrics','module_updates','proposal_updates']
    comp=nested_compare({k:reference[k] for k in keys},{k:candidate[k] for k in keys})
    assert comp['max_abs']==0 and comp['non_tensor_mismatches']==0,comp
    obj=dict(utc=utc(),status='PASS_BOUNDED_PROCESS_CONTINUATION_EXACT_EQUIVALENCE',comparison=comp,keys_compared=keys,
        reference_checkpoint=original['last'],bounded_checkpoint=b['last'],
        qualification_segments=[a['tag'],b['tag']],entry_adapters=[a['entry_adapter'],b['entry_adapter']],
        temperature_handoff_exact=True,affinity=b['affinity'],source_optimization_steps_retained=True,
        qualification_only=True,performance_evidence=False,root_cause_verified=False,long_run_stability_verified=False,
        reserved_eval_rows_accessed=0,accepted_final_directions=0)
    dump(OUT/'BOUNDED_CONTINUATION_QUALIFICATION71_V1.json',obj)
    print(json.dumps({k:obj[k] for k in ['status','comparison','temperature_handoff_exact']}),flush=True)

def formal():
    assert read(OUT/'BOUNDED_CONTINUATION_QUALIFICATION71_V1.json')['status']=='PASS_BOUNDED_PROCESS_CONTINUATION_EXACT_EQUIVALENCE'
    incident=read(OUT/'RUNTIME_INCIDENT71_V2.json');fits=[]
    for seed in (2026711,2026712,2026713):
        segments=[]
        if seed==2026711:
            parent=str(OUT/'formal_seed2026711_resumed_v2_checkpoints.json');checkpoint=incident['last_checkpoint'];temperature=incident['annealing_temperature_exact_replay']['float64_hex'];start=801
        else:
            assert not (OUT/('formal_seed'+str(seed)+'_v3_launch.json')).exists()
            parent=None;checkpoint=None;temperature=None;start=0
        while start<2000:
            stop=min((start//50+1)*50+1,2000)
            tag='formal_seed'+str(seed)+'_bounded'+str(stop-1).zfill(5)+'_v1'
            request=dict(qualification=False,seed=seed,total_epochs=2000,train_tasks=2048,stop=stop,tag=tag,
                parent_manifest=parent,parent_checkpoint_sha256=checkpoint['sha256'] if checkpoint else None,temperature_float64_hex=temperature)
            r=run_request(request);segments.append(dict(tag=tag,start_epoch=r['start_epoch'],last_epoch=r['last_epoch'],exit_file=str(OUT/(tag+'_exit.json')),result_file=str(OUT/(tag+'_segment_result.json'))))
            parent=r['checkpoint_manifest'];checkpoint=r['last'];temperature=r['temperature_float64_hex'];start=r['last_epoch']+1
            dump(OUT/'BOUNDED_CAMPAIGN71_V1_progress.json',dict(utc=utc(),status='ORIGINAL_BUDGET_RUNNING_IN_BOUNDED_PROCESSES',current_seed=seed,last_completed_epoch=start-1,effective_updates_current_seed=r['effective_module_updates'],closed_fits=len(fits),segments_current_fit=len(segments),reserved_eval_rows_accessed=0),exclusive=False)
        manifest=read(parent);assert [c['epoch'] for c in manifest['checkpoints']]==list(range(0,2000,50))+[1999]
        assert r['effective_module_updates']==r['effective_proposal_updates']==32000
        result=dict(utc=utc(),status='NATIVE_SUBSET_FIT_COMPLETE_AFTER_BOUNDED_PROCESS_CONTINUATION',seed=seed,epochs=2000,
            train_tasks=2048,val_tasks=256,module_updates=32000,proposal_updates=32000,
            selected=manifest['selected'],last=manifest['checkpoints'][-1],checkpoints=manifest['checkpoints'],segments=segments,
            module_parameters=r['module_parameters'],proposal_parameters=r['proposal_parameters'],
            original_protocol_sha256=sha(ROOT/'PROTOCOL71_V1.md'),runtime_protocol_sha256=sha(ROOT/'BOUNDED_RUNTIME_PROTOCOL71_V1.md'),
            qualification_sha256=sha(OUT/'BOUNDED_CONTINUATION_QUALIFICATION71_V1.json'),
            discarded_suffix_and_diagnostic_compute_excluded_from_effective_budget=True,
            is_original_full_paper_reproduction=False,new_method=False,reserved_eval_rows_accessed=0,accepted_final_directions=0)
        name='formal_seed'+str(seed)+'_bounded_v1_result.json';dump(OUT/name,result)
        fits.append(dict(seed=seed,result_file=str(OUT/name),result_sha256=sha(OUT/name),selected=result['selected'],last=result['last'],module_updates=32000,proposal_updates=32000))
    obj=dict(utc=utc(),status='THREE_NATIVE_SUBSET_FITS_CLOSED_AFTER_BOUNDED_CONTINUATION_TRAIN_VAL_ONLY',fits=fits,
        total_module_updates=96000,total_proposal_updates=96000,effective_lineage_only=True,
        reserved_eval_rows_accessed=0,root_cause_verified=False,new_method=False,accepted_final_directions=0)
    dump(OUT/'BOUNDED_CAMPAIGN71_V1.json',obj);print(json.dumps(obj),flush=True)

if __name__=='__main__':
    assert sys.argv[1] in ('qualify','formal')
    qualify() if sys.argv[1]=='qualify' else formal()
