"""Publish closed runtime evidence, truthful live dependencies and amended execution."""
import sys,pathlib,base64,math,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
incident=read(OUT/'RUNTIME_INCIDENT71_V2.json')
q=read(OUT/'BOUNDED_CONTINUATION_QUALIFICATION71_V1.json')
rq=read(OUT/'RUNTIME_REPLAY_QUALIFICATION71_V2.json')
assert q['status']=='PASS_BOUNDED_PROCESS_CONTINUATION_EXACT_EQUIVALENCE'
assert rq['status']=='BOUNDED_RUNTIME_REPLAY_AND_AFFINITY_NUMERICAL_EQUIVALENCE_PASS'
assert q['comparison']['max_abs']==rq['comparison']['max_abs']==0
assert q['comparison']['non_tensor_mismatches']==rq['comparison']['non_tensor_mismatches']==0
assert read(OUT/'bounded_campaign71_v1_qualify_exit.json')['exit_code']==0
assert read(OUT/'recovery71_v2_formal_exit.json')['exit_code']==3221225477
assert read(OUT/'campaign71_v4_exit.json')['exit_code']==read(OUT/'queue_free_simulation71_v2_exit.json')['exit_code']==1
assert sha(incident['last_checkpoint']['path'])==incident['last_checkpoint']['sha256']
for version in (1,2,3):assert not (OUT/('free_simulation71_v'+str(version)+'_launch.json')).exists()
procs=json.loads(os.environ['C71_LIVE_PROCESSES_JSON']);procs=procs if isinstance(procs,list) else [procs]
controllers=[p for p in procs if '\\scripts\\bounded_campaign71_v1.py formal' in p['CommandLine']]
workers=[p for p in procs if '\\scripts\\bounded_native71_v1.py' in p['CommandLine']]
queues=[p for p in procs if '\\scripts\\queue_free_simulation71_v3.py' in p['CommandLine']]
assert len(controllers)==len(workers)==len(queues)==1
assert controllers[0]['ProcessId']==read(OUT/'bounded_campaign71_v1_formal_launch.json')['pid']
assert queues[0]['ProcessId']==read(OUT/'queue_free_simulation71_v3_launch.json')['pid']
launches=[]
for p in OUT.glob('formal_seed*_bounded*_v1_launch.json'):
    r=read(p)
    if r['pid']==workers[0]['ProcessId']:launches.append((p,r))
assert len(launches)==1
worker_launch=launches[0][0];tag=worker_launch.name[:-len('_launch.json')]
for attempt in range(5):
    try:progress=read(OUT/(tag+'_progress.json'));break
    except (OSError,json.JSONDecodeError):
        if attempt==4:raise
        time.sleep(.2)
assert progress['epoch']>=801 and progress['reserved_eval_rows_accessed']==0
firmware=json.loads(os.environ['C71_FIRMWARE_JSON'])
revision=bytes(firmware['update_revision_bytes']);assert len(revision)==8
firmware['update_revision_upper32_hex']=hex(int.from_bytes(revision[4:8],'little'))
firmware['interpretation']='Read-only reported firmware context; not proof of a hardware cause. No firmware or system settings changed.'
dump(OUT/'FIRMWARE_CONTEXT71_V1.json',dict(utc=utc(),**firmware))

report='''# Second native runtime incident and qualified bounded continuation

The first recovered formal process terminated with Windows access violation0xc0000005 after the fully logged epoch800. Its captured exit, campaign exit and assessment queue exit are terminal; the old process/queue are no longer running. Windows Application Error identifies NumPy mtrand.cp312-win_amd64.pyd at the native np.random.choice proposal call. The earlier incident named nvcuda64.dll. Treating both as an established CUDA-driver-only problem is unsupported. Synchronous CUDA execution did not prevent recurrence. RUNTIME_INCIDENT71_V2.json records the exact second event, log hashes, epoch800 checkpoint and reconstructed binary64 temperature. Its174 CPU-loaded checkpoint tensors contain no nonfinite values. Neither later seed nor free-VAL assessment launched in the failed campaign.

## Bounded numerical evidence

The unrestricted and process-local efficient-CPU-class replays of epochs801-802 both closed successfully. Their complete compared states match across174 tensors with maximum absolute difference0 and zero non-tensor mismatches, including optimizer/scheduler states, RNG/shuffles, graphs, metrics and update counts. This does not reproduce a deterministic failure at epoch801; it does not establish long-run stability.

The new outer-range-bounded execution then passed a separate cross-process continuation comparison against the native uninterrupted three-epoch reference: again174 tensors, maximum absolute difference0 and zero non-tensor mismatches. The original epoch-loop AST body is unchanged. Exact binary64 annealing temperature is passed between processes; original optimization_steps remains fixed. Qualification-only extra snapshots and replay updates do not enter formal selection or effective training counts.

The first replay script incorrectly expected a native checkpoint at epoch802 even though the retained original optimization_steps=2000 does not request that save. It failed an explicit Python assertion after the two epochs, not a native access violation. Its terminal receipt/source/log are retained. The second version explicitly saves a separate diagnostic snapshot. The first SSH Start-Process attempt had no surviving worker or replay receipts when checked; the captured detached WMI launcher subsequently established actual worker identities. Source versions are preserved rather than overwritten.

## Execution amendment and live state

BOUNDED_RUNTIME_PROTOCOL71_V1.md retains all three original seeds,2000 epochs/32000 module and proposal updates per fit, original source bodies, data and checkpoint grid. New workers restrict only themselves to Windows' discovered lower-performance CPU-set class, retain four numerical threads and original CUDA precision, and stop at the next native50-epoch checkpoint or final1999. No failed worker is automatically retried. A fresh process restores the full qualified state to continue the effective lineage; physical computation includes discarded/replay work beyond that lineage.

BOUNDED_RUNTIME_SNAPSHOT71_V1.json is the bounded observation of the actual live controller, worker and new assessment queue. The new queue waits on the controller's Windows process handle/creation identity and requires all three successful terminal fit receipts. Forecast helper bodies and the frozen FREE_SIMULATION_PROTOCOL71_V1.md remain unchanged; only the terminal-campaign provenance gate and process-local execution control are adapted. Earlier failed queues remain failed historical artifacts. Current training is not complete.

## Read-only hardware context and uncertainty

The machine reports an Intel Core i9-14900KF and registry revision bytes interpreting to microcode0x12b; BIOS1805 is dated2024-10-30. See FIRMWARE_CONTEXT71_V1.json. Intel's primary announcement describes0x12F as an additional mitigation for conditions contributing to13th/14th-generation desktop Vmin instability, including long lightly threaded operation, and recommends current vendor BIOS plus Intel defaults. This makes firmware/CPU stability a plausible diagnostic hypothesis, not proof that this host has degradation or that it caused either fault. Zero WHEA events in the checked hour also do not rule out a hardware problem. No BIOS, driver, voltage/power, shared dependency or unrelated-process configuration was changed.

Primary sources: [Intel0x12F announcement](https://community.intel.com/t/intel-core-13th-and-14th-gen-vmin-shift-instabilty-update-new-microcode-update-0x12f/132113), [Microsoft CPU-set enumeration](https://learn.microsoft.com/en-us/windows/win32/procthread/getsystemcpusetinformation), [Microsoft CPU-set structure/efficiency classes](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information). Current limited evidence does not identify an exact NumPy defect either; no package downgrade was made speculatively.

The already closed structural-orbit finding remains: this split has no unseen five-node topology. Runtime repair, successful numerical qualification and baseline learning do not constitute a new world-model mechanism. No final research direction accepted; full objective ACTIVE/UNACHIEVED. Reserved EVAL stays closed. Raw arrays and weights stay on the server; GitHub receives selected authored code and compact closed evidence.
'''
updates={
 'literature_review.md':'Runtime diagnosis uses primary Intel/Microsoft documentation only as bounded engineering context. Two different native fault modules and old firmware context do not establish a hardware or software cause. No new scientific prior-art gap follows. See BOUNDED_RUNTIME_FINDING71_V1.md for the exact claim boundaries.',
 'baseline_registry.md':'The earlier recovered campaign/queue stopped after epoch800 with access violation; Windows names NumPy mtrand for this second incident. Both the two-epoch CPU-affinity comparison and the separate bounded-process continuation comparison match174 tensors and all compared states exactly. A new bounded-process controller and fail-closed assessment queue now carry the original three-fit budget. See the bounded snapshot for actual live identities; the baseline remains incomplete.',
 'experimental_results.md':'RUNTIME_INCIDENT71_V2.json closes the second terminal native failure at checkpoint800/12816 effective updates,174 finite checkpoint tensors. RUNTIME_REPLAY_QUALIFICATION71_V2.json and BOUNDED_CONTINUATION_QUALIFICATION71_V1.json each report174 tensors/max difference0/non-tensor mismatches0. These are numerical/execution checks, not forecast performance. The formal loop body,2000-epoch budget and checkpoint grid are retained under bounded worker lifetimes.',
 'negative_evidence.md':'CUDA_LAUNCH_BLOCKING=1 did not prevent a second access violation. The second Windows event names NumPy mtrand, whereas the first names nvcuda64; a driver-only diagnosis is unsupported. Epoch801 failure did not recur in the bounded replay, and affinity numerical equivalence does not prove long-run stability. The first replay diagnostic also failed a Python checkpoint-expectation assertion, corrected in a separately versioned script; retain it as an implementation failure, not a model deficit.',
 'hypothesis_space.md':'Runtime explanations remain competing: intermittent native software/runtime faults versus machine stability; neither a different fault module nor i9-14900KF/0x12b context proves causation. Successful bounded replay rejects only a deterministically failing replay at that checkpoint in the tested executions. Process-local CPU class plus bounded lifetime is a qualified execution control, not a research mechanism or a demonstrated root fix.',
 'research_ideas.md':'No scientific candidate is promoted by the runtime work. The next scientific discriminator still requires the complete frozen native free-simulation assessment. The new bounded controller preserves effective training and selection; source-body and numerical equivalence guard against converting an execution workaround into a different baseline. The structural-orbit finding continues to forbid unseen-topology claims from this split.',
 'final_research_direction.md':'No accepted final direction. The original recovered campaign failed again after epoch800; the exact failure and finite checkpoint are preserved. Two independent numerical continuation checks pass with174 tensor/max difference0 and no non-tensor differences. A qualified process-local, bounded-lifetime controller and new terminal-gated assessment queue now continue the unchanged original scientific budget. Live state is in BOUNDED_RUNTIME_SNAPSHOT71_V1.json; neither training nor the full research objective is complete.'}
heading='\n\n## Second runtime incident and bounded continuation, C71\n\n'
previous=OUT/'publication_structure_orbit_v1'/'files';planned={}
for name,body in updates.items():
    before=(previous/name).read_bytes();assert (ROOT/name).read_bytes()==before
    assert heading.encode() not in before
    planned[name]=(before,(heading+body+'\n').encode())
assert not (ROOT/'BOUNDED_RUNTIME_FINDING71_V1.md').exists()
assert not (OUT/'publication_bounded_runtime_v1').exists()
with (ROOT/'BOUNDED_RUNTIME_FINDING71_V1.md').open('x',encoding='utf-8') as f:f.write(report)
ledger_proofs={}
for name,(before,suffix) in planned.items():
    with (ROOT/name).open('ab') as f:f.write(suffix)
    after=(ROOT/name).read_bytes();assert after==before+suffix
    ledger_proofs[name]=dict(before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),original_bytes=len(before),prefix_preserved=True)
snapshot=dict(utc=utc(),status='BOUNDED_FORMAL_CAMPAIGN_AND_NEW_ASSESSMENT_QUEUE_RUNNING_NOT_COMPLETE',
    live_processes=procs,current_worker_launch=read(worker_launch),current_progress=progress,
    queue=read(OUT/'FREE_SIMULATION_QUEUE71_V3.json'),runtime_incident_sha256=sha(OUT/'RUNTIME_INCIDENT71_V2.json'),
    replay_qualification_sha256=sha(OUT/'RUNTIME_REPLAY_QUALIFICATION71_V2.json'),
    bounded_qualification_sha256=sha(OUT/'BOUNDED_CONTINUATION_QUALIFICATION71_V1.json'),
    ledgers=ledger_proofs,root_cause_verified=False,long_run_stability_verified=False,
    reserved_eval_rows_accessed=0,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(OUT/'BOUNDED_RUNTIME_SNAPSHOT71_V1.json',snapshot)
status='''# C71: second runtime fault preserved; qualified bounded execution now running

The previously recovered campaign stopped after epoch800. Windows names NumPy mtrand for the second fault, different from the earlier NVIDIA module; root cause remains unestablished. The finite checkpoint and terminal failure receipts are preserved. Successful two-epoch CPU-affinity and cross-process continuation comparisons each match174 tensors and all compared numerical states exactly.

The new bounded-process campaign continues the original2000 epochs/32000 updates per fit and unchanged checkpoint selection. Its source epoch-loop body is unchanged. New workers use qualified process-local CPU-class restriction and stop at original checkpoint boundaries; any failure stops without automatic retry. The newly captured assessment queue waits on this controller's actual Windows handle and will launch the unchanged frozen development forecast only after all three fits close. See cycle71/outputs/BOUNDED_RUNTIME_SNAPSHOT71_V1.json for the bounded live-state evidence, cycle71/BOUNDED_RUNTIME_PROTOCOL71_V1.md for execution fidelity, and cycle71/BOUNDED_RUNTIME_FINDING71_V1.md for diagnosis limits.

Training is incomplete; no final research direction accepted; full goal ACTIVE/UNACHIEVED. Reserved EVAL remains closed. The exact graph-orbit audit still prevents unseen-topology claims from this cohort. Raw arrays/weights remain on the server; GitHub receives selected code, reports and compact closed evidence. Old status files, failed campaigns and failed queues remain historical records.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V71_BOUNDED_RUNTIME.md').open('x',encoding='utf-8') as f:f.write(status)
closed=['RUNTIME_INCIDENT71_V2.json','RUNTIME_CRASH_EXCERPT71_V2.base64.txt','RUNTIME_REPLAY_QUALIFICATION71_V2.json','BOUNDED_CONTINUATION_QUALIFICATION71_V1.json','FIRMWARE_CONTEXT71_V1.json','BOUNDED_RUNTIME_SNAPSHOT71_V1.json','recovery71_v2_formal_exit.json','campaign71_v4_exit.json','queue_free_simulation71_v2_exit.json','runtime_replay71_v1_unrestricted_launch.json','runtime_replay71_v1_unrestricted_exit.json','runtime_replay71_v1_unrestricted_stdout.log','runtime_replay71_v2_unrestricted_launch.json','runtime_replay71_v2_unrestricted_exit.json','runtime_replay71_v2_unrestricted_result.json','runtime_replay71_v2_efficient_launch.json','runtime_replay71_v2_efficient_exit.json','runtime_replay71_v2_efficient_result.json','bounded_qual71_v1_part1_launch.json','bounded_qual71_v1_part1_exit.json','bounded_qual71_v1_part1_segment_result.json','bounded_qual71_v1_part2_launch.json','bounded_qual71_v1_part2_exit.json','bounded_qual71_v1_part2_segment_result.json','bounded_campaign71_v1_qualify_launch.json','bounded_campaign71_v1_qualify_exit.json','bounded_campaign71_v1_formal_launch.json','queue_free_simulation71_v3_launch.json','FREE_SIMULATION_QUEUE71_V3.json']
scripts=['audit_runtime_incident71_v2.py','runtime_affinity71_v1.py','replay_runtime71_v1.py','replay_runtime71_v2.py','capture_runtime_replay71_v1.py','capture_runtime_replay71_v2.py','bounded_native71_v1.py','bounded_capture71_v1.py','bounded_campaign71_v1.py','capture_bounded_campaign71_v1.py','evaluate_free_simulation71_v3.py','queue_free_simulation71_v3.py','capture_assessment71_v3.py','export_bounded_runtime71_v1.py']
starts=['start_runtime_replay71_v1.ps1','start_runtime_replay71_v2.ps1','start_runtime_replay71_v3.ps1','start_bounded_campaign71_v1.ps1','start_queue_assessment71_v3.ps1']
files=[ROOT/name for name in updates]+[ROOT/'BOUNDED_RUNTIME_FINDING71_V1.md',ROOT/'BOUNDED_RUNTIME_PROTOCOL71_V1.md']+[ROOT/'scripts'/name for name in scripts]+[ROOT/name for name in starts]+[OUT/name for name in closed]
dest=OUT/'publication_bounded_runtime_v1';dest.mkdir(exist_ok=False);patterns=json.loads(os.environ['C71_PRIVATE_EXPORT_PATTERNS_JSON']);rows=[]
for p in sorted(set(files)):
    raw=p.read_bytes();rel=p.relative_to(ROOT).as_posix()
    for pattern in patterns:assert pattern.encode() not in raw,(rel,'Private export pattern')
    try:body=raw.decode('utf-8-sig');encoding='UTF-8'
    except UnicodeDecodeError:body=base64.b64encode(raw).decode('ascii');rel+='.base64.txt';encoding='Base64 exact original bytes'
    b=body.encode('utf-8');target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
    rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),encoding=encoding,server_snapshot_file=str(target),parts=math.ceil(len(body)/3500)))
manifest=dict(utc=utc(),status=snapshot['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),
    raw_arrays_weights_and_live_training_logs_included=False,reserved_eval_rows_accessed=0,
    accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(dest/'manifest.json',manifest)
print(json.dumps(dict(status=manifest['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'),progress=progress)),flush=True)
