"""Dated live-progress reports; never mislabel a partially fitted grid complete."""
from native53 import *
import ast

OUT=R/'outputs/progress_snapshot_v1';assert not OUT.exists();OUT.mkdir()
for path in (R/'scripts').glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=path.name)
supervisor=json.loads((R/'outputs/supervisor_v6.json').read_text())
valid=[]
for folder in list(ART.glob('*'))+list((ART/'runtime_microbatch_v6').glob('*')):
    if not (folder/'summary.json').exists() or not (folder/'state.json').exists():continue
    state=json.loads((folder/'state.json').read_text())
    if state.get('status')!='COMPLETE':continue
    summary=json.loads((folder/'summary.json').read_text())
    if sha(summary['checkpoint'])!=summary['checkpoint_sha256']:raise AssertionError(folder)
    if state['tag']!='shared_codec':valid.append(summary)
assert len({s['tag'] for s in valid})==len(valid)
seen=now();fits=len(valid);updates=sum(s['formal_updates'] for s in valid)
progress={'utc':seen,'status':'LIVE_PROGRESS_NOT_FINAL','completed_predictive_fits':fits,'planned_predictive_fits':40,'completed_formal_updates':updates,'codec_complete':(ART/'shared_codec/summary.json').exists(),'supervisor_state':supervisor['status'],'observed_supervisor_pid':supervisor['pid'],'active_tag':(supervisor.get('active') or {}).get('tag'),'transfer_opened':(ART/'evaluation_v1/OPENING_RECEIPT.json').exists(),'accepted_final_directions':0,'completed_fit_identities':[{'tag':s['tag'],'checkpoint_sha256':s['checkpoint_sha256'],'best_epoch':s['best_epoch'],'epochs_run':s['epochs_run'],'runtime_storage_mode':s.get('runtime_storage_mode','native_full_batch')} for s in valid]}
dump(OUT/'progress.json',progress)
lead=f'''C53 snapshot UTC {seen}: **{fits}/40 predictive fits complete**, {updates} completed formal optimizer updates; shared native AE complete. This is a dated running snapshot, not a finished result table. Second-batch model evaluation opened: {progress['transfer_opened']}. Accepted final research directions: 0.

Two backbones (CrossLinear/PatchTST), four native design configurations and five seeds use the first original Greenhouse batch. Its815/163 train/VAL windows agree with the published163-window held-out specification. The second crop batch is a separate transfer audit after all checkpoints are sealed; it is not another independent real domain. The exact original paper's raw-byte run identity is not supplied, so headline numerical equality is not claimed.
'''
context='''The target remains a reusable time-series world-model mechanism. C49's conditional observation-averaged score is deprioritized; C52's interpolation fixture is inactive in1781 default real windows. Existing wm4health multi-timescale results are effectively tied with the original Direct-MH (300s BIS improvement .011, patient-bootstrap interval [-.037,.065]); the .064 capacity-control advantage and one-seed result do not establish a timescale mechanism. Its same-rate two-branch comparison was not run. Do not relaunch that architecture from this evidence alone.
'''
reports={
 'literature_review.md':'# C53 literature and prior-result boundary\n\n'+lead+'\n'+context+'\nSources: [TSWM paper](https://arxiv.org/abs/2610.01842), [pinned code](https://github.com/HaochenZhang717/TSWM-Official-Repo/tree/2e5e48540ff0da48e8192163969a6c58df1d5a3e), [original data](https://doi.org/10.4121/21960932.v1), [existing multi-timescale result](https://github.com/Gifoe/wm4health/blob/a24a6c54bf70b72f4326ffc0fd45bf19ba842ba8/experiments/vitaldb_multitimescale_dynamics_seed0_v1/outputs/FINAL_REPORT.md). No new novelty claim is made.\n',
 'baseline_registry.md':'# C53 baseline status\n\n'+lead+'\nNative upstream source:125 unchanged Python files, pinned before fitting. Observation concat, frozen-AE concat, frozen-AE Gate and native directional Gate are selected before outcomes; no test-dependent backbone selection. Native concat training/checkpoint-format equivalence qualified on TRAIN. Memory-limited PatchTST AE concat uses effective batch128/microbatch64 with a disclosed dropout/numerical adaptation. Persistence and validation-selected direct ridge are diagnostic controls in the frozen evaluation supplement.\n',
 'experimental_results.md':'# C53 results pending\n\n'+lead+'\nNo completed 40-model comparison or new-crop result is available in this snapshot. Source VAL scores select native checkpoints. Do not use partial completed configurations to declare a winner. See progress.json and the supervisor on the server for current state.\n',
 'negative_evidence.md':'# C53 retained negative evidence\n\n'+context+'\nCPU activation storage was numerically lossless in its small check but slow (27.38s/full update); native-layer checkpointing exceeded the28GiB qualification bound. Both are excluded from completed formal fits. A first microbatch parameter-equality test failed; the retained diagnosis records .000409 maximum parameter-update difference and6.02e-6 post-update prediction difference with dropout disabled in that check. Formal iid dropout draw ordering differs. One integration attempt failed from a read-only logging-wrapper order; no selected weight was produced. These are disclosed implementation/runtime facts, not research discoveries.\n',
 'hypothesis_space.md':'# C53 hypotheses awaiting full evidence\n\n'+lead+'\nH1: native AE/Gate advantages survive this source-protocol subset and the new crop batch. Requires all prescribed fits and the sealed second-batch comparison.\n\nH2: an apparent learned-dynamics failure is explained by ordinary forecasting/conditioning choices or regularization. Persistence and a same-information direct ridge are required controls.\n\nH3: a residual shared world-model limitation remains after data, training, selection and baseline confounds. No evidence for this hypothesis is yet established by C53. A follow-up must use a fresh protocol and independent confirmation rather than tune this transfer batch.\n',
 'research_ideas.md':'# C53 direction status\n\n'+lead+'\n'+context+'\nThe active experiment obtains strong baseline evidence before proposing a method. Neither a preprocessing repair, a small capacity-controlled VitalDB gain nor an unfinished Greenhouse grid is accepted as a new world-model framework. A useful interaction among known components remains eligible if directly supported later.\n',
 'final_research_direction.md':'# C53 current research disposition\n\n'+lead+'\nOverall objective ACTIVE_UNACHIEVED. No final Research Question / mechanism / real-domain minimal-method claim is accepted. The next decision follows the complete native subset, held-out crop audit and ordinary controls. Cross-batch error alone will not establish novelty, causal identification or an architecture requirement.\n',
 'CURRENT_STATE.md':'# C53 live state\n\n'+lead+'\nActive training source: train_native53_v6.py; capture_native53_v6.py; selection SELECTION_SEAL_V6.json. Evaluation source: score_native53_v7.py; verifier: verify_native53.py. The old V1 and V5 supervisor states are historical terminal/controlled-repair traces and must not be used to infer a live worker. Inspect actual process identity before restarting anything.\n\nRaw data, caches and checkpoints remain in the user-authorized large-disk project directory. No result synchronization to the workstation. Public target: Gifoe/wm4health.\n',
}
for name,text in reports.items():
    path=R/name;assert not path.exists();path.write_text(text,encoding='utf-8')
dump(OUT/'report_manifest.json',{'utc':seen,'scope':'dated live progress','files':[{'path':name,'sha256':sha(R/name)} for name in reports]})
latest=P/'CURRENT_RESEARCH_STATUS_V53.md';assert not latest.exists()
latest.write_text(reports['CURRENT_STATE.md'],encoding='utf-8')
print(json.dumps({'status':progress['status'],'completed_predictive_fits':fits,'planned':40,'completed_updates':updates,'second_batch_opened':progress['transfer_opened'],'reports':len(reports)}),flush=True)
