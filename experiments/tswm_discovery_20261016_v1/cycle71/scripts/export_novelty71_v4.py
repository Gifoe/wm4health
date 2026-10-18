"""Append the primary-source update without changing a training protocol."""
import sys,pathlib,json,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
source_pin=read(ROOT/'SOURCE_SCREEN_SPLITJEPA71_V1.json')
assert source_pin['source_files_read']==7 and source_pin['selected_bytes']==22758 and not source_pin['source_execution_claimed']
assert sha(OUT/'HIDDEN_STATE_QUALIFICATION71_V3.json')==read(OUT/'HIDDEN_STATE_CLOSURE71_V1.json')['result_sha256']
heading='\n\n## Joint-shift prior-art refinement, C71 v4\n\n'
updates={
 'literature_review.md':'NOVELTY_BOUNDARY71_V4.md adds primary-source boundaries for SplitJEPA, JEPA-TTT, stratified retention, R2-WAM and structural growth. SplitJEPA code was inspected at a8d65ecb: seven files22758 bytes Git-hash verified, not executed. Paired invariance and fixed-interface predictor adaptation are covered; unpaired joint shifts remain an unvalidated differentiation question. Position-paper agendas are not empirical solutions.',
 'baseline_registry.md':'SplitJEPA source-only registry added; its synthetic rotation uses matched pairs and frozen representations, robot code requires separate prepared pairs. No author source redistributed or native fit claimed. For any later joint-shift study use complete observation-adapter/frozen-dynamics, replay, factorized state-space and larger fixed-state controls; do not create a weak fixed-encoder-only comparison.',
 'experimental_results.md':'No new model-performance result in this source screen. Existing hidden-state execution qualification and ongoing original-budget formal fits retain their previous scope. The three-seed simulation assessment cannot itself validate a sequential unknown-interface claim. No reserved EVAL or new source performance scores accessed.',
 'negative_evidence.md':'Generic stable/variant splitting, predictor-only test-time updates, replay and latent expansion already have direct prior art or public agendas. Their existence does not prove the stricter unpaired joint-change setting solved. A source limitation is not an observed failure. Existing affine/response-control negative results remain; historical real-system test records cannot become a fresh held-out cohort by renaming.',
 'hypothesis_space.md':'Retain a precise joint-shift question: which apparent changes can be calibrated into an existing predictive state, and which require richer state? Compare observation-only, dynamics-only and joint changes with equal available information. Non-identifiable explanations require uncertainty, not a forced correct latent label. Complete ordinary observer/state-space/replay controls must precede a new method.',
 'research_ideas.md':'A/B are consolidated into a focused provisional sequential observation/dynamics reuse route. Its method remains unspecified and unvalidated. SplitJEPA uses meaningful paired correspondences; JEPA-TTT fixes observation mapping. Removing those assumptions alone is not innovation. Next independent work is a prospective unused-data and information-contract audit, while the original C71 fits continue.',
 'final_research_direction.md':'No final recommendation accepted. Sequential observation/dynamics reuse is a focused development question, not a verified gap. Need an independent cohort and a residual surviving complete ordinary controls; this source screen supplies neither a new method nor performance evidence. Keep the full task active.'
}
previous=OUT/'publication_hidden_state_v2'/'files';proofs={};planned=[]
for name,body in updates.items():
 before=(ROOT/name).read_bytes();assert before==(previous/name).read_bytes() and heading.encode() not in before
 suffix=(heading+body+'\n').encode();planned.append((name,before,suffix))
for name,before,suffix in planned:
 with (ROOT/name).open('ab') as f:f.write(suffix)
 after=(ROOT/name).read_bytes();assert after==before+suffix
 proofs[name]=dict(before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),prefix_preserved=True)
progress=read(OUT/'BOUNDED_CAMPAIGN71_V1_progress.json')
closed=dict(utc=utc(),status='CLOSED_PRIMARY_SOURCE_AND_CODE_SCREEN_NO_NEW_PERFORMANCE',source_pin_sha256=sha(ROOT/'SOURCE_SCREEN_SPLITJEPA71_V1.json'),literature_note_sha256=sha(ROOT/'NOVELTY_BOUNDARY71_V4.md'),read_depth=dict(SplitJEPA='primary methods + selected code',JEPA_TTT='primary methods + limitations',stratified_retention='primary position paper',R2_WAM='primary method + diagnostic + limitations',growing_world_models='author position-paper page',Meta_iLaD='primary abstract + repository listing; full PDF unavailable'),new_model_fits=0,new_performance_rows=0,source_code_executed=False,formal_campaign_snapshot=progress,training_protocol_changed=False,ledgers=proofs,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0)
dump(OUT/'NOVELTY_SCREEN71_V4.json',closed)
status='''# Current research state: joint observation/dynamics change remains provisional

The latest source and method boundary is cycle71/NOVELTY_BOUNDARY71_V4.md; machine-readable read-depth and scope are in cycle71/outputs/NOVELTY_SCREEN71_V4.json. Seven SplitJEPA source files were read and hash verified, without source execution or source redistribution. Paired block separation, fixed-interface dynamics adaptation, replay and structural expansion are existing work; none of this screen proves an unpaired sequential joint-shift failure or its solution.

The focused development question is preserving/reusing dynamics across sequential observation and mechanism changes, with complete observer, alignment, state-space and replay controls. The next independent work is a prospective data/information-contract audit that excludes previously scored records. A distinct new learning mechanism and supporting independent residual are still missing. No final direction is accepted; overall goal ACTIVE/UNACHIEVED.

The original C71 three-seed full-budget campaign and frozen free-simulation assessment continue unchanged. Its bounded live-state snapshot is in the screen receipt and is historical after its timestamp. C71 graph-orbit coverage and hidden-state qualification limitations remain. C72 remains closed. Raw data, arrays and weights stay on server G storage; no local result synchronization.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V71_NOVELTY4.md').open('x',encoding='utf-8') as f:f.write(status)
files=list(updates)+['NOVELTY_BOUNDARY71_V4.md','SOURCE_SCREEN_SPLITJEPA71_V1.json','scripts/export_novelty71_v4.py','outputs/NOVELTY_SCREEN71_V4.json']
patterns=json.loads(os.environ['C71_PRIVATE_EXPORT_PATTERNS_JSON']);staged=[]
for rel in sorted(files):
 raw=(ROOT/rel).read_bytes()
 for needle in patterns:assert needle.encode() not in raw,(rel,'Private export pattern')
 staged.append((rel,raw.decode('utf-8-sig').encode()))
dest=OUT/'publication_novelty_v4';dest.mkdir(exist_ok=False);rows=[]
for rel,b in staged:
 target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status=closed['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),raw_arrays_weights_and_upstream_source_included=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0)
dump(dest/'manifest.json',manifest)
print(json.dumps(dict(status=closed['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'),last_closed_epoch=progress['last_completed_epoch'])),flush=True)
