"""Export authored audit evidence; upstream bytes and signals remain excluded."""
import sys,pathlib,json,hashlib,datetime,os
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';PROJECT=ROOT.parent
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
q=read(OUT/'COHORT_ELIGIBILITY73_V1.json');e=read(OUT/'cohort_audit73_v1_exit.json')
assert e['exit_code']==0 and digest(OUT/'cohort_audit73_v1_stdout.log')==e['log_sha256']
assert q['raw_signal_or_image_files_downloaded']==q['new_model_fits']==q['new_model_performance_rows']==0 and q['generator_asts_parsed']==5
assert len(q['findings'])==4 and not any(r['joint_shift_cohort'] for r in q['findings'])
status='''# C73 current state: prospective data contracts screened

The narrowed development question remains sequential observation/dynamics reuse without paired views or future-task encoder exposure. See cycle71/NOVELTY_BOUNDARY71_V4.md for the primary methods and code boundaries. It is not yet a validated gap or new method.

C73 has now checked nine pinned author metadata/generator files and five ASTs across four Causal Chamber designs. None alone documents the proposed joint unknown-interface/plant-mechanism sequence. Intake-load changepoints alter prescribed inputs; pressure control has endogenous fan loads in one fixed condition; optical-polarizer and CRL intervention protocols have different contracts. CITRIS is explicitly retained as a temporal process, separate from static Buchholz samples. See cycle73/COHORT_DECISION73_V1.md and cycle73/outputs/COHORT_ELIGIBILITY73_V1.json. No raw signals/images were downloaded, no author generator executed, and no model trained or scored in this audit. Previously scored C22/C69 records are not a fresh cohort; no project-wide untouched-data certification is claimed.

Next work is to find a valid independently specified cohort, or explicitly construct declared measurement shifts on unused physical trajectories, then run complete ordinary controls before proposing a new mechanism. The original full-budget C71 three-seed training and frozen free-simulation assessment continue unchanged; its result is still pending. No final direction accepted; overall goal ACTIVE/UNACHIEVED. Raw data/weights/arrays stay on authorized server G storage. GitHub receives authored code, reports and compact evidence only; no local result synchronization.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V73.md').open('x',encoding='utf-8') as f:f.write(status)
files=['PROTOCOL73_V1.md','SOURCE_SELECTION73_V1.json','COHORT_DECISION73_V1.md','literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md','scripts/audit_cohort73_v1.py','scripts/capture_audit73_v1.py','scripts/export_audit73_v1.py','outputs/COHORT_ELIGIBILITY73_V1.json','outputs/cohort_audit73_v1_launch.json','outputs/cohort_audit73_v1_exit.json','outputs/cohort_audit73_v1_stdout.log']
patterns=json.loads(os.environ['C73_PRIVATE_EXPORT_PATTERNS_JSON']);staged=[]
for rel in sorted(files):
 raw=(ROOT/rel).read_bytes()
 for needle in patterns:assert needle.encode() not in raw,(rel,'Private export pattern')
 staged.append((rel,raw.decode('utf-8-sig').encode()))
dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);rows=[]
for rel,b in staged:
 target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(target)))
m=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=q['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),raw_data_images_weights_arrays_or_upstream_source_included=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0)
with (dest/'manifest.json').open('x',encoding='utf-8') as f:json.dump(m,f,indent=2)
print(json.dumps(dict(status=m['status'],files=len(rows),bytes=m['bytes'],manifest_sha256=digest(dest/'manifest.json'))),flush=True)
