"""Append evidence-qualified literature decisions to living ledgers and snapshot."""
import sys,pathlib,os,math,base64
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *

source=read(OUT/'SOURCE_AUDIT71_LEJEPA_V2.json');assert source['status']=='AUTHOR_SOURCE_PINNED_NOT_EXECUTED' and source['new_training_fits']==source['new_performance_results']==0
for r in source['files']:
 raw=pathlib.Path(r['server_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256'] and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['git_blob_sha1']
base=read(OUT/'PROGRESS_SNAPSHOT71_V1.json')['reports'];updates={
 'literature_review.md':'NOVELTY_BOUNDARY71_V3.md adds primary-source boundaries for WM3C, WLA, SIHA, LeJEPA identifiability, CIPHER v2 and compositional pH models. It also qualifies the original Modular NRI hidden-node demonstration as true-edge-trained. These literature findings are distinct from our experiments.',
 'baseline_registry.md':'LeJEPA-identifiability author code is pinned at de7503f1b2291ba1e9fc03d9c433180794803440: eleven selected MIT source/config files, exact Git-blob/SHA256 checks, no training or performance reproduction. SOURCE_AUDIT71_LEJEPA_V2.json records acquisition and its repaired transport. Newly reviewed papers remain literature-only baselines.',
 'experimental_results.md':'This literature update adds no model-performance result. It verifies pinned author-source bytes and narrows claims. The three native subset fits and frozen common-history assessment remain in progress/queued as recorded in the current status snapshot; EVAL remains closed.',
 'negative_evidence.md':'Do not present the anisotropic slow-feature/higher-harmonic failure as a new discovery: LeJEPA appendix F explicitly treats it. A nonlinear latent chart followed by structured dynamics is already present in CIPHER v2. These are prior-art exclusions, not new failed fits or evidence against every representation-learning approach.',
 'hypothesis_space.md':'Candidate A must distinguish observation calibration, latent-state observability, structure inference and true cross-composition reuse. WM3C already allows cross-component temporal dependence; SIHA jointly refines hidden trajectories and graphs with complete training trajectories. Missing assumptions or missing inspected code do not establish model failure. See NOVELTY_BOUNDARY71_V3.md.',
 'research_ideas.md':'Candidates A/B/C remain development routes with no accepted new mechanism. Retain optionality while excluding generic chart learning, bidirectional coupling and known slow-feature effects as standalone contributions. Next discriminating evidence is the already frozen complete-model free-simulation assessment, followed only if warranted by an independently specified structural-composition experiment.',
 'final_research_direction.md':'No final research direction is accepted. The new primary-source audit narrows possible contributions without proving that arbitrary observation/dynamics composition is solved. The full research goal remains ACTIVE/UNACHIEVED. Do not replace missing native accuracy/transfer evidence with these literature conclusions.'
}
pending=[]
for name,text in updates.items():
 p=ROOT/name;assert p.resolve().is_relative_to(ROOT.resolve());raw=p.read_bytes();suffix=('\n\n## Primary-source refinement, C71 v3\n\n'+text+'\n').encode();original=raw[:-len(suffix)] if raw.endswith(suffix) else raw
 assert hashlib.sha256(original).hexdigest()==base[name],(name,'Unexpected ledger change; preserve it')
 pending.append((p,raw,original+suffix))
for p,old,new in pending:
 if old!=new:
  tmp=p.with_name(p.name+'.c71v3.tmp')
  with tmp.open('xb') as f:f.write(new)
  tmp.replace(p)
procs=json.loads(os.environ['C71_LIVE_PROCESSES_JSON']);procs=procs if isinstance(procs,list) else [procs]
assert len([p for p in procs if '\\scripts\\train71_v3.py' in p['CommandLine']])==1
assert len([p for p in procs if '\\scripts\\queue_free_simulation71_v1.py' in p['CommandLine']])==1
obj=dict(utc=utc(),status='PRIMARY_SOURCE_NOVELTY_REFINEMENT_TRAINING_LIVE',source_audit_sha256=sha(OUT/'SOURCE_AUDIT71_LEJEPA_V2.json'),literature_addendum_sha256=sha(ROOT/'NOVELTY_BOUNDARY71_V3.md'),ledger_hashes={n:sha(ROOT/n) for n in updates},live_processes=procs,progress=[read(p) for p in sorted(OUT.glob('formal_seed*_v3_progress.json'))],new_model_fits_in_this_update=0,new_performance_results=0,reserved_eval_scored=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(OUT/'NOVELTY_REFINEMENT71_V3.json',obj)
status='''# C71: prior-art boundaries refined; baseline training and assessment queue live

See cycle71/NOVELTY_BOUNDARY71_V3.md for the reviewed primary papers and the scope of each finding. All seven living ledgers now distinguish these literature exclusions from this project's experiments. Exact author-source verification is in cycle71/outputs/SOURCE_AUDIT71_LEJEPA_V2.json; this source selection has not been trained or benchmarked here.

The original Modular NRI hidden-node demonstration used true-edge-trained dynamics; current C71 fits use fully observed states without edge-label supervision. Generic partial observation, state/structure refinement, nonlinear chart learning, bidirectional coupling and slow-feature bias are not standalone new contributions. This does not prove every observation/dynamics-composition problem solved.

Three complete subset fits remain running, with their common-history assessment queued under its frozen protocol. No training setting, checkpoint rule or EVAL access changed. No final research direction accepted; full goal ACTIVE/UNACHIEVED. Raw arrays/weights/data remain on server G storage; GitHub receives compact evidence and authored code/reports.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V71_NOVELTY_V3.md').open('x',encoding='utf-8') as f:f.write(status)
selected=list(updates)+['NOVELTY_BOUNDARY71_V3.md','SOURCE_SELECTION71_LEJEPA_V1.json','scripts/acquire_lejepa_source71_v1.py','scripts/verify_lejepa_source71_v2.py','scripts/export_novelty71_v3.py','outputs/SOURCE_AUDIT71_LEJEPA_V2.json','outputs/NOVELTY_REFINEMENT71_V3.json']
dest=OUT/'publication_novelty_v3';dest.mkdir(exist_ok=False);patterns=json.loads(os.environ['C71_PRIVATE_EXPORT_PATTERNS_JSON']);rows=[]
for rel in selected:
 p=ROOT/rel;raw=p.read_bytes()
 for needle in patterns:assert needle.encode() not in raw,(rel,'private export pattern')
 body=raw.decode('utf-8-sig');b=body.encode();target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append(dict(path=rel,original_path=rel,original_sha256=hashlib.sha256(raw).hexdigest(),encoding='UTF-8',bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),parts=math.ceil(len(body)/3500),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status=obj['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),closure_sha256=sha(OUT/'NOVELTY_REFINEMENT71_V3.json'),raw_arrays_weights_and_live_logs_included=False,reserved_eval_scored=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(dest/'manifest.json',manifest);print(json.dumps(dict(status=obj['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'),progress=obj['progress'])),flush=True)
