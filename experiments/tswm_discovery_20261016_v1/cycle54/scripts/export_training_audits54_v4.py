"""Immutable diagnostic/public-protocol delta, excluding signals and large caches."""
import pathlib,json,hashlib,datetime,time,math
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(b):return hashlib.sha256(b).hexdigest()
def read_state(path):
 for _ in range(5):
  try:return json.loads(path.read_text(encoding='utf-8-sig'))
  except json.JSONDecodeError:time.sleep(.05)
 raise RuntimeError('Changing state could not be read consistently')
def main():
 assert not (OUT/'evaluation54.json').exists() and not (OUT/'READOUT_SEAL54_V1.json').exists(),'This publication must precede heldout scoring'
 assert (OUT/'training_audit_replay_v1/replay_seed0_verification.json').exists()
 dest=OUT/'training_audit_publication_snapshot_v4';assert not dest.exists();dest.mkdir()
 old={}
 for sub,manifestname in [('protocol_publication_snapshot_v1','PROTOCOL_PUBLICATION_MANIFEST_V1.json'),('progress_publication_snapshot_v3','PROGRESS_PUBLICATION_MANIFEST_V3.json')]:
  p=OUT/sub/'manifest.json';m=json.loads(p.read_text());old.update({r['path']:r['sha256'] for r in m['files']});old[manifestname]=sha(p.read_bytes())
 candidates={}
 def add(p,public=None):
  p=pathlib.Path(p);assert p.is_file();rel=public or p.relative_to(ROOT).as_posix();candidates[rel]=p
 for p in sorted(ROOT.rglob('*')):
  r=p.relative_to(ROOT)
  if p.is_file() and 'outputs' not in r.parts and '__pycache__' not in r.parts and p.suffix in ['.py','.ps1','.md','.json'] and old.get(r.as_posix())!=sha(p.read_bytes()):add(p)
 for sub in ['training_objective_audit_v2','reference_control_audit_v2','normalizer_control_audit_v1','em_switch_audit_v2','training_audit_report_v1','training_audit_replay_v1']:
  for p in sorted((OUT/sub).iterdir()):
   if p.is_file() and p.suffix in ['.json','.csv']:add(p)
 for stem in ['training_audit54_v1','training_audit54_v2','reference_control54_v1','reference_control54_v2','normalizer_control54_v1','em_switch54_v1','em_switch54_v2']:
  state=read_state(OUT/(stem+'.json'));assert state['status'] in ['FAILED_PRESERVED','COMPLETE_NO_TRAINING_OR_HELDOUT_CHANGES']
  for suffix in ['.json','.stdout.log','.stderr.log']:
   p=OUT/(stem+suffix)
   if p.exists():add(p)
 add(OUT/'POSTPROCESS_WATCHER_HANDOFF54_V3.json')
 completed=[];supplemental=[]
 for base,items in [(OUT/'runs',completed),(OUT/'em_option_learning_v1/runs',supplemental)]:
  for folder in sorted(base.glob('*')):
   rp=folder/'result.json'
   if not rp.exists():continue
   r=read_state(rp);assert sha(pathlib.Path(r['checkpoint']).read_bytes())==r['checkpoint_sha256']
   items.append({k:r[k] for k in ['method','task','seed','iterations','checkpoint_sha256','runtime_seconds']})
   for p in sorted(folder.iterdir()):
    if p.is_file() and p.suffix in ['.json','.jsonl','.log'] and old.get(p.relative_to(ROOT).as_posix())!=sha(p.read_bytes()):add(p)
 snapshot={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FROZEN_TRAINING_AUDITS_COMPLETE_PRIMARY_AND_EXISTING_OPTION_LEARNING_CONTINUE','primary_planned_fits':12,'primary_completed_fits':len(completed),'supplemental_planned_fits':4,'supplemental_completed_fits':len(supplemental),'primary_complete':completed,'supplemental_complete':supplemental,'primary_state_observed':read_state(OUT/'state54.json'),'supplemental_state_observed':read_state(OUT/'em_option_learning_v1/state.json'),'postprocessing_state_observed':read_state(OUT/'postprocess54_v3.json'),'heldout_evaluation_opened':False,'accepted_directions':0,'overall_goal':'ACTIVE_UNACHIEVED','local_result_file_sync':False}
 pp=dest/'PROGRESS54_V4.json';pp.write_text(json.dumps(snapshot,indent=2),encoding='utf-8');add(pp,pp.name)
 readme=dest/'README.md';readme.write_text('# C54: training-contract attribution and matched native learning controls\n\n'+f"Snapshot UTC {snapshot['utc']}: primary comparison {len(completed)}/12 fits complete; supplemental existing EM-option controls {len(supplemental)}/4 complete. Heldout scoring has not opened. The full comparison remains in progress.\n\n"+'The completed [training-only audit](TRAINING_OBJECTIVE_FINDING54_V1.md) found reference-dependent A gradients under the public default total-derivative recipe. Equal target frequencies, native loss/gradient identity and cached-score replay controls localize the reference effect. The existing source EM option removes this A-gradient dependence at the frozen points; it does not establish trained-model quality. Four matched learning controls were therefore declared before any heldout score. This observation alone does not establish a new world-model primitive.\n\n'+'See [matched-control protocol](EM_OPTION_LEARNING54_V1.md), [literature boundary](LITERATURE_REFERENCE_BOUNDARY54_V1.md), [EM implementation repair](EM_IMPLEMENTATION_AUDIT54.md), and [frozen replay](AUDIT_REPLAY54_V1.md). Two tiny typed parameter cards allow the frozen-point audit to be replayed without retraining the models. Exact array roundtrips and the actually executed seed0 replay are recorded. All1,024 normalizer estimates per model are provided in CSV; full score/derivative caches and runtime checkpoint files remain on the large server disk.\n\n'+'The active scoring pipeline is `postprocess54_v3.py`: wait for both model seals, prepare the4 extra training-only maps, then prepare the12 original maps and score, followed by extra scoring and joint reports. All16 models/maps precede the first heldout comparison. The original12 protocol and its third default seed per task are retained. No new method, independent real-domain result, or accepted final research direction is claimed by this progress snapshot.\n',encoding='utf-8');add(readme,'README.md')
 records=[]
 for public,p in sorted(candidates.items()):
  b=p.read_bytes()
  if old.get(public)==sha(b):continue
  text=b.decode('utf-8-sig') if b.startswith(b'\xef\xbb\xbf') else b.decode('utf-8');transform=None
  if b.startswith(b'\xef\xbb\xbf'):b=text.encode('utf-8');transform='Remove UTF-8 BOM for public JSON/text; original server file retained.'
  assert len(b)<350000,(public,len(b));target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
  records.append({'path':public,'bytes':len(b),'characters':len(text),'parts':math.ceil(len(text)/3500),'sha256':sha(b),'original_sha256':sha(p.read_bytes()),'publication_transform':transform,'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'server_snapshot_file':str(target)})
 manifest={'utc':snapshot['utc'],'status':'IMMUTABLE_TRAINING_ONLY_DIAGNOSTIC_AND_PROTOCOL_DELTA','files':records,'file_count':len(records),'bytes':sum(r['bytes'] for r in records),'transport_chunk_characters':3500,'total_content_parts':sum(r['parts'] for r in records),'excluded':'Raw signal datasets, full serialized model checkpoints, large Gaussian score/derivative caches, changing training logs, credentials and private transport helpers. Two explicitly identified tiny typed audit parameter cards are included.'}
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'files':len(records),'bytes':manifest['bytes'],'content_parts':manifest['total_content_parts'],'manifest_sha256':sha((dest/'manifest.json').read_bytes()),'primary_complete':len(completed),'supplemental_complete':len(supplemental),'heldout_opened':False}),flush=True)
if __name__=='__main__':main()
