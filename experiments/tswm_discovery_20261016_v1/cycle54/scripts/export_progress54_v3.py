"""Immutable progress delta including only complete fit receipts and curves."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 dest=OUT/'progress_publication_snapshot_v3';assert not dest.exists();dest.mkdir()
 old=json.loads((OUT/'protocol_publication_snapshot_v1/manifest.json').read_text());oldmap={r['path']:r['sha256'] for r in old['files']}
 selected=[]
 for path in sorted(ROOT.rglob('*')):
  rel=path.relative_to(ROOT)
  if path.is_file() and 'outputs' not in rel.parts and '__pycache__' not in rel.parts and path.suffix in ['.md','.json','.py','.ps1'] and oldmap.get(rel.as_posix())!=sha(path.read_bytes()):selected.append((path,rel.as_posix()))
 for name in ['QUALIFICATION_EM_FIX54_V1.json','em_queue54_v1.json','em_queue54_v2.json','POSTPROCESS_WATCHER_HANDOFF54_V2.json','PROTOCOL_PUBLICATION_RECEIPT54_V1.json']:
  selected.append((OUT/name,'outputs/'+name))
 completed=[]
 for folder in sorted((OUT/'runs').glob('*')):
  result=folder/'result.json'
  if result.exists():
   r=json.loads(result.read_text());completed.append({k:r[k] for k in ['method','task','seed','iterations','checkpoint_sha256','runtime_seconds']})
   for name in ['result.json','curve.jsonl','EM_OBJECTIVE_AUDIT_V3.json']:
    path=folder/name
    if path.exists():selected.append((path,path.relative_to(ROOT).as_posix()))
  for name in ['stderr.log','stdout.log']:
   path=folder/name
   if folder.name=='em_linear_small_seed0' and path.exists():selected.append((path,path.relative_to(ROOT).as_posix()))
 progress={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'CORRECTED_ORDINARY_EM_BASELINE_COMPLETE_RP_QUEUE_CONTINUES','planned_fits':12,'completed_fits':len(completed),'completed':completed,'heldout_evaluation_started':False,'accepted_research_directions':0,'active_postprocessing':'postprocess54_v2.py waits for the 12-fit seal, then uses evaluate54_v2.py and analyze54_v2.py','baseline_amendment_sha256':sha((ROOT/'plan54_em_amendment_v2.json').read_bytes()),'no_local_result_file_sync':True}
 pp=dest/'PROGRESS54_V3.json';pp.write_text(json.dumps(progress,indent=2),encoding='utf-8');selected.append((pp,pp.name))
 readme=dest/'README.md';readme.write_text('# C54 jointly learned dynamics audit — comparison running\n\n'+f"Snapshot UTC {progress['utc']}: {len(completed)}/12 formal fits complete. Six corrected ordinary LGSSM EM baselines have completed; the RP-GSSM 10,000-update queue continues. Heldout scoring has not started.\n\n"+'The original native EM baseline produced NaNs because its initial covariance update omitted a sample-count normalization. The standard correction was independently verified before heldout scoring; see [implementation audit](EM_IMPLEMENTATION_AUDIT54.md), [amendment](plan54_em_amendment_v2.json) and [qualification](outputs/QUALIFICATION_EM_FIX54_V1.json). This implementation issue is not a new research gap.\n\n'+'See [protocol](PROTOCOL.md), [reproduction contract](REPRODUCE.md), and [PRO-DYN task-scope audit](PRODYN_SCOPE_AUDIT54.md). For the amended run, use `train54_v3.py` / `run_em54_v2.py` and `postprocess54_v2.py`; the RP training script is unchanged. Complete comparisons and scientific interpretation follow only after all twelve model parameters and training-only readouts are frozen. No new research direction has been accepted. Signals, weights, arrays and failed candidates remain on server storage.\n',encoding='utf-8');selected.append((readme,'README.md'))
 records=[]
 for path,public in selected:
  original=path.read_bytes();b=original;transform=None
  if path.name=='POSTPROCESS_WATCHER_HANDOFF54_V2.json':
   obj=json.loads(original.decode('utf-8-sig'));compact={'utc':str(obj['utc']),'reason':str(obj['reason']),'prior_watcher_status_required_and_verified_by_launcher':'WAITING_FOR_ALL_12_SEALED_FITS','stopped_process_ids':[int(h['ProcessId']) for h in obj['handles']],'original_server_receipt_sha256':sha(original),'publication_note':'PowerShell enriched string/provider serialization omitted; compact primitive metadata only. Original full receipt retained on server.'};b=json.dumps(compact,indent=2).encode();transform='Replace expanded PowerShell provider metadata with primitive UTC, reason, validated stopped process IDs and original receipt hash.'
  if path.name=='QUALIFICATION_EM_FIX54_V1.json':
   obj=json.loads(b);obj.pop('native_mstep_source',None);obj['publication_note']='Full native source function omitted; source file and SHA256 retained. Original complete receipt remains on server.';obj['qualification_execution_count_note']='One native and four corrected M-step calls in qualification (one direct corrected update plus three fit_em iterations). These are excluded from formal fits.';b=json.dumps(obj,indent=2).encode();transform='Omit full upstream source text; preserve function hash and numerical diagnostics, clarify discarded M-step count.'
  b.decode('utf-8');assert len(b)<250000,public;target=dest/'files'/public;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
  records.append({'path':public,'bytes':len(b),'sha256':sha(b),'original_sha256':sha(original),'publication_transform':transform,'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'server_snapshot_file':str(target)})
 manifest={'utc':progress['utc'],'status':'IMMUTABLE_PROGRESS_DELTA_NO_HELDOUT_SCORES','file_count':len(records),'bytes':sum(r['bytes'] for r in records),'files':records,'excluded':'Live fit logs, signals, weights, raw arrays, credentials and private control helpers.'}
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({'files':len(records),'bytes':manifest['bytes'],'manifest_sha256':sha((dest/'manifest.json').read_bytes()),'progress':progress}),flush=True)
if __name__=='__main__':main()
