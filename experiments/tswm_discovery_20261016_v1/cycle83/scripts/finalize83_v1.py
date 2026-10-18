"""Verify attribution, retain the original snapshot, and append scientific boundaries."""
import sys,pathlib,time,hashlib,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import run83_v1 as run
import verify83_v1 as independent
import numpy as np
def put_doc(name,text):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text)
def main():
 start=time.perf_counter();result_hash=sha(OUT/'RESULTS83_V1.json');r=read(OUT/'RESULTS83_V1.json');a=read(OUT/'ATTRIBUTION83_V1.json');decision=read(OUT/'DECISION83_V1.json');qq=read(OUT/'QUERY_QUALIFICATION83_V1.json');assert qq['status'].startswith('PASS_') and read(OUT/'attribute83_v1_exit.json')['exit_code']==0;assert (OUT/'publication_v1/manifest.json').exists();manifest=read(OUT/'DATA_MANIFEST83_V1.json');asset=next(z for z in run.old.assets() if z['relative'].endswith('.safetensors'));model=run.load_fim_ode_local(pathlib.Path(asset['path']).parent,'cpu');model.eval();checks=[];native_checks=[];groups=[]
 for item in manifest['rows']:
  tag=item['tag']
  with np.load(item['path'],allow_pickle=False) as data:x=data['context'].copy();t=data['time'].copy()
  fn=run.Native(model,x,t,item['parents'])
  for row in [z for z in a['rows'] if z['tag']==tag and z['status']=='COMPLETE_ORACLE_DIAGNOSTIC']:
   assert sha(row['path'])==row['sha256'];original=next(z for z in r['rollouts'] if z['tag']==tag and z['kind']=='FIM_local_composed' and z['initial_index']==row['initial_index'] and z['clamp']==row['clamp'])
   with np.load(original['path'],allow_pickle=False) as z:y=z['target'].copy();p=z['prediction'].copy()
   with np.load(row['path'],allow_pickle=False) as z:stored={key:z[key].copy() for key in z.files}
   truth=np.stack([independent.exact_rhs(yy,item['family'],item['setting']) for yy in y]);truthp=np.stack([independent.exact_rhs(pp,item['family'],item['setting']) for pp in p]);assert np.allclose(truth,stored['reference_field'],atol=1e-12,rtol=1e-12);assert np.array_equal(p-y,stored['actual_error']);mask=list(range(1,y.shape[1])) if row['clamp'] else list(range(y.shape[1]));den=np.maximum(np.std(truth,axis=0),1e-12);actual=p-y;delta=stored['linearized_error'];actual_norm=np.linalg.norm(actual[:,mask]);derived=dict(reference_field_defect=float(np.mean(np.sqrt(np.mean(stored['reference_defect'][:,mask]**2,0))/den[mask])),prediction_field_defect=float(np.mean(np.sqrt(np.mean(stored['prediction_defect'][:,mask]**2,0))/den[mask])),linearized_relative_error=float(np.linalg.norm((delta-actual)[:,mask])/max(actual_norm,1e-12)),linearized_actual_cosine=float(np.sum(delta[:,mask]*actual[:,mask])/max(np.linalg.norm(delta[:,mask])*actual_norm,1e-12)),reference_support_p95=float(np.quantile(stored['reference_distance'][:,mask],.95)),prediction_support_p95=float(np.quantile(stored['prediction_distance'][:,mask],.95)))
   for key,value in derived.items():assert np.isclose(value,row[key],atol=1e-12,rtol=1e-12),(tag,key)
   for label,query,trueval,expected in [('reference',y[:17],truth[:17],stored['reference_defect'][:17]),('prediction',p[:17],truthp[:17],stored['prediction_defect'][:17])]:
    pred=fn(query)-trueval
    if row['clamp']:pred[:,0]=0.
    err=float(abs(pred-expected).max());assert np.allclose(pred,expected,atol=2e-5,rtol=2e-5),(tag,label,err);native_checks.append(dict(tag=tag,initial_index=row['initial_index'],clamp=row['clamp'],path_type=label,scalar_count=pred.size,max_abs=err))
   checks.append(dict(tag=tag,initial_index=row['initial_index'],clamp=row['clamp'],derived_metric_max_abs=max(abs(derived[k]-row[k]) for k in derived)))
 for family in ['fhn','hill']:
  for noise in [0.,.03]:
   selected=[z for z in a['rows'] if z['status']=='COMPLETE_ORACLE_DIAGNOSTIC' and z['family']==family and z['noise_sigma']==noise];keys=['reference_field_defect','prediction_field_defect','reference_support_p95','prediction_support_p95','linearized_relative_error','linearized_actual_cosine'];groups.append(dict(family=family,noise_sigma=noise,complete=len(selected),expected=16,medians={k:float(np.median([z[k] for z in selected])) if selected else None for k in keys}))
 assert run.old.check_source()==70 and sha(asset['path'])==asset['sha256'] and sha(OUT/'RESULTS83_V1.json')==result_hash;dump(OUT/'ATTRIBUTION_VERIFICATION83_V1.json',dict(utc=utc(),status='PASS_ARRAY_IDENTITIES_DERIVED_METRICS_AND_SAMPLED_NATIVE_PATH_QUERIES',checks=checks,native_path_checks=native_checks,groups=groups,query_batch_qualification_sha256=sha(OUT/'QUERY_QUALIFICATION83_V1.json'),source_files_checked=70,weights_unchanged=True,numerical_screen_unchanged=True,wall_seconds=time.perf_counter()-start))
 table='| Family | Noise | Completed | Reference-path field defect | Predicted-path field defect | Reference support p95 | Predicted support p95 | Linearized error relative discrepancy | Linearized/actual cosine |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for g in groups:table+=f"| {g['family']} | {g['noise_sigma']} | {g['complete']}/{g['expected']} | "+' | '.join('unavailable' if g['medians'][k] is None else f"{g['medians'][k]:.6g}" for k in ['reference_field_defect','prediction_field_defect','reference_support_p95','prediction_support_p95','linearized_relative_error','linearized_actual_cosine'])+' |\n'
 text=f'''# C83 path-level attribution and research judgment

This analysis was specified after the first two tasks, then applied without changing any model, seed, initial condition, ridge selection or evaluation tolerance. It is post-screen diagnostic evidence, not a new method or independent confirmatory test.

{table}

Field-defect scores use the standard deviation of the true field along the reference path, shared for reference/prediction queries. They differ from trajectory NRMSE and the context-query field score. Support distances are per-target standardized nearest-context distances, not guarantees of correct counterfactual inference. The linearized comparison uses true equations and future reference trajectories unavailable at deployment. A small discrepancy would support ordinary local sensitivity as an explanation; a large discrepancy can reflect nonlinear departure or interpolation/linearization approximation. Neither outcome automatically establishes a new research gap.

The extra interface test compares128-query evaluation,17-query prefixes, single-location solver calls, reversed queries and duplicate query sets. All16 checks pass at unchanged atol/rtol2e-5, maximum absolute discrepancy{max(z['max_abs'] for z in qq['checks']):.6g}. Therefore a target-query-set interaction is not an explanation of current field/rollout differences at this tolerance. The source decoder attends from each query to the observed context independently.

Attribution arrays, true equations and derived metrics were checked; {len(native_checks)} reference/predicted-path query groups ({sum(z['scalar_count'] for z in native_checks)} scalars) were replayed with unchanged native fields, maximum absolute discrepancy{max((z['max_abs'] for z in native_checks),default=0.):.6g}. The analytic Jacobians were checked against finite differences at{a['analytic_jacobian_checks']} initial reference points, maximum discrepancy{a['jacobian_max_abs']:.6g}. These checks validate the described diagnostics, not a broad causal interpretation of their correlations.

## What is decided

Frozen C83 development decision: `{decision['decision']}`. Generic stitching of the released local prior is not a qualified innovation. Classical ordinary identification and its explicit function-family information are retained as controls. A failed continuation gate retires this generic advantage claim; it does not prove all learned dynamics priors useless.

The narrower rescue hypothesis concerns synthetic-prior/neighbor-model mismatch. PRETRAINING_PRIOR_BOUNDARY83_V1.md verifies joint finite-trajectory filtering but does not establish its causal responsibility for these results. MODULAR_INFERENCE_BOUNDARY83_V1.md records that generic cuts and partial feedback control are already known. A decisive next attribution would compare matched joint and target-conditional inference with an ordinary exact/qualified posterior before any large training. This remains needs-mechanism/needs-evidence, not an accepted paper direction.

Observation-map transfer is only a different fallback; METAILAD_BOUNDARY83_V1.md checks the actual theorem assumptions without claiming its result covers arbitrary unseen sensors or is wrong outside its scope. No fallback experiment was launched in this cycle.

Qualified final research directions0. Overall research goal ACTIVE/UNACHIEVED. The correct response to these results is narrower causal attribution or retirement, not more architectures or a positive story manufactured from simulation counts.
'''
 put_doc('ATTRIBUTION_AUDIT83_V1.md',text);put_doc('README83_V2.md',(ROOT/'README.md').read_text(encoding='utf-8')+'\n## Final attribution supplement\n\nRead ATTRIBUTION_AUDIT83_V1.md for independently checked path-level diagnostics and the research decision. Additional source boundaries: PRETRAINING_PRIOR_BOUNDARY83_V1.md, MODULAR_INFERENCE_BOUNDARY83_V1.md, METAILAD_BOUNDARY83_V1.md. The original publication_v1 snapshot is retained on the server. Numerical models and results are unchanged.\n')
 source=OUT/'publication_v1';dest=OUT/'publication_v2';dest.mkdir(exist_ok=False);pub={}
 def snapshot(rel,raw,encoding='utf-8'):
  assert all(t.encode() not in raw for t in __import__('json').loads(os.environ.get('C83_PRIVATE_EXPORT_PATTERNS_JSON','[]'))),(rel,'private export pattern');p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);pub[rel]=dict(path=rel,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),server_snapshot_file=str(p),encoding=encoding)
 for row in read(source/'manifest.json')['files']:
  raw=pathlib.Path(row['server_snapshot_file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
  if row['path']=='README.md':raw=(ROOT/'README83_V2.md').read_text(encoding='utf-8').encode()
  snapshot(row['path'],raw,row['encoding'])
 additions=['ATTRIBUTION_PROTOCOL83_V1.md','ATTRIBUTION_AUDIT83_V1.md','PRETRAINING_PRIOR_BOUNDARY83_V1.md','MODULAR_INFERENCE_BOUNDARY83_V1.md','METAILAD_BOUNDARY83_V1.md','README83_V2.md','scripts/finalize83_v1.py','outputs/QUERY_QUALIFICATION83_V1.json','outputs/ATTRIBUTION83_V1.json','outputs/ATTRIBUTION_VERIFICATION83_V1.json','outputs/attribute83_v1_exit.json','outputs/attribute83_v1_stdout.log','outputs/close83_v1_launch.json','outputs/close83_v1_exit.json','outputs/close_attribution83_v1_launch.json','outputs/close_attribution83_v1_exit.json']
 for rel in additions:
  raw=(ROOT/rel).read_text(encoding='utf-8-sig').encode()
  if rel in pub:assert pub[rel]['sha256']==hashlib.sha256(raw).hexdigest()
  else:snapshot(rel,raw)
 dump(dest/'manifest.json',dict(utc=utc(),files=list(pub.values()),file_count=len(pub),bytes=sum(z['bytes'] for z in pub.values()),prior_snapshot_sha256=sha(source/'manifest.json'),numerical_experiment_unchanged=True,large_data_weights_arrays_included=False,qualified_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_payload_files_written=0));print(json.dumps(dict(status='FINAL_SNAPSHOT_READY',files=len(pub),manifest_sha256=sha(dest/'manifest.json'),groups=groups)),flush=True)
if __name__=='__main__':main()
