"""Archive the full frozen composition screen and all post-discovery offset controls."""
import sys,pathlib,os,csv,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common77_v1 import *
r=read(OUT/'COMPOSITION_RESULTS77_V2.json');v=read(OUT/'VERIFICATION77_V1.json');gauge=read(OUT/'GAUGE_PROFILE77_V3.json');pins=read(OUT/'SOURCE_PIN77_V2.json')
assert read(OUT/'run77_v2_exit.json')['exit_code']==0 and read(OUT/'run77_v1_exit.json')['exit_code']==1
assert read(OUT/'gauge77_v3_exit.json')['exit_code']==0 and read(OUT/'gauge77_v2_exit.json')['exit_code']==1
assert v['result_sha256']==sha(OUT/'COMPOSITION_RESULTS77_V2.json') and gauge['composition_result_sha256']==sha(OUT/'COMPOSITION_RESULTS77_V2.json')
assert len(r['rows'])==48 and v['forecast_values_verified']==1787904 and v['replay_checks']==432
assert gauge['profile_rows']==324 and gauge['forecast_values_verified']==9642240 and gauge['independent_scalar_replay_checks']==2916
assert all(z['observed']['finite'] and z['canonical']['finite'] for z in r['rows'])
families=['C75_innovation_monotone','C76_simulation_affine','C76_simulation_monotone']
def row(t,c,source=None,family=None,method=None):return next(z for z in r['rows'] if z['destination']==t and z['condition']==c and (z['method']==method if method else z['calibration_source']==source and z['family']==family))
table='| Destination | Calibration source / family | Identity | Affine | Nonlinear |\n|---|---|---:|---:|---:|\n'
for t in ['EMPS','F16']:
 for method in ['unadapted','known_sensor']:
  table+='| '+t+' | '+method+' | '+' | '.join(format(row(t,c,method=method)['observed_nrmse'],'.9f') for c in ['identity','affine','nonlinear'])+' |\n'
 for source in ['EMPS','F16']:
  for family in families:table+='| '+t+' | '+source+' / '+family+' | '+' | '.join(format(row(t,c,source,family)['observed_nrmse'],'.9f') for c in ['identity','affine','nonlinear'])+' |\n'
profiletable='| Offset diagnostic | EMPS internal NRMSE | EMPS DEV NRMSE | F16 DEV NRMSE |\n|---|---:|---:|---:|\n'
for label in ['fixed_-1.0','fixed_-0.5','fixed_0.0','fixed_0.5','fixed_1.0','privileged_grid_offset']:
 profiletable+='| '+label+' | '+' | '.join(format(next(z['observed_nrmse'] for z in gauge['profiles'] if z['calibration_source']=='EMPS' and z['family']=='C76_simulation_monotone' and z['condition']=='nonlinear' and z['offset_label']==label and z['destination']==destination and z['phase']==phase),'.9f') for destination,phase in [('EMPS','internal'),('EMPS','dev'),('F16','dev')])+' |\n'
report='''# C77: frozen component composition and coordinate-offset attribution

No new research gap is accepted. F16's ordinary simulation-calibrated nonlinear map transfers to the EMPS dynamics with observed NRMSE0.021319733, compared with0.024260188 for the privileged true-sensor fitted-source diagnostic. The reverse transfer is poor, but already poor for identity sensing. Its large apparent nonlinear-composition deficit therefore does not isolate a new failure mechanism.

## Task and information boundary

This is a retrospective diagnostic using existing C75/C76 components, not a prospectively untouched or independently confirmed test. Each transfer arm uses a frozen destination source ARX64 model and a frozen map fitted on the calibration-source system. It never receives the destination's new-view calibration labels or map. Source/destination identity is supplied. The experimenter has already seen every system/condition, and diagonal maps exist as information-privileged references. No new fit, architecture, knot, source model, optimizer or DEV-selected family is introduced by the composition matrix.

The shared sensor function is artificially defined in the separately source-standardized scalar coordinates of EMPS position and F16 acceleration. These physical units are different. This is a controlled mathematical factorization over real trajectories, not evidence that an actual physical sensor is shared between the two devices. There is no within-trajectory plant change, unknown-task identification or causal intervention claim. Supplied recorded future inputs remain an explicit conditional simulation contract, especially for feedback-collected EMPS. Original official test MAT outputs remain numerically closed.

Each forecast gets the destination's128 observed history points and128 supplied future input points on the exact C75 DEV origins. No future output enters the transport function. C75 calibration-fit x SD is used only as a reporting denominator and is unavailable to that function. Privileged canonical-coordinate diagnostics invert the known synthetic generator after prediction; they do not become training signals or operational latent-state claims.

## Complete frozen matrix

Observed NRMSE, all records included. Map families are C75 innovation-monotone, C76 global-affine simulation, and C76 internally selected monotone simulation. Diagonal rows reconstruct prior predictions; off-diagonal rows are frozen transport without destination calibration. No family/source is selected using destination DEV.

'''+table+'''
All48 aggregate rows,216 record rows,96 support-stratum rows,36 canonical contrasts and18 fixed-grid diagnostics accompany this report. The three observation conditions and two systems are not pooled into one score.

With C76 monotone maps, EMPS→F16 identity canonical RMSE is0.951288667 and nonlinear canonical RMSE0.957353893. Its canonical MSE excess over the known-sensor diagnostic is0.776326834 already for identity and0.787903183 for nonlinear sensing; the nonlinear increment beyond identity is only0.011576350. Thus a large overall transfer error should not be relabelled as evidence of nonlinear observation-composition failure. In the reverse direction, the same ordinary family transfers successfully at this task scale. These are two physical systems, not enough independent replication to claim a general law.

History-only support checks retain all forecasts. In EMPS→F16,12.0955% of observed history values exceed the source map's outer knots, and352 of564 origins have their full history inside. Nonlinear canonical RMSE on those352 is still0.930442487 versus0.957353893 overall. This shows a failure among history-in-range windows; it does not exclude later forecast extrapolation, since no future-target-based filtering was performed. F16→EMPS has every observed history in range. On the fixed canonical grid[-2,2], the selected EMPS nonlinear inverse has coordinate bias-0.583440012 and RMSE0.606273629; F16's map has bias-0.007167726 and RMSE0.064145180. Correct within-system prediction is not proof that a calibration recovered a unique transferable coordinate.

## Post-discovery offset intervention

After reading the matrix, a separately frozen addendum profiles offsets[-1,-0.5,0,0.5,1] for every pinned inverse map. No optimizer runs, and original maps and main results stay unchanged. If h is replaced by h+delta, updating its forward inverse consistently gives an exact observation-prediction change governed by z_base+delta*(R1-1), where R1 is the homogeneous ARX response to an all-one output history. This is direct linear algebra for the fitted model, not a new algorithm or an identifiability theorem for the real plant.

EMPS's RMS(R1-1) over128 steps is0.006316588; F16's is1.315448158, about208 times larger. The fitted EMPS is near an offset symmetry, with1-sum(a)=-0.000618663; it is not exactly symmetric. Small source sensitivity is not proof of exact structural nonidentifiability or the sole explanation of source bias. A separate pure-integrator mathematical fixture verifies the exact unit-sum special case without changing either actual fitted model.

The following rows profile the selected EMPS nonlinear map. The final row is an explicitly privileged intervention: delta0.583440012 cancels its mean inverse-coordinate error on the fixed synthetic truth grid. It is unavailable to learners and is never selected as a method or from DEV.

'''+profiletable+'''
The privileged correction worsens source internal NRMSE from0.006006053 to0.007123214 while improving destination F16 DEV from0.851031892 to0.316017477. That tradeoff is consistent with a source-fit calibration offset that compensates source-model error and transfers poorly to a model with very different offset sensitivity. It does not prove that all residuals are offset, that the real sensor is unidentifiable, or that a new world-model mechanism is required. The fixed+0.5 diagnostic is also retained, not substituted as a chosen result. Every one of the324 profiles is reported.

## Verification and execution provenance

All1,787,904 main forecast values, targets, support masks/metrics and contrast arithmetic reconstruct. Independent scalar ARX/interval-map replay covers432 first/last-window forecasts, maximum absolute disagreement2.842170943040401e-14. All18 diagonal rows reconstruct original stored predictions. The actual imported calibration library is hash-checked against its prior published snapshot, and an AST check confirms no new fit/optimizer/update calls in the main runner.

The offset addendum stores and rechecks9,642,240 forecast values across324 rows and54 array artifacts. Independent scalar replay covers2916 profile windows, maximum difference4.399258735077183e-14; the complete translation identity matches direct shifted parameters to1.1768364061026659e-14. Metric reconstruction differences are exactlyzero.

Three engineering failures are retained and excluded from scientific interpretation. Main V1 expected C74 per-case keys in a C75 archive and stopped before any cross-system result; V2 validates the actual pred/record_ids/origins schema. Gauge V1 imported a second process-affinity configuration and stopped before main; its command was observed but had no separate contemporaneous captured PID/log. Gauge V2's artificial unit-sum F16 fixture accidentally created a1.1797485833 unstable secondary mode, amplifying rounding error despite an exact-arithmetic invariant trajectory; it stopped before any profile. Gauge V3 uses a pure AR1 integrator only for that mathematical check. Actual model roots, masks outside the current process, source data and scientific conditions never change. All source versions, correction notes and captured terminal logs remain.

## Prior art and research decision

The primary [Bai and Reyland Wiener identification paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3587721/) discusses identifiability assumptions and internal-coordinate scaling ambiguity. Its stable-system/iid-input assumptions are not asserted here, and it is not cited as a proof of our offset-specific finite-data result. The offset response above is derived and checked directly. The primary [Wiener identification project](https://www.it.uu.se/research/syscon/Ident/Wiener_Id.html) already includes nonlinear sensing and blind adaptation. Existing compositional/continual world-model boundaries remain in C71/NOVELTY_BOUNDARY71_V4.md and C76/NOVELTY_AND_DISPOSITION76_V1.md. Neither a generic coordinate ambiguity nor an unadapted transfer failure supplies sufficient novelty.

The next discriminating control is ordinary history-only offset/gain alignment or state estimation, using the same128 already observed destination history samples and no future target or extra calibration segment. If that repairs the transfer, retire this scalar composition premise as well. If not, amplitude support, source-model approximation, stochastic observation effects, coordinate conventions and complete shared/state-space/replay baselines still need resolution before a general research gap can be accepted. Do not increase F16 model capacity or tune the reported score. The independent C71 original-budget/free-simulation assessment continues. No final recommended direction or new mechanism is accepted; full goal ACTIVE/UNACHIEVED.
'''
with (ROOT/'COMPOSITION_FINDING77_V1.md').open('x',encoding='utf-8') as f:f.write(report)
entries={
 'literature_review.md':'Classical nonlinear-output identification already treats internal-coordinate ambiguities; our offset sensitivity is direct fitted-model algebra, not a claimed new theorem or a transferred iid-input guarantee. Existing compositional/continual world-model boundaries remain in C71/C76. See primary source links and scope limits in COMPOSITION_FINDING77_V1.md.',
 'baseline_registry.md':'No new fits in the composition screen or offset addendum. Transport18 existing maps from three mature families across two frozen learned dynamics; retain12 unadapted/privileged-sensor references. All48 rows reported, no family selection. Post-discovery324 fixed/privileged offset diagnostics are not methods and never replace original maps. Next necessary control is ordinary history-only coordinate alignment under identical information.',
 'experimental_results.md':'All48 aggregate/216 record rows and support/contrast/grid tables are published.\n\n'+table+'\nMain1,787,904 forecast values and432 scalar recurrences verified. Offset profiles add324 rows,9,642,240 rechecked values and2916 scalar recurrences. No official-test output or new model fit.',
 'negative_evidence.md':'F16 learned nonlinear simulation map transfers to EMPS with NRMSE0.021319733 versus privileged known sensor0.024260188. Reverse failure is largely present even for identity sensing: canonical RMSE0.951288667(identity) versus0.957353893(nonlinear). A privileged source-grid offset intervention reduces nonlinear F16 transfer NRMSE0.851031892→0.316017477 while worsening source internal fit. This points to ordinary coordinate/source-model sensitivity and does not establish a new mechanism gap. Preserve all engineering failures; no physical-model failure is inferred from them.',
 'hypothesis_space.md':'H1 nonlinear composition is inherently broken: not isolated; reverse failure exists already for identity and the other direction works. H2 source calibration compensates a weakly constrained coordinate offset: supported by direct sensitivity/intervention, without proving sole causality or physical nonidentifiability. H3 amplitude extrapolation alone: history-in-range failures remain, but future extrapolation is not excluded. H4 genuinely unresolved general world-model reuse: still unproven; needs complete ordinary history-only observer/alignment controls and independent mechanisms.',
 'research_ideas.md':'Do not promote this normalized scalar diagnostic to a general world-model method or use the privileged offset as an algorithm. Next fixed-information falsification is ordinary destination-history-only offset/gain alignment. A broad sequential sensor/dynamics framework still needs distinct mechanism, original prior-art boundary and independent evidence. No more F16-only capacity tuning. Keep the full objective intact rather than redefining success as a repaired benchmark.',
 'final_research_direction.md':'No final direction accepted. Research Question tested: do frozen ordinary observation maps compose across two learned physical dynamics? Evidence: one transfer direction works; the reverse is already poor under identity sensing and highly offset-sensitive. Existing Methods: nonlinear-output identification and ordinary online coordinate/state calibration. Proposed Idea: none validated. Falsification: simple history-only alignment repairs the discrepancy. Feasibility: existing server data/maps with CPU-only evaluation; C71 full-budget GPU run continues. Next Experiment: same-history offset/gain alignment without destination future/calibration labels, then retain or reject the scalar composition premise. General research goal ACTIVE/UNACHIEVED.'}
for name,body in entries.items():
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write('# '+name[:-3].replace('_',' ')+'\n\n'+body+'\n\nSee COMPOSITION_FINDING77_V1.md for all limitations and provenance.\n')
def csvfile(name,rows):
 with (OUT/name).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
aggregate=[];records=[];strata=[];models=[]
for z in r['rows']:
 base={key:z[key] for key in ['destination','condition','method','calibration_source','family','cross_system','destination_new_view_calibration_available_to_arm','privileged']}
 aggregate.append(dict(**base,observed_mse=z['observed']['mse'],observed_rmse=z['observed']['rmse'],observed_nrmse=z['observed_nrmse'],observed_mae=z['observed']['mae'],canonical_mse=z['canonical']['mse'],canonical_rmse=z['canonical']['rmse'],canonical_mae=z['canonical']['mae'],history_outside_value_fraction=z['history_outside_value_fraction'],array_path=z['array_path'],array_sha256=z['array_sha256']))
 for rr in z['per_record']:records.append(dict(**base,case=rr['case'],origins=rr['origins'],observed_rmse=rr['observed']['rmse'],observed_nrmse=rr['observed_nrmse'],canonical_rmse=rr['canonical']['rmse'],canonical_mse=rr['canonical']['mse'],history_inside_origins=rr['history_inside_origins'],history_outside_origins=rr['history_outside_origins']))
 for ss in z['support_strata']:strata.append(dict(**base,stratum=ss['stratum'],origins=ss['origins'],observed_rmse=ss['observed']['rmse'] if ss['observed'] else None,canonical_rmse=ss['canonical']['rmse'] if ss['canonical'] else None,canonical_mse=ss['canonical']['mse'] if ss['canonical'] else None,support_applicable=z['model'] is not None))
 if z['model'] is not None and z['calibration_source']==z['destination']:models.append(dict(calibration_source=z['calibration_source'],condition=z['condition'],family=z['family'],model=z['model']))
csvfile('aggregate_metrics77_v1.csv',aggregate);csvfile('record_metrics77_v1.csv',records);csvfile('history_support77_v1.csv',strata);csvfile('canonical_contrasts77_v1.csv',r['contrasts']);csvfile('fixed_grid_diagnostics77_v1.csv',r['grid_diagnostics']);csvfile('independent_replays77_v1.csv',v['checks']);csvfile('offset_profiles77_v1.csv',gauge['profiles'])
dump(OUT/'MODEL_MAPS77_V1.json',dict(source_pin_sha256=sha(OUT/'SOURCE_PIN77_V2.json'),maps=models))
dump(OUT/'RESULT_SUMMARY77_V1.json',dict(**{k:val for k,val in r.items() if k not in ['rows','contrasts','grid_diagnostics','process_affinity']},aggregate_rows=aggregate))
dump(OUT/'VERIFICATION_SUMMARY77_V1.json',{k:val for k,val in v.items() if k!='checks'})
dump(OUT/'GAUGE_SUMMARY77_V1.json',{k:val for k,val in gauge.items() if k not in ['profiles','artifacts','process_affinity']})
dump(OUT/'GAUGE_ARRAY_MANIFEST77_V1.json',dict(raw_profile_sha256=sha(OUT/'GAUGE_PROFILE77_V3.json'),artifacts=gauge['artifacts']))
closure=dict(utc=utc(),status='CLOSED_FROZEN_COMPOSITION_WITH_OFFSET_ATTRIBUTION_NO_NEW_GAP_ACCEPTED',main_result_sha256=sha(OUT/'COMPOSITION_RESULTS77_V2.json'),verification_sha256=sha(OUT/'VERIFICATION77_V1.json'),profile_sha256=sha(OUT/'GAUGE_PROFILE77_V3.json'),protocol_sha256=r['protocol_sha256'],aggregate_rows=48,record_rows=len(records),support_rows=len(strata),main_forecast_values=1787904,main_replay_checks=432,offset_profile_rows=324,offset_forecast_values=9642240,offset_replay_checks=2916,new_fits=0,original_maps_changed=False,privileged_offset_is_a_method=False,retrospective_not_untouched_confirmation=True,official_test_values_parsed=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0)
dump(OUT/'CLOSURE77_V1.json',closure)
status='''# C77 current state: frozen composition failure has ordinary coordinate explanation

See cycle77/COMPOSITION_FINDING77_V1.md and cycle77/outputs/CLOSURE77_V1.json. Retrospective matrix:18 existing map artifacts combined with two frozen source dynamics across identity/affine/nonlinear synthetic sensors in separately normalized coordinates. No new fits;48 aggregate/216 record rows. F16's ordinary learned nonlinear map transfers to EMPS(NRMSE0.021319733). Reverse transfer is poor already under identity sensing(canonical RMSE0.951288667 versus0.957353893 under nonlinear sensing), so nonlinear-composition failure is not isolated.

A separately frozen post-discovery offset diagnostic shows fitted-model homogeneous offset-response sensitivity differs by about208x. A privileged true-grid correction, unavailable to learners, improves EMPS→F16 nonlinear NRMSE0.851031892→0.316017477 while worsening source internal fit. Every fixed offset and source/destination score retained; original maps/results unchanged. No physical nonidentifiability theorem or novel method is claimed. Main1,787,904 values/432 recurrences and offset9,642,240 values/2916 recurrences independently qualified/reconstructed. Engineering schema/import/fixture failures are preserved.

Next falsification is ordinary history-only offset/gain alignment or state estimation with the same128 observations and no future/calibration labels. Do not claim a new gap before that control. Scalar recalibration remains retired; wider general world-model direction remains unproven. Full goal ACTIVE/UNACHIEVED; accepted final directions0. Independent C71 original-budget three-seed campaign/frozen assessment continues. Official EMPS/F16 test MAT outputs stay closed; arrays/models/data stay on server G. GitHub receives authored compact evidence/code/reports without local payload synchronization.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V77.md').open('x',encoding='utf-8') as f:f.write(status)
files=list(entries)+['PROTOCOL77_V1.md','SCHEMA_CORRECTION77_V1.md','GAUGE_DIAGNOSTIC77_PROTOCOL_V1.md','GAUGE_IMPORT_CORRECTION77_V1.md','GAUGE_FIXTURE_CORRECTION77_V1.md','COMPOSITION_FINDING77_V1.md']
files+=['scripts/'+p.name for p in sorted((ROOT/'scripts').glob('*.py'))]
files+=['outputs/'+name for name in ['SOURCE_PIN77_V1.json','SOURCE_PIN77_V2.json','MODEL_MAPS77_V1.json','RESULT_SUMMARY77_V1.json','VERIFICATION_SUMMARY77_V1.json','GAUGE_SUMMARY77_V1.json','GAUGE_ARRAY_MANIFEST77_V1.json','aggregate_metrics77_v1.csv','record_metrics77_v1.csv','history_support77_v1.csv','canonical_contrasts77_v1.csv','fixed_grid_diagnostics77_v1.csv','independent_replays77_v1.csv','offset_profiles77_v1.csv','CLOSURE77_V1.json','run77_v1_launch.json','run77_v1_exit.json','run77_v1_stdout.log','run77_v2_launch.json','run77_v2_exit.json','run77_v2_stdout.log','gauge77_v2_launch.json','gauge77_v2_exit.json','gauge77_v2_stdout.log','gauge77_v3_launch.json','gauge77_v3_exit.json','gauge77_v3_stdout.log']]
patterns=json.loads(os.environ['C77_PRIVATE_EXPORT_PATTERNS_JSON']);dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);pub=[]
for rel in sorted(files):
 raw=(ROOT/rel).read_bytes()
 for needle in patterns:assert needle.encode() not in raw,(rel,'private export pattern')
 b=raw.decode('utf-8-sig').encode();target=dest/'files'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b);pub.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status=closure['status'],files=pub,file_count=len(pub),bytes=sum(z['bytes'] for z in pub),raw_arrays_weights_included=False,official_test_values_parsed=False,raw_main_result_sha256=sha(OUT/'COMPOSITION_RESULTS77_V2.json'),raw_profile_result_sha256=sha(OUT/'GAUGE_PROFILE77_V3.json'),complete_metrics_and_diagnostics_split_into_compact_files=True,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0)
dump(dest/'manifest.json',manifest);print(json.dumps(dict(status=closure['status'],files=len(pub),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'))),flush=True)
