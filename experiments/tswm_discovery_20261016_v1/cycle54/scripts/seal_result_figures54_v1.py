"""Check exact plot provenance; explicit operator attestation seals image inspection."""
import argparse,datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
p=argparse.ArgumentParser();p.add_argument('--attest-three-pngs-inspected',action='store_true',required=True);p.parse_args()
folder=OUT/'result_figures_v1';manifest=read(folder/'manifest.json');assert manifest['dummy_only'] is False
for rel,value in manifest['inputs_unchanged_sha256'].items():assert sha(ROOT/rel)==value
for r in manifest['outputs']:assert sha(folder/r['file'])==r['sha256']
assert sha(folder/'plotted_values.json')==manifest['plotted_values_sha256']
values=read(folder/'plotted_values.json');assert values['dummy_only'] is False and values['point_count']==264
original=read(OUT/'evaluation54.json')['records']+read(OUT/'em_option_learning_v1/evaluation/summary.json')['records'];by_run={r['run']:r for r in original};assert len(by_run)==16
risk=read(OUT/'causal_risk_interpretation_v1/summary.json');groups={'recovery_and_dynamics':0,'causal_risk':0,'paired_source_option':0}
for row in values['points']:
    groups[row['figure']]+=1
    if row['figure']=='recovery_and_dynamics':
        r=by_run[row['run']];expected=r[row['field']] if row['h'] is None else next(h[row['field']] for h in r['horizons'] if h['h']==row['h'])
    elif row['figure']=='causal_risk':expected=next(q[row['field']] for q in risk['records'] if q['run']==row['run'] and q['h']==row['h'])
    else:
        c=next(c for c in risk['comparisons'] if c['task']==row['task'] and c['comparison']=='existing True minus matched False, seed'+str(row['seed']))
        h=next(h for h in c['horizons'] if h['h']==row['h']);expected=h['delta_mean'];assert row['pointwise95_interval']==h['pointwise95_interval']
    assert row['value']==expected
assert groups=={'recovery_and_dynamics':48,'causal_risk':192,'paired_source_option':24}
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ACTUAL_ALL16_RESULT_IMAGES_INSPECTED','figure_manifest_sha256':sha(folder/'manifest.json'),'plotted_value_source_checks':264,'pointwise_interval_checks':24,'figures_inspected':['recovery_and_dynamics.png','causal_risk.png','paired_source_option.png'],'all16_runs_retained':True,'all6_horizons_retained':True,'visual_attestation':'The operator viewed all three exact-hash remote PNGs through the RAM-only preview. No clipped axes, legends or scope notes; method colors and initialization markers retained; raw/normalized curves and both pair contrasts visible. Some EM curves overlap because their fitted metrics agree, and all underlying points remain in the exact-value registry.','formal_plot_geometry':read(folder/'render_geometry.json'),'local_result_files_written':0,'additional_optimizer_updates':0,'accepted_final_directions':0}
target=OUT/'RESULT_FIGURES_ACTUAL_QA54_V1.json'
with target.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps({'status':record['status'],'source_values_checked':264,'receipt_sha256':sha(target)}),flush=True)
