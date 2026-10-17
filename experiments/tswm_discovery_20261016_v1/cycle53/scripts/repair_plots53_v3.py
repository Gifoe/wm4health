"""Use fixed plot geometry in the existing isolated runtime; never refit/rescore."""
from native53 import *
import subprocess

OUT=R/'outputs/plot_repair_v3';assert not OUT.exists();OUT.mkdir()
state=OUT/'state.json'
def save(status,**extra):dump(state,{'utc':now(),'pid':os.getpid(),'status':status,**extra})
save('STARTED_FIXED_GEOMETRY_PLOT_RECOVERY')
assert json.loads((R/'outputs/plot_repair_v1/state.json').read_text())['status']=='FAILED_PLOT_RECOVERY_NO_FIT_RESTART'
EVAL=ART/'evaluation_v1';A=ART/'analysis_v1'
raw=json.loads((EVAL/'raw_manifest.json').read_text());assert len(raw)==84
preserved={}
for item in raw:
    path=EVAL/item['path'];assert sha(path)==item['sha256'];preserved[path]=item['sha256']
for name in ['summary.json','verification.json','raw_manifest.json','forward_checks.json','input_manifest.json']:
    path=EVAL/name;preserved[path]=sha(path)
for name in ['summary.json','manifest.json','aggregate_metrics.csv','all_channel_horizons.csv','paired_contrasts.json','fit_diagnostics.csv']:
    path=A/name;preserved[path]=sha(path)
old=[]
for version in ['figures','figures_v2','figures_v3']:
    old.extend({'path':version+'/'+p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((A/version).iterdir()) if p.is_file())
dump(OUT/'preserved_partial_plots.json',old)
venv=pathlib.Path(STORAGE['cache_root'])/'cycle53/plot_venv_v1'
python=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
assert python.is_file()
env=os.environ.copy();env.pop('PYTHONPATH',None)
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[name]='1'
env['MPLBACKEND']='Agg';env['MPLCONFIGDIR']=str(pathlib.Path(STORAGE['cache_root'])/'cycle53/mpl_config_v4')
log=OUT/'render_native53_v4.log'
with log.open('wb') as stream:
    child=subprocess.Popen([str(python),'-B','-u',str(R/'scripts/render_native53_v4.py')],cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
    save('RUNNING_FIXED_GEOMETRY_RENDER',child_pid=child.pid)
    code=child.wait()
if code:save('FAILED_FIXED_GEOMETRY_RENDER_NO_FIT_RESTART',exit_code=code);sys.exit(1)
for path,expected in preserved.items():assert sha(path)==expected,path.name
figures=A/'figures_v4';manifest=json.loads((figures/'manifest.json').read_text());assert len(manifest)==15
for item in manifest:assert sha(figures/item['path'])==item['sha256']
dump(OUT/'qualification.json',{'utc':now(),'status':'PASS_PNG_SVG_PLOTS_WITH_ALL_SCORED_ARTIFACTS_UNCHANGED',
    'plots':5,'output_files':15,'forecast_groups_hash_verified_unchanged':84,'input_metric_artifacts_hash_verified':len(preserved)-84,
    'partial_plot_files_preserved':len(old),'plot_numpy':'2.2.6','plot_pandas':'2.3.3','plot_matplotlib':'3.10.7',
    'global_environment_modified':False,'model_or_score_recomputed':False,'plot_source_sha256':sha(R/'scripts/render_native53_v4.py'),
    'figure_manifest_sha256':sha(figures/'manifest.json'),'pip_report_sha256':sha(R/'outputs/plot_repair_v1/pip_report.json'),
    'scope':'Earlier native crashes occurred during automated layout and a later PDF backend export. This version retains fixed geometry and exports PNG/SVG plus captions, omitting PDF. All failed partial files remain available. Root cause of the native crashes remains unestablished. Plot data are unchanged; all raw forecast and metric artifact hashes checked before/after.'})
save('PNG_SVG_PLOTS_COMPLETE_NO_SCORES_CHANGED')
print(json.dumps({'status':'PNG_SVG_PLOTS_COMPLETE_NO_SCORES_CHANGED','plots':5,'files':15}),flush=True)
