"""Render label-layout correction and verify all scored artifacts remain unchanged."""
from native53 import *
import subprocess
OUT=R/'outputs/plot_layout_revision_v1';assert not OUT.exists();OUT.mkdir()
state=OUT/'state.json'
def save(status,**extra):dump(state,{'utc':now(),'status':status,'pid':os.getpid(),**extra})
save('STARTED_LAYOUT_ONLY_CORRECTION')
EVAL=ART/'evaluation_v1';A=ART/'analysis_v1'
assert json.loads((R/'outputs/plot_repair_v3/qualification.json').read_text())['status']=='PASS_PNG_SVG_PLOTS_WITH_ALL_SCORED_ARTIFACTS_UNCHANGED'
raw=json.loads((EVAL/'raw_manifest.json').read_text());before={}
for item in raw:
    path=EVAL/item['path'];assert sha(path)==item['sha256'];before[path]=item['sha256']
for path in [EVAL/'verification.json',EVAL/'summary.json',EVAL/'raw_manifest.json',A/'summary.json',A/'manifest.json',A/'all_channel_horizons.csv',A/'paired_contrasts.json']:
    before[path]=sha(path)
python=pathlib.Path(STORAGE['cache_root'])/'cycle53/plot_venv_v1/Scripts/python.exe'
env=os.environ.copy();env.pop('PYTHONPATH',None);env['MPLBACKEND']='Agg'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
env['MPLCONFIGDIR']=str(pathlib.Path(STORAGE['cache_root'])/'cycle53/mpl_config_v5')
with (OUT/'render.log').open('wb') as stream:
    child=subprocess.Popen([str(python),'-B','-u',str(R/'scripts/render_native53_v5.py')],cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
    save('RUNNING_LAYOUT_CORRECTION',child_pid=child.pid);code=child.wait()
if code:save('FAILED_LAYOUT_CORRECTION_NO_FIT_RESTART',exit_code=code);sys.exit(1)
for path,expected in before.items():assert sha(path)==expected,path.name
manifest=json.loads((A/'figures_v5/manifest.json').read_text());assert len(manifest)==15
for item in manifest:assert sha(A/'figures_v5'/item['path'])==item['sha256']
dump(OUT/'qualification.json',{'utc':now(),'status':'PASS_LAYOUT_REVISION_WITH_NUMERIC_ARTIFACTS_UNCHANGED',
    'plots':5,'files':15,'forecast_groups_hash_verified_unchanged':84,'other_numeric_files_hash_verified_unchanged':7,
    'revision':'Shortened equivalent contrast labels, wider contrast-panel gutter, increased vertical-axis label clearance. Values, intervals, colors and series remain unchanged.',
    'plot_source_sha256':sha(R/'scripts/render_native53_v5.py'),'previous_figure_manifest_sha256':sha(A/'figures_v4/manifest.json'),
    'figure_manifest_sha256':sha(A/'figures_v5/manifest.json'),'model_or_metric_recomputed':False})
save('LAYOUT_REVISION_COMPLETE_NUMERIC_ARTIFACTS_UNCHANGED')
print(json.dumps({'status':'LAYOUT_REVISION_COMPLETE_NUMERIC_ARTIFACTS_UNCHANGED','plots':5}),flush=True)
