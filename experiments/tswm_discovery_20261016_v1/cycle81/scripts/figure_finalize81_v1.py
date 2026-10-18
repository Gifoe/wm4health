"""Make failed integrations categorical, then finalize an append-only server snapshot."""
import sys,pathlib,hashlib,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common81_v1 import *
import numpy as np
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 r=read(OUT/'RESULTS81_V1.json');systems=read(OUT/'COHORT81_V1.json')['development'];kinds=['native','fim_readout_residual','fim_readout_zero','rff_residual','affine_residual','ordinary_weak_dictionary'];labels=['Native FIM','Learned residual','Learned zero','Random residual','Affine residual','Weak dictionary'];colors=['#333333','#0072B2','#56B4E9','#D55E00','#009E73','#CC79A7'];lookup={(z['system'],z['kind']):z for z in r['rows'] if z['initial_condition_index']==1};fig,(ax,bad)=plt.subplots(2,1,figsize=(10,6),height_ratios=[5,.6],sharex=True,layout='constrained');failures=[]
 for j,(kind,label,color) in enumerate(zip(kinds,labels,colors)):
  xx=np.arange(6)+(j-2.5)*.12;valid=np.asarray([lookup[sid,kind]['solver']['success'] and lookup[sid,kind]['finite_complete'] for sid in systems]);yy=np.asarray([lookup[sid,kind]['nrmse'] or 1. for sid in systems]);ax.scatter(xx[valid],yy[valid],s=35,label=label,color=color)
  if (~valid).any():
   bad.scatter(xx[~valid],np.zeros((~valid).sum()),marker='x',s=55,color=color)
   for i in np.where(~valid)[0]:failures.append(dict(system=systems[int(i)],kind=kind))
 ax.set_yscale('log');ax.set_ylabel('New-IC NRMSE (log scale)');ax.set_title('Frozen-feature adaptation: all six development systems');ax.legend(ncols=3,fontsize=8);ax.grid(axis='y',alpha=.25);bad.set_yticks([0],['Failed\nintegration']);bad.set_ylim(-.5,.5);bad.set_xticks(range(6),[str(z) for z in systems]);bad.set_xlabel('ODEBench system (fixed development cohort)');bad.spines[['top','right']].set_visible(False);path=storage('artifact')/'development_new_ic81_v2.png';fig.savefig(path,dpi=170);plt.close(fig)
 dump(OUT/'FIGURE_QA81_V2.json',dict(utc=utc(),figure_path=str(path),figure_sha256=sha(path),failed_new_ic_integrations=failures,failures_have_no_numeric_error_position=True,numerical_result_sha256=sha(OUT/'RESULTS81_V1.json'),numerical_result_unchanged=True))
 text=(ROOT/'FEATURE_ADAPTATION_SCREEN81_V2.md').read_text(encoding='utf-8').replace('figures/development_new_ic81_v1.png','figures/development_new_ic81_v2.png');text+='\n\nFigure QA: failed integrations appear in a separate categorical row in the canonical figure. The retained historical V1 figure used an x at display height10 for failure, which is not an NRMSE measurement. The table and original failure status are unchanged.\n'
 with (ROOT/'FEATURE_ADAPTATION_SCREEN81_V3.md').open('x',encoding='utf-8') as f:f.write(text)
 readme=(ROOT/'README.md').read_text(encoding='utf-8').replace('FEATURE_ADAPTATION_SCREEN81_V2.md','FEATURE_ADAPTATION_SCREEN81_V3.md')
 with (ROOT/'README81_FINAL.md').open('x',encoding='utf-8') as f:f.write(readme)
 old=OUT/'publication_v2';dest=OUT/'publication_v3';dest.mkdir(exist_ok=False);pub=[]
 def add(rel,raw,encoding='utf-8'):
  if encoding=='utf-8':assert all(token.encode() not in raw for token in json.loads(os.environ.get('C81_PRIVATE_EXPORT_PATTERNS_JSON','[]'))),(rel,'private export pattern')
  p=dest/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);pub.append(dict(path=rel,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),server_snapshot_file=str(p),encoding=encoding))
 for row in read(old/'manifest.json')['files']:
  if row['path']=='README.md':continue
  raw=pathlib.Path(row['server_snapshot_file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'];add(row['path'],raw,row['encoding'])
 for rel in ['FEATURE_ADAPTATION_SCREEN81_V3.md','outputs/FIGURE_QA81_V2.json','scripts/figure_finalize81_v1.py']:add(rel,(ROOT/rel).read_text(encoding='utf-8-sig').encode())
 add('README.md',readme.encode());add('figures/development_new_ic81_v2.png',path.read_bytes(),'binary');assert len({z['path'] for z in pub})==len(pub)
 dump(dest/'manifest.json',dict(utc=utc(),files=pub,file_count=len(pub),bytes=sum(z['bytes'] for z in pub),prior_snapshot_manifest_sha256=sha(old/'manifest.json'),canonical_checker='scripts/verify81_v3.py',canonical_report='FEATURE_ADAPTATION_SCREEN81_V3.md',canonical_figure='figures/development_new_ic81_v2.png',numerical_experiment_unchanged=True,large_data_weights_prediction_arrays_included=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_payload_files_written=0));print(json.dumps(dict(status='FINAL_FIGURE_AND_SNAPSHOT_READY',files=len(pub),manifest_sha256=sha(dest/'manifest.json'))),flush=True)
if __name__=='__main__':main()
