"""Fixed illustrative paths, not best/worst examples or model-selection evidence."""
import sys,pathlib,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import numpy as np
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 art=storage('artifact');base=art/'composition_screen_v1';methods=['FIM_local_composed','weak_polynomial3','weak_rff128'];colors=['#0072B2','#009E73','#D55E00'];names=['FIM local','Weak polynomial','Weak RFF128'];fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained');selection=[]
 for fi,(family,coordinate) in enumerate([('fhn',2),('hill',1)]):
  tag=f'{family}0_s2026831_noise00'
  for ci,clamp in enumerate([False,True]):
   ax=axes[fi,ci];allvalues=[];truth=None
   for kind,color,label in zip(methods,colors,names):
    path=base/f'{tag}_{kind}_ic0_clamp{int(clamp)}.npz';assert path.exists()
    with np.load(path,allow_pickle=False) as a:y=a['target'];p=a['prediction'];t=a['time']
    good=p.shape==y.shape and np.isfinite(p).all();selection.append(dict(tag=tag,kind=kind,initial_index=0,clamp=clamp,coordinate=coordinate,path=str(path),sha256=sha(path),finite_complete=good))
    if truth is None:truth=y[:,coordinate];ax.plot(t,truth,color='#000000',lw=2,label='Reference');allvalues.extend(truth.tolist())
    if good:ax.plot(t,p[:,coordinate],color=color,lw=1.5,label=label);allvalues.extend(p[:,coordinate].tolist())
    else:ax.plot([],[],color=color,label=label+' (failed)')
   wide=max(abs(np.asarray(allvalues)))>20*max(1.,max(abs(truth)))
   if wide:ax.set_yscale('symlog',linthresh=1.);ax.text(.02,.98,'All finite values shown; symlog',transform=ax.transAxes,va='top',fontsize=8)
   ax.set_title(f'{family.upper()}: '+('clamp node0, observe recipient' if clamp else 'free coupled evolution'));ax.set_xlabel('Time');ax.set_ylabel(f'State coordinate{coordinate}');ax.grid(alpha=.2);ax.legend(fontsize=8)
 figure=art/'selected_paths83_v1.png';fig.savefig(figure,dpi=170);plt.close(fig);dump(OUT/'SELECTED_PATHS83_V1.json',dict(utc=utc(),purpose='Illustration selected by fixed setting0 seed2026831 noise0 initial0, not by outcome',selection=selection,figure_path=str(figure),figure_sha256=sha(figure),no_axis_value_clipping=True));print(json.dumps(dict(status='FIXED_PATH_FIGURE_READY',path=str(figure),sha256=sha(figure))),flush=True)
if __name__=='__main__':main()
