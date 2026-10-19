"""Evidence plots from completed results; private diagnostic, not paper claims."""
import ctypes,json,os,pathlib,re
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
if os.name=='nt':
 fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 cofe=json.loads((ROOT/'outputs/cofe_etth1_affinity1/results.json').read_text());risk=json.loads((ROOT/'outputs/continuous_counterfactual/results.json').read_text())
 fig,axes=plt.subplots(2,2,figsize=(12,8.3),layout='constrained');colors=['#a65036','#1d6f96','#43865c'];offsets=[-3.,0.,3.]
 for row,color in zip(cofe,colors):
  axes[0,0].plot(offsets,[row['test'][str(o)]['coverage90'] for o in offsets],'o-',label=row['variant'],color=color)
  axes[0,1].plot(offsets,[row['test'][str(o)]['interval90_width_normalized'] for o in offsets],'o-',label=row['variant'],color=color)
 axes[0,0].axhline(.9,ls='--',lw=1,color='gray');axes[0,0].set(ylim=(0,1.04),xlabel='Synthetic normalized level shift',ylabel='Empirical 90% coverage',title='COFE full model: one dataset / one seed')
 axes[0,1].set(xlabel='Synthetic normalized level shift',ylabel='90% interval width (normalized)',title='Dispersion correction removes location sensitivity')
 axes[0,0].legend(fontsize=8,loc='lower right')
 log=(ROOT/'outputs/kuma_native_pems03_affinity1/stdout.log').read_text();rows=re.findall(r'Epoch (\d+) \| train MSE ([.\d]+) \| val MSE ([.\d]+)',log);a=np.array(rows,dtype=float)
 axes[1,0].plot(a[:,0],a[:,1],label='Training',color='#1d6f96');axes[1,0].plot(a[:,0],a[:,2],label='Validation',color='#a65036');axes[1,0].axvline(11,color='gray',ls='--',lw=1)
 axes[1,0].set(xlabel='Epoch',ylabel='Normalized MSE',title='KUMA full released PEMS03 run; selected epoch 11');axes[1,0].legend(fontsize=8)
 cases=risk['rows'];gap=np.array([max(e['relative_gap'] for e in r['extrema']) for r in cases]);axes[1,1].semilogy(np.arange(1,19),np.maximum(gap,1e-10),'o',color='#1d6f96');axes[1,1].axhline(.01,color='#a65036',ls='--',label='Frozen stop gate (1%)')
 axes[1,1].set(xlabel='Frozen system / dimension / horizon case',ylabel='Generic SDP gap (fraction; numerical)',title='Existing tools resolve all 18 restricted Gaussian cases',ylim=(3e-11,.03),xticks=[1,6,12,18]);axes[1,1].legend(fontsize=8)
 for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
 fig.suptitle('Cycle 5 evidence: concrete repairs and rejected solver need',fontsize=14)
 out=ROOT/'outputs/figures';out.mkdir(exist_ok=True);fig.savefig(out/'cycle5_diagnostics.png',dpi=170);fig.savefig(out/'cycle5_diagnostics.svg');plt.close(fig)
 print('Saved diagnostic PNG and editable SVG')
if __name__=='__main__':main()
