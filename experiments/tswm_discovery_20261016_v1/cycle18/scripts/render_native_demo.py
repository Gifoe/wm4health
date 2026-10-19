"""Plot original released demo data and frozen model draws; no model selection."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
out=ROOT/'outputs/analysis';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
for axis,name in zip(axes,['lorenz63','chua','sprottM']):
    saved=np.load(ROOT/'outputs/native_demo'/(name+'.npz'));cl=int(saved['context_length'])
    truth=saved['data'][cl:cl+10000];pred=saved['predictions'][0]
    axis.plot(truth[:,0],truth[:,1],lw=.35,alpha=.45,c='#777777',label='Released future reference')
    axis.plot(pred[:,0],pred[:,1],lw=.4,alpha=.65,c='#c74747',label='Frozen DynaMix draw (2021)')
    axis.set_title(f'{name} (context {cl})\n10,000-step forecast',fontsize=10);axis.set_xlabel('Coordinate 1');axis.set_ylabel('Coordinate 2')
axes[0].legend(loc='best',fontsize=7)
fig.suptitle('Source-native DSR examples: autonomous simulation with one unchanged model',fontsize=12)
fig.supxlabel('Three released demonstration cases, not the 54-system paper aggregate or a novel-method result.',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(out/('native_dynamix_demo.'+ext),dpi=180)
plt.close(fig)
