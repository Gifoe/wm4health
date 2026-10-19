import pathlib,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/pilot_v1'
summary=json.loads((OUT/'summary.json').read_text());proof=json.loads((OUT/'verification.json').read_text())
assert proof['status']=='passed' and proof['summary_sha256']==hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest()
colors={'N':'#000000','JZ':'#D55E00','JZ_N':'#E69F00','PZ':'#0072B2','PY':'#009E73','PZ_NC':'#CC79A7'}
fig,axes=plt.subplots(2,2,figsize=(11,6.5),constrained_layout=True)
for row,name in enumerate(['solar','jena']):
    for arm,color in colors.items():
        curve=json.loads((OUT/name/arm/'curve.json').read_text())
        x=[r['step'] for r in curve]
        axes[row,0].plot(x,[r['fused_mae'] for r in curve],label=arm,color=color,lw=1.35)
        if arm in ['N','JZ','JZ_N']:
            axes[row,1].plot(x,[r['num_mae'] for r in curve],label=arm,color=color,lw=1.35)
    for j in range(2):
        ax=axes[row,j];ax.set_xlabel('Training updates');ax.set_ylabel('Validation MAE (train-scaled)')
        ax.set_title(name+(' — fused forecast' if j==0 else ' — numerical pathway'))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8,ncol=3)
    axes[row,1].text(.02,.02,'N / PZ / PY / PZ_NC numerical paths are identical',
         transform=axes[row,1].transAxes,fontsize=7.5)
fig.suptitle('Context responsibility pilot — one seed; TRAIN/VAL only',fontsize=13)
folder=ROOT/'outputs/figures';folder.mkdir(exist_ok=True)
for ext in ['png','svg','pdf']:fig.savefig(folder/('learning_curves.'+ext),dpi=180)
print(json.dumps({'figure':'learning_curves','formats':['png','svg','pdf']}))
