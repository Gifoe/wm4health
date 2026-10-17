"""Source-bound seed dots and persistence diagnostics, with numeric geometry QA."""
from common57 import *
import csv,io,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    e=read(OUT/'EVALUATION57_V1.json');s=read(OUT/'SUMMARY57_V1.json')
    assert s['evaluation_sha256']==sha(OUT/'EVALUATION57_V1.json')
    methods=['native','independent','running09c','running01c','full','proper']
    labels=['Native','Independent','Running .9','Running .1','Full bank','Proper']
    colors=['#0072B2','#E69F00','#009E73'];markers=['o','s','^']
    artifact=paths()['artifact_root']/'figures_v1';artifact.mkdir(parents=True,exist_ok=False)
    public=OUT/'figures';public.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'svg.fonttype':'none','pdf.fonttype':42})
    points=[];exports=[]
    specs=[('primary_effect57_v1','primary_h20_plan_effect_mse','Plan-effect conditional-mean MSE (h=20)'),('persistence57_v1','rho','Learned persistence rho')]
    for filename,key,ylabel in specs:
        fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
        source_values=[r[key]['mean'] if isinstance(r[key],dict) else r[key] for r in e['records']]
        use_log=bool(key=='primary_h20_plan_effect_mse' and min(source_values)>0)
        for ax,sensor,title in zip(axes,SENSORS,['Linear sensor','Nonlinear sensor (asinh)']):
            for xpos,method in enumerate(methods):
                rows=sorted([r for r in e['records'] if r['sensor']==sensor and r['method']==method],key=lambda r:r['seed'])
                vals=[]
                for index,r in enumerate(rows):
                    val=r[key]['mean'] if isinstance(r[key],dict) else r[key];vals.append(val)
                    if method=='proper':x=xpos;color='#222222';marker='D'
                    else:x=xpos+[-.14,0,.14][index];color=colors[index];marker=markers[index]
                    artist=ax.scatter([x],[val],s=42,color=color,marker=marker,edgecolors='white',linewidths=.5,zorder=4)
                    assert np.max(abs(np.asarray(artist.get_offsets(),dtype=float)-[[x,val]]))<1e-12
                    points.append({'figure':filename,'tag':r['tag'],'sensor':sensor,'method':method,'seed':r['seed'],'metric':key,'x':x,'value':val})
                if method!='proper':
                    line=ax.hlines(np.mean(vals),xpos-.23,xpos+.23,color='#222222',lw=1.4,zorder=3)
                    assert abs(line.get_segments()[0][0,1]-np.mean(vals))<1e-12
            ax.set_title(title);ax.set_xticks(range(6),labels,rotation=28,ha='right');ax.set_xlim(-.5,5.5)
            ax.grid(axis='y',color='#D9D9D9',lw=.65,zorder=0);ax.spines[['top','right']].set_visible(False)
            if use_log:ax.set_yscale('log')
            if key=='rho':ax.axhline(.95,color='#666666',ls='--',lw=1.2);ax.set_ylim(0,1.02)
        axes[0].set_ylabel(ylabel)
        handles=[Line2D([],[],marker=m,color=c,linestyle='',label=f'Seed {seed}',markersize=6) for c,m,seed in zip(colors,markers,SEEDS)]
        handles.append(Line2D([],[],marker='D',color='#222222',linestyle='',label='Deterministic proper control',markersize=6))
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,frameon=False,fontsize=9)
        fig.subplots_adjust(left=.08,right=.985,top=.90,bottom=.27,wspace=.12)
        svg=public/(filename+'.svg');png=artifact/(filename+'.png');pdf=artifact/(filename+'.pdf')
        fig.savefig(svg,bbox_inches='tight');fig.savefig(png,dpi=180,bbox_inches='tight');fig.savefig(pdf,bbox_inches='tight');plt.close(fig)
        exports.append({'name':filename,'svg':str(svg),'svg_sha256':sha(svg),'png':str(png),'png_sha256':sha(png),'pdf':str(pdf),'pdf_sha256':sha(pdf),'log_y':use_log})
    assert len(points)==64 and len({(p['figure'],p['tag']) for p in points})==64
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=['figure','tag','sensor','method','seed','metric','x','value']);writer.writeheader();writer.writerows(points)
    with (OUT/'FIGURE_VALUES57_V1.csv').open('x',encoding='utf-8',newline='') as f:f.write(buf.getvalue())
    record={'utc':utc(),'status':'FIGURES_RENDERED_NUMERIC_GEOMETRY_QA_PASS_VISUAL_INSPECTION_PENDING','source_evaluation_sha256':sha(OUT/'EVALUATION57_V1.json'),'source_summary_sha256':sha(OUT/'SUMMARY57_V1.json'),'point_count':64,'seed_mean_geometry_verified':True,'all_actual_dot_offsets_verified':True,'exports':exports,'values_csv_sha256':sha(OUT/'FIGURE_VALUES57_V1.csv'),'plot_script_sha256':sha(__file__),'contract_sha256':sha(ROOT/'visual-composer/visual-contract.md'),'visual_inspection':'PENDING_ACTUAL_PNG_REVIEW','no_local_result_files_written':True}
    dump(OUT/'FIGURES_NUMERIC_QA57_V1.json',record)
    print(json.dumps({'status':record['status'],'points':len(points),'exports':len(exports)}),flush=True)

if __name__=='__main__':main()
