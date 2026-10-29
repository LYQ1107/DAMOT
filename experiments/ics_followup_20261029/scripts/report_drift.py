"""Full CSV/LaTeX export and a fixed, prespecified two-panel S3 figure."""
import csv,json,os,sys
os.environ.setdefault('MKL_THREADING_LAYER','GNU')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(ROOT/'code/tools/analysis'))
from ics_probe_analyze import write_csv
from protocol import write_json
OUT=ROOT/'results/drift_decomposition'

def esc(x):return str(x).replace('_',r'\_').replace('%',r'\%')
def tex(path,rows,cols):
    lines=[r'\begin{tabular}{'+'l'*len(cols)+'}',r'\toprule',' & '.join(esc(k) for k in cols)+r' \\',r'\midrule']
    for row in rows:
        values=[]
        for k in cols:
            v=row[k]
            try:v='%.5f'%float(v) if '.' in str(v) or 'e-' in str(v) else v
            except (ValueError,TypeError):pass
            values.append(esc(v))
        lines.append(' & '.join(values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];path.write_text('\n'.join(lines)+'\n')

def main():
    names=[n for n in ['F','HF','NOGEOM'] if (OUT/n/'complete.json').exists()]
    collected={}
    for stem in ['drift_energy','cross_stage_retrieval','drift_restoration']:
        rows=[r for n in names for r in csv.DictReader((OUT/n/(stem+'.csv')).open())]
        write_csv(str(OUT/(stem+'.csv')),rows);collected[stem]=rows
    tex(OUT/'drift_energy.tex',collected['drift_energy'],['model','from_stage','to_stage','bucket','basis','k_effective','mean_energy_share','mean_in_sq','mean_out_sq'])
    tex(OUT/'cross_stage_retrieval.tex',collected['cross_stage_retrieval'],['model','query_stage','gallery_stage','domain','mAP_same','mAP_cross','forgetting_proxy'])
    tex(OUT/'drift_restoration.tex',collected['drift_restoration'],['model','query_stage','gallery_stage','domain','basis','k_effective','forgetting_restored','forgetting_reduction'])
    energy=collected['drift_energy'];comparisons=[]
    idx={(r['model'],r['from_stage'],r['to_stage'],r['bucket'],r['basis'],r['k_requested']):r for r in energy}
    for r in energy:
        if r['model']!='F':continue
        for model in [n for n in names if n!='F']:
            s=idx[(model,r['from_stage'],r['to_stage'],r['bucket'],r['basis'],r['k_requested'])]
            new={k:r[k] for k in ['from_stage','to_stage','bucket','basis','k_requested','k_effective']};new['comparison']=model+' minus F'
            for field in ['mean_in_sq','mean_out_sq','mean_total_sq','mean_energy_share']:
                new['F_'+field]=float(r[field]);new[model+'_'+field]=float(s[field]);new['difference_'+field]=float(s[field])-float(r[field])
            comparisons.append(new)
    if comparisons:
        # Consistent columns for two comparison models.
        for r in comparisons:
            model=r['comparison'].split(' ')[0]
            for field in ['mean_in_sq','mean_out_sq','mean_total_sq','mean_energy_share']:r['comparison_'+field]=r.pop(model+'_'+field)
        write_csv(str(OUT/'coupling_absolute_drift_comparison.csv'),comparisons)
    fig,ax=plt.subplots(1,2,figsize=(17,5.7));colors=['#2466a8','#df8a24','#338865'];width=.8/len(names)
    choices=[('FAP residual PCA','16'),('Weather oracle','16'),('Identity oracle','16'),('ICS covariance eigenvectors','16'),('ICS stored bank_pcs','2'),('Random','16')]
    labels=['FAP PCA\n16D','Weather oracle\nrank 4','Identity oracle\n16D','ICS covariance\n16D','ICS bank_pcs\n2D','Random\n16D']
    for i,n in enumerate(names):
        vals=[]
        for basis,k in choices:
            r=idx[(n,'3','4','old',basis,k)];vals.append(100*float(r['mean_energy_share']))
        ax[0].bar(np.arange(6)+(i-(len(names)-1)/2)*width,vals,width,label=n,color=colors[i])
    ax[0].set_xticks(np.arange(6));ax[0].set_xticklabels(labels,fontsize=10);ax[0].set_ylabel('Mean per-RoI drift energy in subspace (%)');ax[0].set_title('Stage 3 → 4, old-domain test RoIs');ax[0].legend(frameon=False)
    conditions=[('baseline',0)]+[(n,k) for n in ['FAP residual PCA','Identity oracle'] for k in [2,8,16]]
    labels2=['None','FAP 2D','FAP 8D','FAP 16D','ID 2D','ID 8D','ID 16D']
    for i,n in enumerate(names):
        vals=[]
        for basis,k in conditions:
            if basis=='baseline':rows=[r for r in collected['cross_stage_retrieval'] if r['model']==n and r['query_stage']=='4'];field='forgetting_proxy'
            else:rows=[r for r in collected['drift_restoration'] if r['model']==n and r['query_stage']=='4' and r['basis']==basis and int(r['k_requested'])==k];field='forgetting_restored'
            assert len(rows)==3;vals.append(100*np.mean([float(r[field]) for r in rows]))
        ax[1].plot(np.arange(7),vals,'o-',color=colors[i],label=n)
    ax[1].axhline(0,color='.5',lw=.7);ax[1].set_xticks(np.arange(7));ax[1].set_xticklabels(labels2,rotation=20,ha='right',fontsize=10);ax[1].set_ylabel('Same-stage mAP − cross-stage mAP (pp)');ax[1].set_title('Stage 4 queries: equal mean over three old domains');ax[1].legend(frameon=False)
    for a in ax:a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    fig.suptitle('FAP/weather/identity bases: F stage1 fit; ICS: destination bank; statistics: test videos',fontsize=12);fig.tight_layout(rect=(0,0,1,.95))
    for ext in ['png','pdf']:fig.savefig(str(OUT/('drift_decomposition.'+ext)),dpi=200,bbox_inches='tight')
    plt.close(fig)
    write_json(OUT/'figure_protocol.json',dict(models=names,left='Fixed stage3→4, old test RoIs; m16 bases plus actual stored ICS2 (covariance16 is separate)',right='Stage4←j, equal mean of the three old domains; all six restorations',all_transitions_and_ranks_retained_in_CSV=True,selection_by_metric=False))
    print('S3 report generated for',names)
if __name__=='__main__':main()
