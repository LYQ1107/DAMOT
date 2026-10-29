"""S1.1: audit effective model and dataset pipelines, without running a model."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code/tools/run'))
from apr_complement_experiment import Config
from protocol import diff,write_views,write_json,views
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420');NG=Path('/data1/liuyeqiang/damot_apr_complement_hf_nogeom_20261029_1443')
def transforms(x,path=''):
    out=[]
    if isinstance(x,dict):
        if 'type' in x:out.append((path,x))
        for k,v in x.items():out+=transforms(v,path+'.'+k)
    elif isinstance(x,(tuple,list)):
        for i,v in enumerate(x):out+=transforms(v,path+'[%d]'%i)
    return out
def main():
    rows=[]
    for name,source in [('HF',HF),('NOGEOM',NG)]:
        for k in [1,4]:
            cfg=Config.fromfile(str(source/'configs'/('stage%d.py'%k)))
            reference=Config.fromfile(str((Path('/data1/liuyeqiang/damot_rebuttal_20261026/configs')/('F_stage%d_e5_embedded.py'%k)) if name=='HF' else HF/'configs'/('stage%d.py'%k)))
            dest=ROOT/'results/S1'/name/('stage%d'%k);dest.mkdir(parents=True,exist_ok=True)
            value=views(cfg);active=[]
            for path,t in transforms(value):
                if t['type'] in ['SeqRandomCrop','RandomCrop']:active.append(dict(path=path,transform=t))
                if t['type'] in ['SeqRandomFlip','RandomFlip'] and float(t.get('flip_ratio',.5))>0:active.append(dict(path=path,transform=t))
                if t['type'] in ['SeqResize','Resize'] and t.get('ratio_range') is not None and list(t['ratio_range'])!=[1,1]:active.append(dict(path=path,transform=t))
            write_views(cfg,dest/'resolved_views.txt',dict(effective_model_pipelines=True,active_stochastic_geometry=active))
            write_json(dest/'config_diff.json',dict(source=str(source/'configs'/('stage%d.py'%k)),reference=str(reference.filename),model_differences=diff(reference.model,cfg.model,'model'),dataset_differences=diff(reference.data.train,cfg.data.train,'data.train'),audit_mutations=[],active_stochastic_geometry=active))
            h=cfg.model.track_head
            row=dict(model=name,stage=k,active_geometry=len(active),causal_pca_mode=h.causal_pca_mode,causal_num_pcs=h.causal_num_pcs,causal_trim_alpha=h.causal_trim_alpha,roi_proto_couple_ics=h.roi_proto_couple_ics,roi_proto_couple_beta=h.roi_proto_couple_beta,roi_proto_couple_basis=h.roi_proto_couple_basis)
            for label in ['student','contrastive']:
                row[label+'_frequency']=json.dumps([t for _,t in transforms(value[label+'_pipeline']) if t['type']=='SeqSPAFrequencyTransform'],sort_keys=True)
            rows.append(row)
            assert name!='NOGEOM' or not active,active
    with (ROOT/'results/S1/audit_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
