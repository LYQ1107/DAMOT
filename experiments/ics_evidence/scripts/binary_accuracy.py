"""Report ordinary binary accuracy as well as P2 balanced accuracy; same fitted probes."""
import numpy as np
from evidence_cpu import ROOT,load,csvout
from report import rows
import ics_probe_analyze as p

out=[]
for model in ['F','HF']:
    data,bank,Z,fit=load(model)
    old=rows(ROOT/'part1'/model/'deletion_replicates.csv')
    wf,wt=data['weather'][fit],data['weather'][~fit]
    a,b=np.isin(wf,['night','clear_holdout_45v']),np.isin(wt,['night','clear_holdout_45v'])
    targets=(wt[b]=='night').astype(int)
    baseline=None
    for r in old:
        name,rep=r['basis'],int(r['replicate'])
        if name=='Full':raw=Z
        else:
            E=np.load(str(ROOT/'part1'/model/'bases'/('%s_%02d.npy'%(name,rep))))
            raw=Z-(Z@E)@E.T
        bacc,pred=p.probe_predictions(raw[fit][a],(wf[a]=='night').astype(int),raw[~fit][b],targets,0)
        assert abs(100*bacc-float(r['night_clear_bacc_pp']))<1e-9
        accuracy=100*float(np.mean(pred==targets))
        if name=='Full':baseline=accuracy
        out.append(dict(model=model,basis=name,replicate=rep,accuracy_pp=accuracy,accuracy_delta_pp=accuracy-baseline,
                        balanced_accuracy_pp=100*bacc,test_rois=len(targets),test_night_rois=int(targets.sum())))
csvout(ROOT/'part1/binary_accuracy_replicates.csv',out)
summary=[]
for model in ['F','HF']:
    for name in ['Full','INT','ICS','W','ID','R']:
        group=[r for r in out if r['model']==model and r['basis']==name]
        summary.append(dict(model=model,basis=name,repeats=len(group),
            accuracy_pp=float(np.mean([r['accuracy_pp'] for r in group])),
            accuracy_delta_pp=float(np.mean([r['accuracy_delta_pp'] for r in group])),
            accuracy_std_pp=float(np.std([r['accuracy_pp'] for r in group]))))
csvout(ROOT/'part1/binary_accuracy.csv',summary)
print('Ordinary and balanced binary accuracies computed; all historical bAcc cells exactly reproduced.')
