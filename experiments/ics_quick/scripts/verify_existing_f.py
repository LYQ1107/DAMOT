"""Audit the first F baseline without rerunning or discarding its results."""
from common import *
root=Q/'X2/F/overcast/beta0'
table={r['']:r for r in csv.DictReader((root/'summary_metrics.csv').open())}
ref=[r for r in csv.DictReader((RUN/'results/exp1/stage_domain_metrics.csv').open()) if r['model']=='F' and r['stage']=='4' and r['eval_domain']=='overcast']
checks=[];rows=[]
for cat in CLASSES+('OVERALL',):
    row=dict(model='F',domain='overcast',beta=0,category=cat,checkpoint=INPUTS['F_stage4'],checkpoint_sha256=sha(INPUTS['F_stage4']),frames=4448,videos=22)
    for k in ['HOTA','DetA','IDF1','AssA']:row[k+'_percent']=float(table[cat][k])
    rows.append(row)
for r in ref:
    for k in ['HOTA','DetA','IDF1','AssA']:
        val=float(table[r['category']][k]);old=float(r[k+'_percent']);checks.append(dict(category=r['category'],metric=k,observed=val,reference=old,delta=val-old,passed=abs(val-old)<1e-6))
write_csv(root/'metrics.csv',rows);write_csv(root/'baseline_parity.csv',checks)
dump(root/'baseline_mismatch.json',dict(passed=all(r['passed'] for r in checks),checks=checks,reason='Historical F baseline not reproduced exactly; no beta1 accepted or run for F',historical_cudnn_benchmark=True,first_attempt_cudnn_benchmark=False))
print(checks,flush=True)
