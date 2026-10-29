"""Publish complete F tracking and complete F/HF offline blocks, marking the rest pending."""
import csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PUB=Path('/data1/liuyeqiang/damot_ics_followup_publish_20261029');DEST=PUB/'experiments/ics_followup_20261029'
sys.path.insert(0,str(ROOT/'code/tools/run'))
from apr_complement_experiment import Config
from protocol import write_views,write_json
from stage_runtime_publication import selected,copy
from collect_final import markdown,n,configuration
def read(p):return list(csv.DictReader(Path(p).open()))
def write(p,rows):
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert (ROOT/'manifests/analysis_HF_complete.json').exists()
    assert (ROOT/'results/S4/F/native_projection_validation.json').exists()
    subprocess.check_call([str(ROOT/'env/bin/python'),str(ROOT/'scripts/report_drift.py')])
    for name in ['analysis_F_stage4_e5_v2','analysis_HF_stage4_e5','analysis_HF_stage4_e5_v2']:selected(ROOT/'results'/name,PUB/'experiments/ics_probe'/name)
    selected(ROOT/'results/drift_decomposition',PUB/'experiments/ics_probe/drift_decomposition');selected(ROOT/'results/S1',DEST/'S1_config_audit')
    froot=ROOT/'results/S4/F';cfg=Config.fromfile(str(configuration('F')));allrows=[]
    tags=['baseline']+['%s_beta%g'%(b,v) for b in ['ICS_stored_m2','FAP_clear_fit_m2','Identity_fit_m16','Random_m2'] for v in [.5,1.]]
    means=[]
    for tag in tags:
        current=[]
        for domain in ['overcast','rainy','dawn_dusk','night']:
            folder=froot/tag/domain;assert (folder/'complete.json').exists();write_views(cfg,folder/'resolved_views.txt',dict(inference='exact FP32 cache replay through native tracker',projection='unrenormalized; ICS basis uses QR span of stored PCS'))
            current+=read(folder/'metrics.csv')
        allrows+=current
        for category in ['car','OVERALL']:
            rows=[r for r in current if r['category']==category];assert len(rows)==4
            row=dict(checkpoint='F',setting=tag,category=category)
            for metric in ['HOTA','DetA','IDF1','AssA']:
                row[metric+'_percent']=sum(float(r[metric+'_percent']) for r in rows)/4;row['delta_'+metric+'_pp']=sum(float(r['delta_'+metric+'_pp']) for r in rows)/4
            means.append(row)
    write(froot/'all_domains.csv',allrows);write(froot/'four_domain_means.csv',means);selected(froot,DEST/'S4/F')
    for model in ['F','HF']:
        for domain in ['overcast','rainy','dawn_dusk','night']:
            for filename in ['load.json','complete.json','resolved_config.py']:copy(ROOT/'cache/S4'/model/domain/filename,DEST/'S4'/model/'capture_metadata'/domain/filename)
    selected(ROOT/'logs',DEST/'logs')
    for folder in (ROOT/'results/feats').iterdir():
        if folder.is_dir() and folder.name.startswith(('F_','HF_')):
            for f in folder.glob('*meta.json'):copy(f,DEST/'metadata'/folder.name/f.name)
    for f in (ROOT/'scripts').glob('*.py'):copy(f,DEST/'scripts'/f.name)
    for f in (ROOT/'code/tools/run').glob('ics_*.py'):copy(f,PUB/'tools/run'/f.name)
    for f in (ROOT/'manifests').glob('*.json'):
        if not f.name.startswith('publication_'):copy(f,DEST/'provenance'/f.name)
    C=[];D=[]
    for model in ['F','HF']:
        p=ROOT/'results'/('analysis_%s_stage4_e5_v2'%model);e=next(r for r in read(p/'csv/P1_energy.csv') if r['subspace']=='ICS (checkpoint)');a=next(r for r in read(p/'csv/P1_angles.csv') if r['subspace']=='Oracle weather (labels)')
        C.append([model,n(100*float(e['Real weather shift']),2),n(100*float(e['Identity']),2),n(e['Weather/Identity']),n(a['angle1'],2)+' / '+n(a['angle2'],2)])
        for r in read(p/'csv/P3_intervention.csv'):
            if float(r['beta'])==1:D.append([model,r['subspace'],r['k_requested']+'/'+r['k_effective'],n(100*float(r['id_mAP'])),n(100*float(r['weather_bacc'])),n(r['weather_fisher'],6)])
    E=[]
    for model in ['F','HF']:
        p=ROOT/'results/drift_decomposition'/model;e=next(r for r in read(p/'drift_energy.csv') if r['from_stage']=='3' and r['to_stage']=='4' and r['bucket']=='old' and r['basis']=='ICS stored bank_pcs');b=[r for r in read(p/'cross_stage_retrieval.csv') if r['query_stage']=='4'];rest=read(p/'drift_restoration.csv');vals=[]
        for basis,k in [('FAP residual PCA',2),('FAP residual PCA',16),('Identity oracle',2),('Identity oracle',16)]:vals.append(n(100*sum(float(r['forgetting_restored']) for r in rest if r['query_stage']=='4' and r['basis']==basis and int(r['k_requested'])==k)/3))
        E.append([model,n(e['mean_in_sq'],6),n(e['mean_out_sq'],6),n(100*sum(float(r['forgetting_proxy']) for r in b)/3)]+vals)
    content=['<!-- ICS_FOLLOWUP_BEGIN -->','# DAMOT ICS follow-up — measured results, staged delivery','',f'Run: `{ROOT}`. Probe/split seed0, S4 random seed9973, prepared S5 seed1790355234. Runtime source is delivered in the repository; per-job source fingerprints are in `ics_followup_20261029/provenance/`. Starting method commit `1d804fb036631f0fca048e93c58a5143ab93e030` plus the source updates in this delivery.','', '**Complete:** F/HF S1–S3 and the full 36-cell F tracking grid. **Pending:** NOGEOM probes, HF tracking grid (waiting for an idle GPU), and the four Night-only short runs (wait for the original NOGEOM controller to finish). No new full four-domain training is launched.','','## Table C — actual stored subspace','',markdown(['Model','Domain energy %','Identity energy %','Weather/Identity','Weather-oracle angles °'],C+[ ['NOGEOM / S5-a…d','pending','pending','pending','pending'] ]),'','Checkpoints F/HF stage4 epoch5; full P1/P2/binary/confusion outputs in `ics_probe/analysis_*_v2/`. Both angles are reported. Run, seed and source above; checkpoint paths/hashes and all24 per-model probe metadata files are retained. Stored PCS spans are QR-orthonormalized for geometric projections, without replacing them by covariance eigenvectors. BDD offsets are unpaired domain shifts.','','## Table D — negative-control k scan','', 'Display beta=1; every beta=0/.5/1 and ten-random-basis mean/SD is retained in each full CSV. Covariance and actual stored PCS are separate (max stage4 principal angle F:1.751934°, HF:1.830549°; unchanged threshold <1°).','',markdown(['Model','Basis','k/rank','ID mAP %','Weather bAcc %','Raw Fisher'],D),'','F/HF stage4 epoch5; car/test, seed0. Fisher uses the projected unit-normalized Z before another row normalization; classifier convention is unchanged. Run/source above; full 78-cell scans per model in their CSV and LaTeX.','','## Table E — drift and attribution','',markdown(['Model','ICS2 in drift','ICS2 out drift','Baseline forgetting pp','FAP2 restore pp','FAP16 restore pp','ID2 restore pp','ID16 restore pp'],E),'','Fixed slice: stage3→4 old-domain test RoIs for absolute mean squared drift; stage4←j equal mean over the three old domains for retrieval. All adjacent/1→4 transitions and dimensions2/8/16 are delivered in `ics_probe/drift_decomposition/`, with LaTeX and a two-panel figure. FAP/weather/ID use F stage1 fit videos; ICS uses the destination bank. F/HF stage1–4 epoch5, seed0, run/source above. HF changes high-noise and coupling, so this contrast alone does not isolate APR coupling. Wide FAP restoration also helps; an exclusive identity-versus-weather attribution is not established.','','## Table F — complete F inference interventions','', 'Equal mean of the four target domains. Parentheses are changes from the approved fresh F beta0, in percentage points. Full-domain and full-class metrics are in `ics_followup_20261029/S4/F/`; HF remains pending.','',markdown(['Model / class','Setting','HOTA (Δ)','DetA (Δ)','IDF1 (Δ)','AssA (Δ)'],[[r['checkpoint']+' / '+r['category'],r['setting']]+[n(r[m+'_percent'])+' ('+('%+.3f'%r['delta_'+m+'_pp'])+')' for m in ['HOTA','DetA','IDF1','AssA']] for r in means]),'', 'F stage4 epoch5; fit seed0, random seed9973, run/source above. Native beta0 equals cache replay for every frame; all112 fresh six-class/OVERALL metric checks pass. Enabled native head and replay also agree bit-for-bit on two predeclared first-Overcast-video tests (202 frames each). No feature renormalization is added. Old/new baseline differences, the failed gate, evaluator recheck and user approval are all delivered. FAP2 here is clear-fit only, unlike the all-fit FAP PCA used in P1 and the F-stage1 FAP reference in S3.','','## Table G — Night-only 2×2 short training','',markdown(['Group','Estimator','APR coupling','Status'],[['a','proto_shrinkage_cov_ema','true','prepared; pending'],['b','proto_shrinkage_cov_ema','false','prepared; pending'],['c','pairwise_residual','true','prepared; pending'],['d','pairwise_residual','false','prepared; pending']]),'', 'All groups use HF stage3 epoch5, seed1790355234, two Night epochs, the same exact saved rank0 APR history and corresponding fixed rank RNG snapshots. Bank660 is inherited, min_samples96 retained. Start/end counters, actual successful bank calls and every insufficient-pairs event are instrumented. Run/source above; `S5_common_history.json` records checkpoint/library/RNG hashes. GPU training has not started.','', '<!-- ICS_FOLLOWUP_END -->']
    doc=PUB/'experiments/SUMMARY_20261029.md';old=doc.read_text();start=old.index('<!-- ICS_FOLLOWUP_BEGIN -->');end=old.index('<!-- ICS_FOLLOWUP_END -->')+len('<!-- ICS_FOLLOWUP_END -->');doc.write_text(old[:start]+'\n'.join(content)+old[end:])
    write_json(DEST/'status.json',dict(run=str(ROOT),F_S1_S2_S3_S4='complete',HF_S1_S2_S3='complete',HF_S4='pending GPU',NOGEOM_S1_S2_S3='pending original run completion',S5='prepared, not trained',selection_by_metric=False))
    files=[f for base in [DEST,PUB/'experiments/ics_probe'] for f in sorted(base.rglob('*')) if f.is_file() and f.name!='SHA256SUMS']+[doc]
    assert not any(f.suffix in ['.npz','.pth','.pt','.ckpt'] for f in files)
    (DEST/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(PUB))+'\n' for f in files));print('Staged complete F/HF offline blocks and complete F tracking; remaining tasks explicitly pending')
if __name__=='__main__':main()
