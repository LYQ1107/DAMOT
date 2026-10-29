"""Stage completed outputs and explicit pending status; never stage features/weights."""
import csv,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
from protocol import write_json
ROOT=Path(__file__).resolve().parents[1];PUB=Path('/data1/liuyeqiang/damot_ics_followup_publish_20261029');DEST=PUB/'experiments/ics_followup_20261029'
BASE='ebc3190b2fa82d9e117315998c0d7435033e113c'
def rows(path):return list(csv.DictReader(path.open())) if path.exists() else []
def copy(src,dst):
    if src.is_dir():
        for f in src.rglob('*'):
            if f.is_file() and f.suffix not in ['.npz','.pth','.pkl','.pyc'] and not f.name.endswith('.pkl.gz'):copy(f,dst/f.relative_to(src))
    else:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(str(src),str(dst))
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    analysis=[]
    for folder in (ROOT/'results').glob('analysis_*'):
        if (folder/'summary.json').exists():
            copy(folder,PUB/'experiments/ics_probe'/folder.name);analysis.append(folder.name)
    copy(ROOT/'results/S1',DEST/'S1_config_audit')
    copy(ROOT/'results/S2_covariance_pcs_check.json',DEST/'S2_covariance_pcs_check.json')
    copy(ROOT/'results/drift_decomposition',PUB/'experiments/ics_probe/drift_decomposition')
    for p in [ROOT/'code/tools/analysis/ics_probe_analyze.py',ROOT/'code/tools/analysis/ics_probe_drift.py']:copy(p,PUB/'tools/analysis'/p.name)
    for p in [ROOT/'scripts/protocol.py',ROOT/'scripts/validate_followup.py',ROOT/'scripts/report_drift.py',ROOT/'scripts/analyze_group.py']:copy(p,DEST/'scripts'/p.name)
    for name in ['S2_F_v2.log','S3_F.log','numerical_validation.log','report_drift.log','extract_HF_stage4_e5_failed_nas.log','S1_HF_driver_failed_nas.log']:
        if (ROOT/'logs'/name).exists():copy(ROOT/'logs'/name,DEST/'logs'/name)
    for name in ['numerical_validation.json','S5_common_history.json','S5_model_preflight.json']:
        if (ROOT/'manifests'/name).exists():copy(ROOT/'manifests'/name,DEST/'provenance'/name)
    copy(ROOT/'manifests/user_request.txt',DEST/'user_request.txt')
    status=dict(run_directory=str(ROOT),code_base_commit=BASE,completed_analyses=analysis,S4='pending full-domain baseline identity checks and interventions',S5='prepared only; no new short training yet; waiting for NOGEOM controller_complete.json',NOGEOM='original run resumed after transient NAS outage; never treated as complete because GPUs were temporarily idle',updated_UTC=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()))
    write_json(DEST/'status.json',status)
    (DEST/'README.md').write_text('''# ICS follow-up, 2026-10-29\n\nThis is a staged delivery. See `status.json` for completed and pending work. A pending experiment has no result.\n\nCompleted F analysis is in `../ics_probe/analysis_F_stage4_e5_v2/` and `../ics_probe/drift_decomposition/`. The original analysis remains intact. All numbers are measured; no expected outcome is substituted.\n\nS2's covariance/PCS equivalence gate failed at every F checkpoint. At stage4 the largest principal angle is 1.751934 degrees, above the unchanged 1 degree threshold. The user approved reporting the actual stored `bank_pcs` separately from covariance eigenspace scans. Both families remain in the CSV and LaTeX tables.\n\nThe Fisher trace ratio uses unrenormalized projected vectors on car/test RoIs. Weather classifier inputs retain the earlier L2-normalization convention, and logistic probes are refitted after each intervention. Random k-dimensional results average ten bases. Weather-oracle rank is at most four.\n\nS3 uses F stage1 fit videos as the common basis reference, literally as requested. No cross-model coordinate alignment is performed. Drift fractions are means of per-RoI energy fractions; absolute in/out energies and ratios of sums are exported separately. Cross-stage retrieval is a proxy, not measured tracking forgetting. The fixed figure shows stage3→4 old-domain energy and the equal mean of the three stage4←j retrieval cases; every other transition and dimension is retained in CSV.\n\nTwo protocol details are explicitly disclosed: the stored bank is not the exact eigenspace of the saved covariance; and HF saved exact APR histories in `stage3_final_rank*.pth`, despite lacking a `<stage>_roi_proto.pth` filename. S5 will use the same exact rank0 history file for all groups rather than approximate reconstruction. Its hash is in provenance.\n\nSource data, features, and checkpoints remain on the server. No `.npz` or model weights are committed. `.npy` files contain the requested small projection bases only.\n''')
    p=ROOT/'results/analysis_F_stage4_e5_v2';energy=rows(p/'csv/P1_energy.csv');e=energy[0];binary=rows(p/'csv/P2_binary.csv')[0]
    text=['# DAMOT experiments, 2026-10-29','',f'Execution directory: `{ROOT}`. Probe seed: 0. Training seed for prepared S5: 1790355234. Code baseline: `{BASE}`; exact analysis changes are in this file\'s publication commit. F checkpoints: `/data1/liuyeqiang/damot_rebuttal_20261026/cache/weights/F_stage{{1..4}}_e5.pth`.','', '**Staged delivery:** F S2/S3 and HF/NOGEOM configuration audit are complete. HF/NOGEOM probe groups, S4 tracking interventions, and S5 short training are pending. No new four-domain training is started by this follow-up.','', '## Table C — subspace properties','', '| Model | Domain shift energy % | Identity energy % | Weather/Identity | Angles to weather oracle (degrees) | Night/clear bAcc % |','|---|---:|---:|---:|---|---:|',f'| F, actual bank_pcs | {100*float(e["Real weather shift"]):.3f} | {100*float(e["Identity"]):.3f} | {float(e["Weather/Identity"]):.3f} | 54.818 / 84.594 | {100*float(binary["night_vs_clear_bacc"]):.3f} |']
    for model in ['HF','NOGEOM','S5-a','S5-b','S5-c','S5-d']:text.append('| '+model+' | pending | pending | pending | pending | pending |')
    text+=['','Source: `ics_probe/analysis_F_stage4_e5_v2/csv/P1_energy.csv`, `P1_angles.csv`, `P2_binary.csv`. Same execution directory, checkpoints, seed and code baseline as above. Domain shift uses unpaired BDD domains and is not pure weather.','','## Table D — negative-control k scan','', 'Fixed slice: beta=1; every beta=0/.5/1 cell and Fisher ratio is in the full CSV. ID mAP values are percentages. Covariance eigenvectors and stored PCS are separate.','','| F basis | k requested | k effective | ID mAP % | Weather bAcc % | Raw Fisher |','|---|---:|---:|---:|---:|---:|']
    for r in rows(p/'csv/P3_intervention.csv'):
        if float(r['beta'])==1:text.append('| %s | %s | %s | %.3f | %.3f | %.6f |'%(r['subspace'],r['k_requested'],r['k_effective'],100*float(r['id_mAP']),100*float(r['weather_bacc']),float(r['weather_fisher'])))
    text+=['','Source: `ics_probe/analysis_F_stage4_e5_v2/csv/P3_intervention.csv` (78 cells, random10). F stage4 epoch5, seed0, run and code baseline above. The stored/covariance equivalence check fails (max angle 1.751934 degrees).','','## Table E — drift attribution','', 'Fixed slice: stage4 queries against each of the three older-domain galleries; equal domain mean. Values are percentage points of same-stage minus cross-stage mAP.','','| F restoration | Dimensions | Remaining forgetting proxy (pp) | Reduction (pp) |','|---|---:|---:|---:|']
    rr=rows(ROOT/'results/drift_decomposition/F/cross_stage_retrieval.csv');rr=[r for r in rr if r['query_stage']=='4'];base=sum(float(r['forgetting_proxy']) for r in rr)/3
    text.append('| None | — | %.4f | 0 |'%(100*base))
    restorations=rows(ROOT/'results/drift_decomposition/F/drift_restoration.csv')
    for name in ['FAP residual PCA','Identity oracle']:
        for k in [2,8,16]:
            selected=[r for r in restorations if r['query_stage']=='4' and r['basis']==name and int(r['k_requested'])==k]
            rem=sum(float(r['forgetting_restored']) for r in selected)/3;text.append('| %s | %d | %.4f | %.4f |'%(name,k,100*rem,100*(base-rem)))
    text+=['','Source: `ics_probe/drift_decomposition/`, including full CSV, LaTeX and the two-panel figure. Common basis reference F_stage1 fit videos, test statistics, seed0, F_stage1–4 checkpoints; run and code baseline above. Identity restoration helps more at two dimensions. Both identity16 and FAP16 restoration help, so a strict exclusive attribution is not established.','','## Table F — inference-time tracking interventions','', '| Checkpoints | Domains | Bases / beta | HOTA / DetA / IDF1 / AssA |','|---|---|---|---|','| F stage4, HF stage4 | all four target domains | stored ICS2, clear-fit FAP2, identity16, random2; .5/1 | pending beta0 full-domain consistency gate |','','Run: `'+str(ROOT)+'/results/S4/`. Checkpoints F_stage4_e5 and HF stage4 epoch5; basis seed9973, fit split seed0; code baseline above. No numbers are reported before native beta0 and archived metrics agree.','','## Table G — Night-only 2×2, two epochs','', '| Group | Estimator | APR coupling | Performance / forgetting / subspace |','|---|---|---|---|','| a | proto_shrinkage_cov_ema | true | prepared; pending training |','| b | proto_shrinkage_cov_ema | false | prepared; pending training |','| c | pairwise_residual | true | prepared; pending training |','| d | pairwise_residual | false | prepared; pending training |','','Run: `'+str(ROOT)+'/results/S5/HF_{a,b,c,d}`. Shared starting checkpoint HF stage3 epoch5, SHA256 `0dbc117207f248eae2632bffe382d38dc0e0cc67409d44986399ae67a2a9090b`; seed1790355234; prepared source/model checks pass (752/752 keys, bank updates660 inherited, minimum96 unchanged). Exact APR history SHA256 `a8c2ea74e8d0d9aed6e069f9a7cb338bd0f39599a2e9f455dc07f8a6d2cf74bc`. Code baseline above. S5 waits for NOGEOM completion and four idle GPUs.','','The complete result directory, per-item configuration differences, effective pipelines, numerical checks and preserved failure logs accompany this staged delivery.']
    (PUB/'experiments/SUMMARY_20261029.md').write_text('\n'.join(text)+'\n')
    # Hash every delivered file; the hash list deliberately excludes itself.
    files=[]
    roots=[DEST,PUB/'experiments/ics_probe/analysis_F_stage4_e5_v2',PUB/'experiments/ics_probe/drift_decomposition']
    for r in roots:
        for f in sorted(r.rglob('*')):
            if f.is_file() and f.name!='SHA256SUMS':files.append(f)
    (DEST/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(PUB))+'\n' for f in files))
    print('Staged files',len(files),'analyses',analysis)
if __name__=='__main__':main()
