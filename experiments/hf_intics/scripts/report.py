"""Measured stage/domain results and same-seed differences to HF."""
import csv
import json
from pathlib import Path
Q=Path(__file__).resolve().parents[1]
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
DOMAINS=['overcast','rainy','dawn_dusk','night']
METRICS=['HOTA','DetA','IDF1','AssA']

def read(path):
    return list(csv.DictReader(path.open()))
def write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

rows=[]
for stage in range(1,5):
    for domain in DOMAINS[:stage]:
        cell=Q/'results'/('stage%d'%stage)/domain
        if (cell/'complete.json').exists():rows+=read(cell/'metrics.csv')
reference=read(HF/'results/stage_domain_metrics.csv')
index={(int(r['stage']),r['eval_domain'],r['category']):r for r in reference}
write(Q/'results/stage_domain_metrics.csv',rows)
paired=[]
for r in rows:
    old=index[int(r['stage']),r['eval_domain'],r['category']]
    assert int(r['GT_count'])==int(old['GT_count']),'HF evaluation pool differs'
    v={k:r[k] for k in ['stage','eval_domain','category']}
    v['seed']=1790355234
    for metric in METRICS:
        key=metric+'_percent';v['HF_'+metric]=float(old[key]);v['INTICS_'+metric]=float(r[key])
        v['delta_'+metric]=float(r[key])-float(old[key])
    paired.append(v)
write(Q/'results/paired_to_HF.csv',paired)
full=len(rows)==70
statistics=[]
if full:
    for label,source in [('HF',reference),('INTICS',rows)]:
        for category in ['OVERALL','car']:
            for metric in METRICS:
                cells={(int(r['stage']),r['eval_domain']):float(r[metric+'_percent'])
                    for r in source if r['category']==category}
                diagonal=[cells[i,DOMAINS[i-1]] for i in range(1,5)]
                f=[max(cells[i,DOMAINS[j-1]] for i in range(j,4))-cells[4,DOMAINS[j-1]]
                    for j in range(1,4)]
                statistics.append(dict(model=label,category=category,metric=metric,
                    FinalMean=sum(cells[4,d] for d in DOMAINS)/4,
                    DiagonalMean=sum(diagonal)/4,
                    AIP=sum(sum(cells[i,d] for d in DOMAINS[:i])/i for i in range(1,5))/4,
                    AvgF=sum(f)/3,FinalOld=sum(cells[4,d] for d in DOMAINS[:3])/3,
                    BWT=sum(cells[4,DOMAINS[j-1]]-diagonal[j-1] for j in range(1,4))/3))
    write(Q/'results/continual_metrics.csv',statistics)
    differences=[]
    for r in [x for x in statistics if x['model']=='INTICS']:
        old=next(x for x in statistics if x['model']=='HF' and x['category']==r['category'] and x['metric']==r['metric'])
        differences.append(dict(category=r['category'],metric=r['metric'],
            **{k:r[k]-old[k] for k in ['FinalMean','DiagonalMean','AIP','AvgF','FinalOld','BWT']}))
    write(Q/'results/paired_continual_deltas.csv',differences)
gatefile=Q/'episode/stage1_gate.json'
gate=json.loads(gatefile.read_text()) if gatefile.exists() else None
state='complete' if full else ('stopped at stage1 gate' if gate and gate['status']=='stop' else 'in progress')
clarification_file=Q/'manifests/angle_clarification.json'
clarification=json.loads(clarification_file.read_text()) if clarification_file.exists() else None
if clarification and clarification['status']=='pending':
    state='awaiting angle-definition clarification; stopped under the initially registered PC1/PC1 rule'
lines=['# HF + online intervention-residual ICS','', 'Status: **'+state+'**. Single training seed **1790355234**, identical to HF.',
    '', 'Baseline: '+str(HF)+'. Clear source S → Overcast → Rainy → Dawn/Dusk → Night, five epochs/domain, four V100s, FP32, 16 samples/GPU, two workers/GPU. Geometry, student low-frequency transform, contrastive high-frequency noise, APR and beta=1 coupling are retained. Only the requested ICS settings change; config_diff.json lists every output-path and model difference.',
    '', 'Online update: normalized student key-positive embedding minus normalized clean student embedding at exactly the same boxes; detached residuals, no mean subtraction or residual renormalization, covariance EMA .98, shrinkage0, two PCs, HF PC EMA .9 followed by QR, alpha .25, loss weight .1, min_samples32. The APR basis reader and trim operator are unchanged.',
    '', 'Preflight:', '']
if full:
    new_hota=next(r for r in statistics if r['model']=='INTICS' and r['category']=='OVERALL' and r['metric']=='HOTA')
    old_hota=next(r for r in statistics if r['model']=='HF' and r['category']=='OVERALL' and r['metric']=='HOTA')
    night=next(r for r in paired if str(r['stage'])=='4' and r['eval_domain']=='night' and r['category']=='OVERALL')
    lines[4:4]=['Observed paired result: final OVERALL four-domain mean HOTA %.6f vs HF %.6f (delta %+.6f). Night HOTA changes by %+.6f; the three-old-domain HOTA mean changes by %+.6f. HOTA AvgF is %.6f vs HF %.6f (delta %+.6f; larger means more forgetting). The final four-domain average does not improve on this single paired seed.'%
        (new_hota['FinalMean'],old_hota['FinalMean'],new_hota['FinalMean']-old_hota['FinalMean'],
         night['delta_HOTA'],new_hota['FinalOld']-old_hota['FinalOld'],
         new_hota['AvgF'],old_hota['AvgF'],new_hota['AvgF']-old_hota['AvgF']),'']
if clarification and clarification['status']=='pending':
    lines += ['The 45° condition requires clarification before further training or final publication. The initially registered comparison is bank PC1/offline PC1 (82.480923°), which stopped the process. The principal angle from bank PC1 to the full offline2D subspace is33.756089°; both values were independently verified with scipy.linalg.subspace_angles. The two2D principal angles are33.453465°/88.357491°. See manifests/angle_clarification.json and probe/stage1/angle_definition_check.json. [SciPy defines the comparison in terms of the input column spaces](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.subspace_angles.html). All first-stage results and saved states remain available; no post-stage1 training or final publication has occurred.','']
elif clarification and clarification['status']=='resolved_continue':
    lines += ['User-authorized gate correction: the 45° condition compares bank PC1 with the full offline two-dimensional FAP subspace. Its principal angle is33.756089°, so stage1 passes together with the HOTA delta−0.307201. The original PC1/PC1 diagnostic82.480923° caused an earlier stop and remains recorded; the user explicitly chose the subspace definition and authorized continuation after seeing both values. Original stop/protocol/validation evidence is archived at '+clarification['archive_relative_path']+'. The same single-seed run continues from the complete stage1 boundary; no stage1 retraining or basis/strength retuning. Details: manifests/angle_clarification.json and manifests/stage_boundary_recovery.json.','']
for name in ['U1','U2','U3']:
    p=Q/'preflight'/(name+'.json')
    if p.exists():
        v=json.loads(p.read_text())
        if name=='U1':lines.append('- U1: '+v['status']+'; total and all component losses bitwise equal, '+str(v['modified'])+'.')
        elif name=='U2':lines.append('- U2: principal angles '+', '.join('%.6f°'%x for x in v['principal_angles_deg'])+'; reported without an angle threshold.')
        else:
            new=v['records']['INTICS'];old=v['records']['HF']
            lines.append('- U3: '+v['status']+'; paired disposable HF/new50-iteration runs, iteration-time increase %.4f%%.'%v['relative_iteration_time_increase_pct'])
            lines.append('  New key/ref actual nonidentity rates %.4f%%/%.4f%%; mean weighted loss_track_trim %.9f; first initialization by rank %s.'%
                (100*new['key_ratio'],100*new['ref_ratio'],new['loss_track_trim_mean'],
                 [s['first_initialized_iteration'] for s in new['summaries']]))
            lines.append('  Mean critical-rank iteration time on iterations11–50: HF %.6fs, INTICS %.6fs.'%
                (old['iteration_mean_last40'],new['iteration_mean_last40']))
            lines.append('  Training-step timing excluding batch fetch and the independent CSV write: HF %.6fs, INTICS %.6fs, signed change %+.4f%%. HF logs were local and candidate logs were on NAS; these are single paired timing measurements.'%
                (old['compute_mean_last40'],new['compute_mean_last40'],v['relative_compute_time_increase_pct']))
lines += ['', 'The original HF metadata uses cumulative teacher scale_factor and local student scale_factor=1; geometry assertions compare shapes, flips, exact box coordinates and the equivalent composed scale, while retaining the original metadata. Both views remain on identical pixels/geometry. Fixed-batch cache is disposable and stored in this task’s /dev/shm cache symlink.',
    '', 'Stage1 gate (user-clarified angle definition): OVERALL HOTA delta <=−0.5, or the principal angle between bank PC1 and the offline FAP top2 subspace >45°. Offline directions use only fit videos, normalized paired FAP/clean residuals without centering. The initial gate inference ran while the four training ranks retained live model/RNG state; the earlier axis-based stop and subsequent authorized stage-boundary recovery are disclosed below.', '']
if gate:
    lines += ['Stage1 decision: **'+gate['status'].upper()+'**; '+
        ('; '.join(gate['reasons']) if gate['reasons'] else 'both conditions passed under the user-clarified angle definition')+'.',
        'Overcast OVERALL HOTA %.6f vs HF %.6f (delta %+.6f). Bank PC1/offline PC1 angle %.6f°; PC1/offline2D angle %.6f°; two subspace principal angles %s.'%
        (gate['HOTA'],gate['HF_R11_HOTA'],gate['HOTA_delta'],gate['first_direction_angle_deg'],
         gate['first_direction_to_offline_span_deg'],gate['principal_angles_deg']),
        'Raw decision: episode/stage1_gate.json.','']
for probe_stage in [1,4]:
    p1file=Q/'probe'/('stage%d'%probe_stage)/'P1.json'
    if not p1file.exists():continue
    p1=json.loads(p1file.read_text())
    lines += ['Stage%d P1 energies use test videos; offline FAP directions use fit videos only.'%probe_stage,
        '', '| Bank directions | Cross-domain shift energy | Identity energy |',
        '|---|---:|---:|',
        '| Both PCs | %.4f%% | %.4f%% |'%(100*p1['P1_weather_shift_energy_share'],100*p1['P1_identity_energy_share'])]
    for v in p1.get('P1_per_direction',[]):
        lines.append('| PC%d | %.4f%% | %.4f%% |'%(v['bank_row'],
            100*v['weather_shift_energy_share'],100*v['identity_energy_share']))
    if probe_stage==4:
        lines += ['Stage4 bank PC1/offline PC1 angle %.6f°; PC1/offline2D angle %.6f°; two subspace principal angles %s.'%
            (p1['first_direction_angle_deg'],p1['first_direction_to_offline_span_deg'],p1['principal_angles_deg'])]
    lines += ['']
lines += ['| Stage | Domain | Category | HOTA | DetA | IDF1 | AssA | ΔHOTA vs HF |',
    '|---:|---|---|---:|---:|---:|---:|---:|']
for r in paired:
    if r['category'] in ['OVERALL','car']:
        lines.append('| %s | %s | %s | %s | %+.6f |'%(r['stage'],r['eval_domain'],r['category'],
            ' | '.join('%.6f'%r['INTICS_'+m] for m in METRICS),r['delta_HOTA']))
if full:
    lines += ['', 'Final scores use R4j; diagonals use Rjj. AIP averages the four seen-domain stage averages. AvgF = mean over old domains of max(Rij, i=j..3)−R4j, without clipping. FinalOld averages R4j for the three old domains. BWT averages R4j−Rjj. Full precision and paired deltas: results/continual_metrics.csv and results/paired_continual_deltas.csv.','']
    lines += ['| Model | Category | Metric | Final mean | Diagonal mean | AIP | AvgF | FinalOld | BWT |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in statistics:
        lines.append('| %s | %s | %s | %s |'%(r['model'],r['category'],r['metric'],
            ' | '.join('%.6f'%r[k] for k in ['FinalMean','DiagonalMean','AIP','AvgF','FinalOld','BWT'])))
    lines += ['']
    lines += ['| Category | Metric | ΔFinal mean | ΔDiagonal mean | ΔAIP | ΔAvgF | ΔFinalOld | ΔBWT |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in differences:
        lines.append('| %s | %s | %s |'%(r['category'],r['metric'],
            ' | '.join('%+.6f'%r[k] for k in ['FinalMean','DiagonalMean','AIP','AvgF','FinalOld','BWT'])))
else:lines += ['', 'Missing later stages/cells have no result; four-domain final/AIP/AvgF/FinalOld/BWT are not fabricated.','']
lines += ['Probe results: [stage1 P1](probe/stage1/P1.json), and (if stage4 is reached) [stage4 P1](probe/stage4/P1.json), [Table1 retrieval](probe/stage4/evidence/part1/INTICS/cross_view_retrieval.csv), [Table2b deletion](probe/stage4/evidence/part1/INTICS/deletion.csv) and [FigureC crops](probe/stage4/evidence/part1/INTICS/figures/C_INT_crops.pdf). Bank diagnostics read the rank0 student bank from the frozen checkpoint; all four rank-local runtime states are retained as in HF. The actual bank_pcs.npy and offline_fap_k2.npy with SHA256 hashes accompany each P1 report. Random-basis probe controls use the historical fixed seed0; they are not additional training seeds.','']
evidence_root=Q/'probe/stage4/evidence/part1/INTICS'
if (evidence_root/'complete.json').exists():
    lines += ['Evidence definitions: P1 angles compare the bank with the fit-video FAP-only offline basis. The evidence package INT basis combines FAP/photo/student residuals; its ICS basis is the actual frozen bank. FigureC shows extrema of the offline three-view INT first axis, with ten distinct tracks at each end.','',
        '| Table1 view | Cross-view identity mAP | Ratio to clean |',
        '|---|---:|---:|']
    for r in read(evidence_root/'cross_view_retrieval.csv'):
        lines.append('| %s | %.6f | %.6f |'%(r['view'],float(r['mAP_pp']),float(r['ratio'])))
    lines += ['', '| Table2b deleted basis | Weather balanced accuracy | Night/Clear balanced accuracy | Identity mAP | ΔIdentity mAP | Fisher drop |',
        '|---|---:|---:|---:|---:|---:|']
    for r in read(evidence_root/'deletion.csv'):
        lines.append('| %s | %.6f | %.6f | %.6f | %+.6f | %.6f%% |'%
            (r['basis'],float(r['weather_bacc_pp']),float(r['night_clear_bacc_pp']),
             float(r['id_mAP_pp']),float(r['id_mAP_pp_delta']),float(r['fisher_drop_pct'])))
    deletion=read(evidence_root/'deletion.csv')
    bank_deletion=next(r for r in deletion if r['basis']=='ICS')
    int_deletion=next(r for r in deletion if r['basis']=='INT')
    evidence_gates=json.loads((evidence_root/'gates.json').read_text())
    lines += ['', 'Bank interpretation (inference from these probes): the final online bank has partial intervention alignment and retains identity information. Deleting its two directions changes identity mAP by %+.6f points and reduces weather Fisher by %.6f%%; deleting the offline three-view INT directions changes identity mAP by %+.6f points and reduces Fisher by %.6f%%. Historical evidence checks: cross-view retrieval %s, INT deletion %s, ICC %s (INT−ID %.6f, required >=0.3). These probe checks describe the measured representation; the requested training stop was evaluated at stage1.'%
        (float(bank_deletion['id_mAP_pp_delta']),float(bank_deletion['fisher_drop_pct']),
         float(int_deletion['id_mAP_pp_delta']),float(int_deletion['fisher_drop_pct']),
         'passes' if evidence_gates['cross_view_all_pass'] else 'fails',
         'passes' if evidence_gates['deletion_pass'] else 'fails',
         'passes' if evidence_gates['ICC_pass'] else 'fails',evidence_gates['ICC_INT_minus_ID'])]
    lines += ['']
early = []
for stage in [2,3]:
    p = Q/'manifests'/('early_frozen_stage%d.json'%stage)
    if p.exists():
        info = json.loads(p.read_text())
        early.append(dict(stage=stage,status=info['status']))
if early:
    lines += ['Evaluation scheduling: immutable completed-stage snapshots were evaluated on additional wholly idle GPUs while the same four-GPU training continued, with at most two inference jobs concurrently. The final controller reuses complete cells and verifies their checkpoint hashes. Records: '+json.dumps(early)+'. Training settings and the mandatory stage1 gate were retained.','']
recoveries=[]
for stage in range(1,5):
    p=Q/'manifests'/('stage%d_dataset.json'%stage)
    if p.exists():
        info=json.loads(p.read_text())
        if info.get('resume_epoch',0):recoveries.append(dict(stage=stage,epoch=info['resume_epoch']))
boundary_file=Q/'manifests/stage_boundary_recovery.json'
if boundary_file.exists():
    boundary=json.loads(boundary_file.read_text())
    verified=[]
    for rank in range(4):
        p=Q/'manifests'/('stage_boundary_restore_rank%d.json'%rank)
        if p.exists() and json.loads(p.read_text())['status']=='passed':verified.append(rank)
    lines += ['Runtime deviation: user-authorized continuation from the complete stage1 checkpoint and all four final rank states after correcting the gate interpretation. Restoration audits verify model parameters, rank-local buffers, APR runtime state, counters and Python/NumPy/Torch/CUDA RNGs before any stage2 update. Verified ranks at report time: '+str(verified)+'. New-domain workers and optimizer are created by the unchanged HF stage runner; FAP audit call counters are also restored. Numerical equivalence to uninterrupted training is not claimed. No additional seed or stage1 retraining. Original records and approval are retained in manifests/stage_boundary_recovery.json and the stop archive.','']
if recoveries:
    lines += ['Runtime deviation: epoch-boundary recovery occurred: '+json.dumps(recoveries)+
        '. Rank-local banks, prototypes and saved RNGs were restored; data-loader worker RNG recovery is not bit-exact.','']
elif not boundary_file.exists():
    lines += ['No epoch-boundary recovery is recorded. Requested ICS changes, geometric-metadata equivalence checks, disposable preflight weights and the stage1 pause are disclosed above; all other HF training settings remain unchanged.','']
placement=Q/'manifests/NAS_migration.json'
if placement.exists():
    lines += ['Storage deviation: the user authorized NAS storage after the local filesystem filled during disposable preflight. The entire new task was copied to its private NAS namespace and all 1138 original files/links were verified before replacing the logical output directory with a symlink. The original HF checkpoint retention and stage-endpoint hard links are retained; NAS hard-link/rename/readback checks passed. The absent shared NAS mount is replaced only through this task’s input image aliases to the same NAS objects/local cache; annotations, video/frame selection, geometry, batch and workers are unchanged. Full paths and hashes: manifests/NAS_migration.json and manifests/input_storage_aliases.json. Completed HF50 preflight is reused; the incomplete new-mode attempt is archived and restarted from source S with the same seed. HF preflight logged locally and the restarted candidate logs on NAS, so elapsed and compute timings are both reported; physical GPU IDs are recorded in job manifests. No formal checkpoint recovery occurred during this migration.','']
(Q/'SUMMARY.md').write_text('\n'.join(lines))
(Q/'manifests/result_status.json').write_text(json.dumps(dict(status=state,full=full,metric_rows=len(rows),
    evaluation_cells=len(rows)//7,seed=1790355234),indent=2)+'\n')
print('REPORT_READY',state,len(rows))
