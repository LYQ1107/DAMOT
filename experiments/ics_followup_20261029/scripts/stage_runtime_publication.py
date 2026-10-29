"""Stage runtime support and the approved S4 baseline deviation (no weights)."""
import csv,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PUB=Path('/data1/liuyeqiang/damot_ics_followup_publish_20261029');DEST=PUB/'experiments/ics_followup_20261029';HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(str(src),str(dst))
def selected(src,dst):
    for f in src.rglob('*'):
        if not f.is_file() or f.suffix in ['.npz','.pth','.pkl','.pyc'] or f.name.endswith('.pkl.gz') or f.name in ['track_coco.json','bbox_coco.json','segm_coco.json']:continue
        copy(f,dst/f.relative_to(src))
def main():
    core=['darth/models/track_heads/causal_quasi_dense_track_head.py','darth/models/utils/roi_causal_bank.py','darth/models/utils/roi_proto_memory.py']
    for rel in core:
        assert (ROOT/'code'/rel).read_bytes()==(HF/'code'/rel).read_bytes()
        copy(ROOT/'code'/rel,PUB/rel)
    for rel in ['darth/models/track_heads/__init__.py','darth/models/track_heads/inference_subspace_track_head.py','darth/datasets/eval/coco_eval_utils.py']:copy(ROOT/'code'/rel,PUB/rel)
    for name in ['apr_complement_experiment.py','evaluate_apr_complement.py','ics_short_stage.py','ics_short_stage_eval.py','ics_tracking_intervention.py','ics_validate_native_projection.py','ics_audit_historical_baseline.py']:copy(ROOT/'code/tools/run'/name,PUB/'tools/run'/name)
    for p in (ROOT/'scripts').glob('*.py'):copy(p,DEST/'scripts'/p.name)
    selected(ROOT/'results/S1',DEST/'S1_config_audit')
    for name in ['S4_F_fresh_baseline_authorization.json','hold_S4_F_resolved.json','S4_failed_baseline_inventory.json']:
        copy(ROOT/'manifests'/name,DEST/'provenance'/name)
    for name in ['fresh_vs_historical_baseline.csv']:
        copy(ROOT/'results/S4/F'/name,DEST/'S4/F'/name)
    for name in ['baseline_audit','baseline_gate_failure','historical_prediction_recheck']:selected(ROOT/'results/S4/F'/name,DEST/'S4/F'/name)
    for name in ['S4_F.log','S4_F_baseline_audit.log','S1_audit.log']:
        copy(ROOT/'logs'/name,DEST/'logs'/name)
    for label in ['HF_stage4_e5']:
        for f in (ROOT/'results/feats'/label).glob('*meta.json'):copy(f,DEST/'metadata'/label/f.name)
    status=json.loads((DEST/'status.json').read_text());status.update(S4='F fresh V100 beta0 approved by user; native/cache equality and fresh baseline metrics passed on all four domains; nonzero grid is running. Historical metrics and all differences preserved.',S5='prepared and checked on CPU; waiting for original NOGEOM completion and S4; only Night 2 epochs will execute')
    (DEST/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    readme=(DEST/'README.md').read_text()+'''\n## Approved F tracking baseline\n\nThe historical A100 F metrics are not exactly reproduced by new V100 inference. Re-evaluating the historical prediction file with the current evaluator exactly reproduces its HOTA/DetA/AssA, while the newly generated predictions differ slightly. The precise cause is not fully isolated; the historical environment differs and its checkpoint binary hash was not recorded at evaluation time.\n\nThe user explicitly approved using this run's native FP32 beta0 as the F intervention reference and publishing every old/new difference. See `S4/F/fresh_vs_historical_baseline.csv`, the audit, preserved failed gate and authorization. New beta0 replay is checked against every native frame and against all eight car/OVERALL reference metrics per domain. HF retains its own frozen V100 reference. Nonzero result aggregation remains pending until every cell completes.\n\n## Runtime source and short training\n\nThe causal head, bank and prototype-memory support files are unchanged copies of the already trained HF runtime. They are mirrored into the repository for checkpoint and APR-history compatibility. The short-stage entrypoint asserts resume stage4, two epochs, four GPUs, batch capacity16 and the original seed; it skips stages1–3. The historical full-episode helper supplies shared APIs; this follow-up does not invoke its full-training mode. Small saved optimizer-serialization and checkpoint-retention changes affect storage only.\n\nOperational scripts assume the documented RUN/code and RUN/scripts layout and retain explicit read-only input paths. Source fingerprints and per-job launch records are retained on the server. Unrelated runtime modules are not copied into the repository.\n\nS4 caches original FP32 detections and association embeddings once, then uses the unchanged native tracker for every projection. Projection applies to every native association embedding; bases are estimated from car RoIs. The enabled native head is independently compared against replay on the predeclared first Overcast video (ICS2 beta.5 and identity16 beta1). No normalization is added. Detector outputs are fixed; native track acceptance can still change tracking DetA.\n'''
    (DEST/'README.md').write_text(readme)
    doc=PUB/'experiments/SUMMARY_20261029.md';text=doc.read_text();text=text.replace('pending beta0 full-domain consistency gate','running; F uses user-approved fresh V100 beta0, old/new comparison published')
    notice='\nF S4 baseline deviation approved by the user: car HOTA new-minus-old is +0.046923 / +0.027106 / -0.014647 / -0.013192 pp for Overcast / Rainy / Dawn-Dusk / Night. Every car/OVERALL metric difference is in `ics_followup_20261029/S4/F/fresh_vs_historical_baseline.csv`. Old predictions reproduce old HOTA exactly under the current evaluator. Interventions use the new native beta0; HF keeps its frozen V100 reference.\n'
    text=text.replace('## Table G',notice+'\n## Table G')
    if '<!-- ICS_FOLLOWUP_BEGIN -->' not in text:text='<!-- ICS_FOLLOWUP_BEGIN -->\n'+text.rstrip()+'\n<!-- ICS_FOLLOWUP_END -->\n'
    doc.write_text(text)
    # Refresh the delivered-file hash list after adding runtime and audit files.
    import hashlib
    files=[f for root in [DEST,PUB/'experiments/ics_probe/analysis_F_stage4_e5_v2',PUB/'experiments/ics_probe/drift_decomposition'] for f in sorted(root.rglob('*')) if f.is_file() and f.name!='SHA256SUMS']
    (DEST/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(PUB))+'\n' for f in files))
    print('Staged runtime support and approved baseline audit; no nonzero or short-training result is claimed')
if __name__=='__main__':main()
