"""Prepare only Night-stage 2x2 runs from exact saved HF history."""
import copy,json,sys,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code/tools/run'))
from apr_complement_experiment import Config,torch,sha,STAGES,strip_init
from protocol import diff,write_json,write_views
HF=Path('/data1/liuyeqiang/damot_apr_complement_hf_20261028_1420')

def main():
    source=HF/'episode/adapt/stage03_after_dawn_dusk/epoch_5.pth'
    statepath=HF/'episode/state/stage3_final_rank0.pth'
    state=torch.load(str(statepath),map_location='cpu');history=state['APR']['history']
    assert set(history)==set(STAGES[:3]);assert all(1 in h and h[1].shape==(100,256) for h in history.values())
    common=ROOT/'cache/HF_stage3_history.pth'
    if not common.exists():torch.save(history,str(common))
    loaded=torch.load(str(common),map_location='cpu')
    assert all(torch.equal(loaded[d][c],history[d][c]) for d in history for c in history[d])
    provenance=dict(source_checkpoint=str(source),checkpoint_sha256=sha(source),history_path=str(common),history_sha256=sha(common),source_history_sidecar=str(statepath),source_sidecar_sha256=sha(statepath),prototypes=300,method='Exact saved stage3 APR history (rank0) shared by all groups and all ranks',deviation='No <stage>_roi_proto.pth filename exists. Exact APR histories were saved in stage3_final_rank*.pth; no approximate reconstruction is needed.',rank_rng_sources={str(r):str(HF/'episode/state'/('stage3_final_rank%d.pth'%r)) for r in range(4)})
    write_json(ROOT/'manifests/S5_common_history.json',provenance)
    for group,mode,coupled in [('a','proto_shrinkage_cov_ema',True),('b','proto_shrinkage_cov_ema',False),('c','pairwise_residual',True),('d','pairwise_residual',False)]:
        out=ROOT/'results/S5'/('HF_'+group);out.mkdir(parents=True,exist_ok=True)
        baseline=Config.fromfile(str(HF/'configs/stage4.py'));cfg=copy.deepcopy(baseline)
        cfg.model.track_head.causal_pca_mode=mode;cfg.model.track_head.roi_proto_couple_ics=coupled
        if mode=='pairwise_residual':cfg.model.track_head.causal_residual_normalize_before_delta=True
        changes=diff(baseline.model,cfg.model,'model')
        assert all(c['field'] in ['model.track_head.causal_pca_mode','model.track_head.roi_proto_couple_ics','model.track_head.causal_residual_normalize_before_delta'] for c in changes)
        assert cfg.model.track_head.causal_min_samples==96
        cfg.total_epochs=2
        if cfg.get('runner'):cfg.runner.max_epochs=2
        cfg.work_dir=str(out/'episode/adapt/stage04_after_night')
        cfg.custom_hooks=[h for h in cfg.custom_hooks if h['type']!='APRExperimentStateHook']+[dict(type='ICSShortStageStateHook',root=str(out),priority='LOW')]
        cfg.checkpoint_config.max_keep_ckpts=1
        cfg.checkpoint_config.save_optimizer=False
        cfg.load_from=None;cfg.resume_from=None
        configpath=out/'config.py';cfg.dump(str(configpath))
        snaps=out/'episode/stage_snapshots';snaps.mkdir(parents=True,exist_ok=True)
        link=snaps/'stage03_after_dawn_dusk.pth'
        if not link.exists():link.symlink_to(source)
        (out/'manifests').mkdir(exist_ok=True)
        write_json(out/'manifests/common.json',dict(provenance,group=group,seed=int(cfg.seed),epochs=2))
        (out/'manifests/data.json').write_bytes((HF/'manifests/data.json').read_bytes())
        write_json(out/'config_diff.json',dict(model_changes=changes,common_protocol_changes=diff(baseline._cfg_dict,cfg._cfg_dict),entrypoint='train_seq3_episode.py --resume-episode-from-stage 4',allowed_training_stages=[4],history=provenance,storage_only_changes=['Keep latest epoch only; omit optimizer serialization (optimizer computation is unchanged)','Do not serialize redundant stage00; final stage04 is linked to epoch2'],source_inputs_read_only=True))
        write_views(cfg,out/'resolved_views.txt',dict(epochs=2,geometry='same effective pipelines as HF stage4',bank='Inherited unchanged from HF stage3 checkpoint at start'))
    print('S5 prepared; no training was launched',flush=True)
if __name__=='__main__':main()
