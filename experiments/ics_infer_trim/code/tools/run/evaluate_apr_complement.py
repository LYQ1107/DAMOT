#!/usr/bin/env python3
"""Native student evaluation and F comparison; no checkpoint selection."""
import argparse
import csv
from functools import partial
import json
import os
from pathlib import Path
import pickle
import sys
import time

from apr_complement_experiment import (
    CODE, OLD, STAGES, CLASSES, EXPECTED, Config, torch, np,
    dump, sha, strip_init, build_dataset, build_model)


def save_pickle(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def csv_write(path, rows):
    assert rows, path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_final_report_table(root, rows):
    """Write the wide table used by the rebuttal report.

    The table is deliberately built from the final (stage 4, epoch 5)
    checkpoint only.  ``Avg`` is the arithmetic mean of the four reported
    domain values; it is a presentation summary, not the official HOTA
    aggregation and is therefore labelled explicitly in the sidecar note.
    """
    metrics = ['HOTA_percent', 'DetA_percent', 'IDF1_percent', 'AssA_percent']
    final = [r for r in rows if int(r['stage']) == 4]
    report = []
    for category in ['OVERALL', 'car']:
        by_domain = {
            r['eval_domain']: r for r in final if r['category'] == category
        }
        assert set(by_domain) == set(STAGES)
        row = {'category': category, 'stage': 4, 'epoch': 5}
        for domain in STAGES:
            for metric in metrics:
                row['%s_%s' % (domain, metric)] = float(by_domain[domain][metric])
        for metric in metrics:
            row['Avg_%s' % metric] = sum(
                float(by_domain[d][metric]) for d in STAGES) / len(STAGES)
        report.append(row)
    csv_write(root / 'results/report_table.csv', report)

    labels = {
        'overcast': 'Overcast', 'rainy': 'Rainy',
        'dawn_dusk': 'Dawn-Dusk', 'night': 'Night', 'Avg': 'Avg'
    }
    lines = [
        '# Final stage report table',
        '',
        'Values are percentages from the final stage-4 epoch-5 student checkpoint.',
        'Avg is the arithmetic mean of the four domain columns; it is not an official HOTA aggregate.',
        '',
        '| category | ' + ' | '.join(
            '%s %s' % (labels[d], m.replace('_percent', ''))
            for d in list(STAGES) + ['Avg'] for m in metrics) + ' |',
        '|---|' + '---|' * (len(metrics) * (len(STAGES) + 1)),
    ]
    for row in report:
        values = []
        for domain in list(STAGES) + ['Avg']:
            for metric in metrics:
                values.append('%.8f' % row['%s_%s' % (domain, metric)])
        lines.append('| %s | %s |' % (row['category'], ' | '.join(values)))
    (root / 'results/report_table.md').write_text('\n'.join(lines) + '\n')
    (root / 'results/report_table_notes.txt').write_text(
        '来源：stage=4, epoch=5；OVERALL 和 car 均保留。\n'
        '所有数值单位为百分比。Avg 为四个评估域数值的算术平均，\n'
        '不替代官方 HOTA 聚合，也不用于选择 checkpoint 或阈值。\n')


def all_frame_hota_index(self):
    by_video = {v['id']: [] for v in self.gt_data['videos']}
    for im in self.gt_data['images']:
        by_video[im['video_id']].append(im)
    return {v: {im['id']: i for i, im in enumerate(
        sorted(images, key=lambda x: x['frame_id']))}
        for v, images in by_video.items()}


def evaluate(root, stage, domain):
    from mmcv.cnn import fuse_conv_bn
    from mmcv.parallel import collate, scatter
    from torch.utils.data import DataLoader, Subset
    from darth.datasets.eval.coco_eval_dataset import BDD100K
    # Same correction used for the historical F rebuttal reference.
    BDD100K._compute_image_to_timestep_mappings = all_frame_hota_index
    torch.multiprocessing.set_start_method('fork', force=True)
    torch.backends.cudnn.benchmark = False
    out = root / 'results' / ('stage%d' % stage) / domain
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        record = json.loads((out / 'complete.json').read_text())
        assert (record['videos'], record['frames']) == EXPECTED[domain]
        return
    checkpoint = root / 'episode/stage_snapshots' / (
        'stage%02d_after_%s.pth' % (stage, STAGES[stage-1]))
    assert checkpoint.exists()
    checksum = sha(checkpoint)
    data = json.loads((root / 'manifests/data.json').read_text())[domain]
    cfg = Config.fromfile(str(CODE / 'configs/mot/qdtrack/bdd_7domain' /
                             ('qdtrack_faster-rcnn_r50_fpn_20e_bdd_%s.py' % domain)))
    strip_init(cfg.model)
    cfg.model.tracker.init_score_thr = .30
    cfg.model.tracker.obj_score_thr = .20
    cfg.model.detector.test_cfg.rcnn.score_thr = .05
    if 'test_cfg' in cfg.model.detector.roi_head:
        cfg.model.detector.roi_head.test_cfg.score_thr = .05
    cfg.data.test.ann_file = data['annotation']
    cfg.data.test.img_prefix = data['image_view'] + '/'
    cfg.data.test.classes = CLASSES
    cfg.data.test.test_mode = True
    cfg.dump(str(out / 'used_config.py'))
    dump(out / 'job.json', dict(pid=os.getpid(), stage=stage, domain=domain,
                               checkpoint=str(checkpoint), sha256=checksum,
                               gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                               command=sys.argv, start=time.time()))
    ds = build_dataset(cfg.data.test)
    assert sorted(ds.cat2label.values()) == list(range(6))
    assert [ds.coco.cats[cid]['name'] for cid, label in
            sorted(ds.cat2label.items(), key=lambda x:x[1])] == list(CLASSES)
    videos = {}
    for i, im in enumerate(ds.data_infos):
        videos.setdefault(im['video_id'], []).append(i)
    assert (len(videos), len(ds)) == EXPECTED[domain]
    pending = []
    for v, indices in videos.items():
        p = out / 'predictions' / ('video_%s.pkl' % v)
        if p.exists():
            with open(p, 'rb') as f:
                r = pickle.load(f)
            assert r['indices'] == indices and r['checkpoint_sha256'] == checksum
            assert len(r['outputs']) == len(indices)
        else:
            pending.extend(indices)
    if pending:
        ck = torch.load(str(checkpoint), map_location='cpu')
        model = build_model(cfg.model)
        target = model.state_dict()
        state = {k: v for k, v in ck['state_dict'].items() if k in target}
        assert set(state) == set(target)
        model.load_state_dict(state, strict=True)
        for k in ['detector.backbone.conv1.weight',
                  'track_head.embed_head.fc_embed.weight']:
            assert torch.equal(model.state_dict()[k], state[k])
        dump(out / 'load.json', dict(
            student_target_keys=len(target), loaded_student_keys=len(state),
            missing_student_keys=[], ignored_training_teacher_and_bank_keys=[
                k for k in ck['state_dict'] if k not in state], strict=True))
        del ck, state
        model.CLASSES = CLASSES
        model = fuse_conv_bn(model).cuda().eval()
        loader = DataLoader(Subset(ds, pending), batch_size=1, shuffle=False,
                            num_workers=2, collate_fn=partial(collate, samples_per_gpu=1))
        active, outputs = None, []
        with torch.no_grad():
            for index, batch in zip(pending, loader):
                im = ds.data_infos[index]
                v = im['video_id']
                if active != v:
                    assert im['frame_id'] == 0
                    model.tracker.reset()
                    active, outputs = v, []
                outputs.append(model(return_loss=False, rescale=True,
                                     **scatter(batch, [0])[0]))
                if index == videos[v][-1]:
                    save_pickle(out / 'predictions' / ('video_%s.pkl' % v), dict(
                        video_id=v, indices=videos[v], outputs=outputs,
                        checkpoint_sha256=checksum))
                    print('VIDEO COMPLETE', stage, domain, v, len(outputs), flush=True)
        del model
        torch.cuda.empty_cache()
    results = dict(det_bboxes=[], track_bboxes=[])
    seen = []
    for v, indices in videos.items():
        with open(out / 'predictions' / ('video_%s.pkl' % v), 'rb') as f:
            r = pickle.load(f)
        assert r['indices'] == indices and r['checkpoint_sha256'] == checksum
        seen.extend(indices)
        for output in r['outputs']:
            for k in results:
                results[k].append(output[k])
    assert seen == list(range(len(ds)))
    save_pickle(out / 'raw_results.pkl', results)
    metrics = ds.evaluate(
        results, metric=['track'], results_path=str(out / 'evaluation'),
        results_kwargs=dict(scalabel=False, append_empty_preds=True),
        track_kwargs=dict(iou_thr=.5, ignore_iof_thr=.5,
                          ignore_by_classes=False, nproc=1, majority_voting=False))
    dump(out / 'evaluation_flat_metrics.json', metrics)
    for kind in ['summary', 'hota', 'track']:
        source = out / 'evaluation' / (kind + '_metrics.csv')
        assert source.exists()
        import shutil
        shutil.copyfile(source, out / source.name)
    gt = np.zeros(6, dtype=np.int64)
    for info in ds.data_infos:
        ann = ds.get_ann_info(info)
        gt += np.bincount(ann['labels'], minlength=6)
    summary = {r['']:r for r in csv.DictReader((out / 'summary_metrics.csv').open())}
    clear = {r['']:r for r in csv.DictReader((out / 'track_metrics.csv').open())}
    rows = []
    for category in list(CLASSES) + ['OVERALL']:
        s, c = summary[category], clear[category]
        n_gt = int(gt.sum()) if category == 'OVERALL' else int(gt[CLASSES.index(category)])
        rows.append(dict(
            stage=stage, epoch=5, eval_domain=domain, category=category,
            HOTA_percent=float(s['HOTA']), DetA_percent=float(s['DetA']),
            IDF1_percent=float(s['IDF1']), AssA_percent=float(s['AssA']),
            IDSw_count=int(c['IDSw']), GT_count=n_gt,
            IDSw_per_1000GT=1000*int(c['IDSw'])/n_gt if n_gt else '',
            frames=len(ds), videos=len(videos), detector_score_thr=.05,
            tracker_init_score_thr=.30, tracker_obj_score_thr=.20,
            conv_bn_fusion=True, checkpoint=str(checkpoint),
            checkpoint_sha256=checksum, annotation_sha256=data['annotation_sha256']))
    csv_write(out / 'metrics.csv', rows)
    dump(out / 'complete.json', dict(
        status='complete', stage=stage, domain=domain, epoch=5,
        frames=len(ds), videos=len(videos), exit_code=0,
        all_frame_HOTA=True, time=time.time()))
    print('CELL COMPLETE', stage, domain, flush=True)


def aggregate(root):
    rows = []
    for stage in range(1, 5):
        for domain in STAGES[:stage]:
            out = root / 'results' / ('stage%d' % stage) / domain
            done = json.loads((out / 'complete.json').read_text())
            assert done['status'] == 'complete'
            assert (done['videos'], done['frames']) == EXPECTED[domain]
            rows.extend(csv.DictReader((out / 'metrics.csv').open()))
    assert len(rows) == 70
    csv_write(root / 'results/stage_domain_metrics.csv', rows)
    write_final_report_table(root, rows)
    reference = [r for r in csv.DictReader(
        (OLD / 'results/exp1/stage_domain_metrics.csv').open()) if r['model']=='F']
    index = {(int(r['stage']), r['eval_domain'], r['category']):r for r in reference}
    paired = []
    for r in rows:
        key = (int(r['stage']), r['eval_domain'], r['category'])
        if key not in index:
            continue
        old = index[key]
        assert int(r['GT_count']) == int(old['GT_count'])
        val = {k:r[k] for k in ['stage','eval_domain','category']}
        for name in ['HOTA_percent','DetA_percent','IDF1_percent','AssA_percent',
                     'IDSw_count','IDSw_per_1000GT']:
            val['F_'+name] = float(old[name])
            val['new_'+name] = float(r[name])
            val['new_minus_F_'+name] = float(r[name])-float(old[name])
        paired.append(val)
    csv_write(root / 'results/comparison_to_F.csv', paired)
    forgetting = []
    for cls in ['OVERALL','car']:
        for name in ['IDF1_percent','AssA_percent','HOTA_percent']:
            cells = {(int(r['stage']),r['eval_domain']):float(r[name])
                     for r in rows if r['category']==cls}
            f = [max(cells[i,STAGES[j-1]] for i in range(j,4))
                 - cells[4,STAGES[j-1]] for j in range(1,4)]
            forgetting.append(dict(
                category=cls, metric=name,
                FinalOld_percent=sum(cells[4,d] for d in STAGES[:3])/3,
                AvgF_percentage_points=sum(f)/3,
                F_overcast_pp=f[0], F_rainy_pp=f[1], F_dawn_dusk_pp=f[2],
                formula='mean_j=1..3[max_i=j..3 R_ij - R_4j]; no clipping'))
    csv_write(root / 'results/forgetting_metrics.csv', forgetting)
    lines = [
        '四卡 F + 高频噪声 + ICS 补空间 APR 实验结果',
        'RUN_ROOT: '+str(root),
        'Clear 源模型 epoch20 起步；Overcast→Rainy→Dawn-Dusk→Night，每域5 epoch。',
        'F 的几何、student低频、ICS、car-only原型筛选、损失权重保持；contrastive加入0.03噪声，APR改为β=1补空间投影后余弦。',
        '4张V100、FP32、每卡16；使用非重入激活重算；软件版本和硬件与原四卡A100训练不同。',
        '测试阈值0.05/0.30/0.20，student native QDTrack，Conv-BN fusion，按视频重置，HOTA包含全部帧。',
        '70条逐类记录覆盖10个已学域单元，未评估未来域；完整精度见CSV。',
        '下列差值是高频与APR同时变更的整体实验结果，不能单独归因APR。']
    for stage, domain in enumerate(STAGES,1):
        r = next(r for r in paired if int(r['stage'])==stage and
                 r['eval_domain']==domain and r['category']=='OVERALL')
        lines.append('%s: HOTA=%.8f, IDF1=%.8f, AssA=%.8f; 对F差值=%.8f/%.8f/%.8f个百分点。' % (
            domain,r['new_HOTA_percent'],r['new_IDF1_percent'],r['new_AssA_percent'],
            r['new_minus_F_HOTA_percent'],r['new_minus_F_IDF1_percent'],r['new_minus_F_AssA_percent']))
    for r in forgetting:
        lines.append('%s %s: FinalOld=%.8f%%, AvgF=%.8f个百分点。' % (
            r['category'],r['metric'],r['FinalOld_percent'],r['AvgF_percentage_points']))
    (root / 'summary.txt').write_text('\n'.join(lines)+'\n')
    dump(root / 'progress.json', dict(status='complete', evaluated_cells=10,
                                     verified_class_rows=70, time=time.time()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--stage', type=int)
    p.add_argument('--domain', choices=STAGES)
    p.add_argument('--aggregate', action='store_true')
    args = p.parse_args()
    if args.aggregate:
        aggregate(args.run_root)
    else:
        assert args.stage in range(1,5) and STAGES.index(args.domain)<args.stage
        evaluate(args.run_root, args.stage, args.domain)


if __name__ == '__main__':
    main()
