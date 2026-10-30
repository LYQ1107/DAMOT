#!/usr/bin/env python
"""Extract RoI association embeddings for the ICS subspace probe.

For every sampled frame the script runs the frozen tracker on several views of
the same image (clean, FAP, photometric, ...) and pools the association
embedding of every ground-truth box. Ground truth is used ONLY to place boxes
and to label identities/weather for the offline analysis; nothing is trained.

Output, one file per (checkpoint label, domain):
    <out-dir>/<label>/<domain>.npz      embeddings + metadata arrays
    <out-dir>/<label>/meta.json         checkpoint / ICS-bank / run info

Example (BDD100K, final DAMOT checkpoint):
    python tools/analysis/ics_probe_extract.py \
        --checkpoint $RUN/cache/weights/F_stage4_e5.pth --label F_stage4_e5 \
        --domain night $RUN/data/annotations/night.json $RUN/data/views/night/ \
        --out-dir $OUT/feats

Python 3.7 compatible (server env: torch 1.12, mmcv-full 1.6, mmdet 2.25).
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
import os
import os.path as osp
import re
import sys
import tempfile
import time

import numpy as np

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CLASSES = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
NORM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
NORM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# Perturbation strengths used for the dose-response curve (Fig. 4).
DOSE_VIEWS = (
    ['dark_%s' % s for s in ('0.80', '0.60', '0.40', '0.25')] +
    ['fog_%s' % s for s in ('0.15', '0.30', '0.45', '0.60')] +
    ['fapk_%s' % s for s in ('0.90', '0.70', '0.50', '0.30')])
BASE_VIEWS = ('clean', 'fap', 'photo', 'student')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config', default=None,
                   help='model config; default: config embedded in checkpoint meta')
    p.add_argument('--label', required=True, help='name of the output sub-folder')
    p.add_argument('--domain', nargs=3, action='append', required=True,
                   metavar=('NAME', 'ANN_JSON', 'IMG_PREFIX'),
                   help='repeatable; NAME is used as the weather label')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--box-source', choices=['gt', 'teacher'], default='gt')
    p.add_argument('--teacher-score-thr', type=float, default=None,
                   help='default: the checkpoint APR memory score threshold')
    p.add_argument('--views', default='clean,fap,photo,student',
                   help='comma list from %s' % (BASE_VIEWS,))
    p.add_argument('--dose-frames-per-video', type=int, default=3,
                   help='frames per video that also get the dose views; 0 = off')
    p.add_argument('--classes', default='car',
                   help="comma list of class names, or 'all'")
    p.add_argument('--max-videos', type=int, default=40)
    p.add_argument('--frames-per-video', type=int, default=30)
    p.add_argument('--frame-stride', type=int, default=2)
    p.add_argument('--max-rois-per-frame', type=int, default=40)
    p.add_argument('--min-box', type=float, default=16.0,
                   help='min box side (pixels, after resize)')
    p.add_argument('--img-scale', type=int, nargs=2, default=(1296, 720))
    p.add_argument('--fap-low-ratio', type=float, default=0.10)
    p.add_argument('--fap-keep-prob', type=float, default=0.90)
    p.add_argument('--video-regex', default=None,
                   help='regex with named groups "scene" and "variant" applied to '
                        'the video name (VKITTI2 pairing). Default: scene=video '
                        'name, variant=domain NAME')
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--device', default='cuda')
    p.add_argument('--view-batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4,
                   help='CPU view workers; 0 runs synchronously')
    p.add_argument('--dry-run', action='store_true',
                   help='only check annotations / image paths, no model')
    return p.parse_args()


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #
def load_coco_video(ann_file, read_annotations=True):
    with open(ann_file) as f:
        d = json.load(f)
    cats = {c['id']: c['name'] for c in d['categories']}
    videos = {v['id']: v.get('name', str(v['id'])) for v in d.get('videos', [])}
    imgs = {im['id']: im for im in d['images']}
    anns = {}
    for a in (d['annotations'] if read_annotations else []):
        anns.setdefault(a['image_id'], []).append(a)
    return cats, videos, imgs, anns


def stable_pick(items, k):
    """Deterministic, evenly spaced subset (keeps order)."""
    if k <= 0 or len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def sample_frames(imgs, videos, args):
    by_video = {}
    for im in imgs.values():
        by_video.setdefault(im['video_id'], []).append(im)
    vids = sorted(by_video, key=lambda v: str(videos.get(v, v)))
    vids = stable_pick(vids, args.max_videos)
    plan = []
    for v in vids:
        frames = sorted(by_video[v], key=lambda im: im['frame_id'])
        frames = frames[::max(1, args.frame_stride)][:args.frames_per_video]
        for j, im in enumerate(frames):
            plan.append((im, j < args.dose_frames_per_video))
    return plan


def parse_video(name, domain, regex):
    if regex is None:
        return name, domain
    m = re.search(regex, name)
    if m is None:
        return name, domain
    g = m.groupdict()
    return g.get('scene', name), g.get('variant', domain)


# --------------------------------------------------------------------------- #
# image views
# --------------------------------------------------------------------------- #
class ViewMaker(object):
    def __init__(self, args):
        from darth.datasets.pipelines.transforms import SPAFrequencyTransform
        from mmdet.datasets.pipelines import PhotoMetricDistortion
        self.fap = SPAFrequencyTransform(mode='low_freq_mask',
                                         low_ratio=args.fap_low_ratio,
                                         keep_prob=args.fap_keep_prob,
                                         apply_noise=False, noise_ratio=0.0)
        self.SPA = SPAFrequencyTransform
        self.photo = PhotoMetricDistortion()
        self.low_ratio = args.fap_low_ratio

    def _fap(self, img, keep_prob=None):
        t = self.fap if keep_prob is None else self.SPA(
            mode='low_freq_mask', low_ratio=self.low_ratio, keep_prob=keep_prob,
            apply_noise=False, noise_ratio=0.0)
        return t({'img': img.copy(), 'img_fields': ['img']})['img']

    def _photo(self, img):
        r = {'img': img.copy().astype(np.float32), 'img_fields': ['img']}
        return self.photo(r)['img']

    def make(self, name, img, seed):
        """img: resized BGR float32 in [0, 255]."""
        np.random.seed(seed % (2 ** 31 - 1))
        if name == 'clean':
            out = img
        elif name == 'fap':
            out = self._fap(img)
        elif name == 'photo':
            out = self._photo(img)
        elif name == 'student':           # training student view: photo -> FAP
            photo = self._photo(img)
            np.random.seed(seed % (2 ** 31 - 1))
            out = self._fap(photo)
        elif name.startswith('dark_'):
            out = img * float(name.split('_')[1])
        elif name.startswith('fog_'):
            a = float(name.split('_')[1])
            out = (1.0 - a) * img + a * 200.0
        elif name.startswith('fapk_'):
            out = self._fap(img, keep_prob=float(name.split('_')[1]))
        else:
            raise ValueError(name)
        return np.clip(out, 0, 255).astype(np.float32)


def view_key(name):
    return 'feat_' + name.replace('.', 'p')


def init_view_worker(args):
    # Spawned CPU workers do not inherit a live CUDA context. Each process has
    # its own NumPy RNG, so scheduling cannot change the seeded augmentations.
    import torch
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    global WORKER_VIEW_MAKER
    WORKER_VIEW_MAKER = ViewMaker(args)


def prepare_view_task(task):
    path, scale, vlist, seed = task
    img, ratio = preprocess(path, scale)
    return img.shape[:2], ratio, [WORKER_VIEW_MAKER.make(v, img, seed) for v in vlist]


def prepared_frames(plan, name, prefix, views, args, pool):
    def task(im, do_dose):
        seed = int(hashlib.md5(('%s/%s' % (name, im['id'])).encode())
                   .hexdigest()[:8], 16) + args.seed
        return (osp.join(prefix, im['file_name']), args.img_scale,
                list(views) + (DOSE_VIEWS if do_dose else []), seed)
    if pool is None:
        for n, (im, do_dose) in enumerate(plan):
            yield n, im, do_dose, prepare_view_task(task(im, do_dose))
        return
    pending = deque()
    it = iter(enumerate(plan))
    for _ in range(args.workers + 1):
        try:
            n, (im, do_dose) = next(it)
        except StopIteration:
            break
        pending.append((n, im, do_dose, pool.submit(prepare_view_task, task(im, do_dose))))
    while pending:
        n, im, do_dose, future = pending.popleft()
        prepared = future.result()
        try:
            nn, (next_im, next_dose) = next(it)
        except StopIteration:
            pass
        else:
            pending.append((nn, next_im, next_dose,
                            pool.submit(prepare_view_task, task(next_im, next_dose))))
        yield n, im, do_dose, prepared


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def strip_init(obj):
    if isinstance(obj, dict):
        obj.pop('init_cfg', None)
        obj.pop('pretrained', None)
        for v in obj.values():
            strip_init(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strip_init(v)


def load_model(args):
    import torch
    from mmcv import Config
    import darth.datasets  # noqa: F401  (registers pipelines/datasets)
    import darth.models  # noqa: F401
    from darth.models import build_model

    ck = torch.load(args.checkpoint, map_location='cpu')
    state = ck.get('state_dict', ck)
    meta = ck.get('meta', {}) if isinstance(ck, dict) else {}
    if args.config:
        cfg = Config.fromfile(args.config)
        cfg_src = args.config
    else:
        text = meta.get('config', '')
        if not text:
            raise SystemExit('checkpoint has no embedded config; pass --config')
        fd, path = tempfile.mkstemp(suffix='.py')
        with os.fdopen(fd, 'w') as f:
            f.write(text)
        cfg = Config.fromfile(path)
        os.unlink(path)
        cfg_src = 'checkpoint meta'
    strip_init(cfg.model)
    model = build_model(copy.deepcopy(cfg.model))
    target = model.state_dict()
    chosen = {k: v for k, v in state.items()
              if k in target and tuple(v.shape) == tuple(target[k].shape)}
    missing = [k for k in target if k not in chosen]
    need = [k for k in missing
            if k.startswith('detector.') or k.startswith('track_head.embed_head')
            or k.startswith('track_head.roi_extractor')
            or k.startswith('track_head.causal_bank.')]
    if args.box_source == 'teacher':
        if not hasattr(model, 'teacher_detector'):
            raise SystemExit('teacher box mode requires the full adaptation model')
        need.extend(k for k in missing if k.startswith('teacher_detector.'))
    if need:
        raise SystemExit('checkpoint lacks required keys, e.g. %s' % need[:5])
    model.load_state_dict(chosen, strict=False)
    model = model.to(args.device)
    # DarthQDTrack overrides train()/eval() without returning self.
    model.eval()
    if args.box_source == 'teacher':
        model.teacher_detector.eval()
        if args.teacher_score_thr is None:
            args.teacher_score_thr = float(model.track_head.roi_proto_memory.score_thr)

    # ICS bank of the STUDENT track head, read straight from the checkpoint.
    bank = {}
    for name in ('pcs', 'cov_bank', 'initialized', 'num_updates'):
        k = 'track_head.causal_bank.' + name
        if k in state:
            bank[name] = state[k].detach().float().cpu().numpy()
    th = cfg.model.get('track_head', {})
    bank_cfg = {k: th.get(k) for k in (
        'use_causal_direction_bank', 'causal_pca_mode', 'causal_num_pcs',
        'causal_start_pc', 'causal_trim_alpha', 'causal_normalize_feats',
        'causal_residual_normalize_before_delta', 'use_roi_proto_memory',
        'roi_proto_class_ids')}
    info = dict(checkpoint=osp.abspath(args.checkpoint), config=cfg_src,
                matched_keys=len(chosen), target_keys=len(target),
                missing_keys=len(missing), bank_cfg=bank_cfg,
                bank_initialized=bool(bank.get('initialized', np.zeros(1)).any()),
                bank_num_updates=int(bank.get('num_updates', np.zeros(1)).sum()),
                missing_key_names=missing,
                perturbation_protocol='fixed geometry; shared random seed across dose levels; '
                'student view contains photometric distortion and FAP only')
    if not info['bank_initialized'] or info['bank_num_updates'] <= 0:
        raise SystemExit('checkpoint has no initialized, updated ICS bank')
    return model, bank, info


def preprocess(path, scale):
    import mmcv
    img = mmcv.imread(path)
    if img is None:
        raise IOError('cannot read %s' % path)
    img = img.astype(np.float32)
    img, s = mmcv.imrescale(img, tuple(scale), return_scale=True)
    return img, float(s)


def to_tensor(views_bgr, device):
    import mmcv
    import torch
    arrs = []
    for im in views_bgr:
        x = mmcv.imnormalize(im, NORM_MEAN, NORM_STD, to_rgb=True)
        x = mmcv.impad_to_multiple(x, 32)
        arrs.append(x.transpose(2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(np.stack(arrs))).to(device)


def embed(model, imgs_t, boxes_t):
    import torch
    with torch.no_grad():
        x = model.detector.extract_feat(imgs_t)
        feats = model.track_head.extract_bbox_feats(
            x, [boxes_t for _ in range(imgs_t.shape[0])])
    return feats.view(imgs_t.shape[0], boxes_t.shape[0], -1).float().cpu().numpy()


def teacher_boxes(model, clean, img_shape, scale, im, path, args):
    """Return scaled clean-image teacher car boxes; never rescale them again."""
    import torch
    tensor = to_tensor([clean], args.device)
    meta = dict(img_shape=tuple(img_shape) + (3,),
                ori_shape=(int(im['height']), int(im['width']), 3),
                pad_shape=(int(tensor.shape[-2]), int(tensor.shape[-1]), 3),
                scale_factor=np.array([scale] * 4, dtype=np.float32),
                flip=False, filename=path, ori_filename=im['file_name'],
                img_norm_cfg=dict(mean=NORM_MEAN, std=NORM_STD, to_rgb=True))
    with torch.no_grad():
        result = model.teacher_detector.simple_test(tensor, [meta], rescale=False)[0]
    if isinstance(result, tuple):
        result = result[0]
    cars = np.asarray(result[CLASSES.index('car')], dtype=np.float32)
    cars = cars[cars[:, 4] >= float(args.teacher_score_thr)]
    return cars[:, :4].copy(), cars[:, 4].copy()


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    views = [v.strip() for v in args.views.split(',') if v.strip()]
    for v in views:
        assert v in BASE_VIEWS, v
    if 'clean' not in views:
        views = ['clean'] + views
    keep_cls = None if args.classes == 'all' else set(args.classes.split(','))
    out_dir = osp.join(args.out_dir, args.label)
    os.makedirs(out_dir, exist_ok=True)

    plans = {}
    for name, ann_file, prefix in args.domain:
        cats, videos, imgs, anns = load_coco_video(ann_file, read_annotations=args.box_source == 'gt')
        plan = sample_frames(imgs, videos, args)
        missing = [im['file_name'] for im, _ in plan
                   if not osp.exists(osp.join(prefix, im['file_name']))]
        print('[plan] %-18s videos=%d frames=%d dose_frames=%d missing(sampled)=%d'
              % (name, len(set(im['video_id'] for im, _ in plan)), len(plan),
                 sum(d for _, d in plan), len(missing)), flush=True)
        if missing:
            raise SystemExit('image not found, e.g. %s' %
                             osp.join(prefix, missing[0]))
        plans[name] = (cats, videos, anns, prefix, plan)
    if args.dry_run:
        return

    import torch
    import cv2
    torch.set_num_threads(2)
    cv2.setNumThreads(0)
    torch.manual_seed(args.seed)
    model, bank, info = load_model(args)
    print('[model] %s' % json.dumps(info), flush=True)
    pool = None
    if args.workers > 0:
        pool = ProcessPoolExecutor(max_workers=args.workers,
                                   mp_context=multiprocessing.get_context('spawn'),
                                   initializer=init_view_worker, initargs=(args,))
    else:
        init_view_worker(args)

    for name, (cats, videos, anns, prefix, plan) in plans.items():
        t0 = time.time()
        rec = {k: [] for k in ('video', 'scene', 'variant', 'instance_id',
                               'frame_id', 'image_id', 'category', 'bbox', 'img_path')}
        if args.box_source == 'teacher':
            rec['teacher_score'] = []
            rec['bbox_resized'] = []
        feats = {}
        for n_done, im, do_dose, prepared in prepared_frames(
                plan, name, prefix, views, args, pool):
            path = osp.join(prefix, im['file_name'])
            img_shape, s, prepared_views = prepared
            boxes, metas = [], []
            if args.box_source == 'teacher':
                b, teacher_scores = teacher_boxes(model, prepared_views[views.index('clean')],
                                                  img_shape, s, im, path, args)
                boxes = (b / s).tolist()  # metadata only, original-image coordinates
                metas = [(-1, CLASSES.index('car')) for _ in boxes]
            else:
                for a in anns.get(im['id'], []):
                    cname = cats.get(a['category_id'], str(a['category_id']))
                    if keep_cls is not None and cname not in keep_cls:
                        continue
                    if a.get('iscrowd', 0) or a.get('ignore', 0) or a.get('instance_id', -1) < 0:
                        continue
                    x, y, w, h = a['bbox']
                    boxes.append([x, y, x + w, y + h])
                    cl = cname.lower()
                    metas.append((a['instance_id'], CLASSES.index(cl)
                                  if cl in CLASSES else -1))
                b = np.array(boxes, dtype=np.float32) * s
            if not boxes:
                continue
            b[:, [0, 2]] = b[:, [0, 2]].clip(0, img_shape[1] - 1)
            b[:, [1, 3]] = b[:, [1, 3]].clip(0, img_shape[0] - 1)
            keep = np.where(((b[:, 2] - b[:, 0]) >= args.min_box) &
                            ((b[:, 3] - b[:, 1]) >= args.min_box))[0]
            if keep.size == 0:
                continue
            keep = keep[:args.max_rois_per_frame]
            b = b[keep]
            boxes_t = torch.from_numpy(b).to(args.device)
            vlist = list(views) + (DOSE_VIEWS if do_dose else [])
            out = {}
            for i in range(0, len(vlist), args.view_batch_size):
                chunk = vlist[i:i + args.view_batch_size]
                # Reuse random draws across FAP strengths (nested masks), and
                # use identical photometric parameters for photo/student.
                ims = prepared_views[i:i + args.view_batch_size]
                e = embed(model, to_tensor(ims, args.device), boxes_t)
                for v, ev in zip(chunk, e):
                    out[v] = ev
            n = b.shape[0]
            for v in views + DOSE_VIEWS:
                arr = out.get(v)
                if arr is None:
                    arr = np.full((n, out['clean'].shape[1]), np.nan, np.float32)
                feats.setdefault(view_key(v), []).append(arr.astype(np.float32))
            vname = str(videos.get(im['video_id'], im['video_id']))
            scene, variant = parse_video(vname, name, args.video_regex)
            for kk in keep:
                rec['video'].append(vname)
                rec['scene'].append(scene)
                rec['variant'].append(variant)
                rec['instance_id'].append(int(metas[kk][0]))
                rec['category'].append(int(metas[kk][1]))
                rec['frame_id'].append(int(im['frame_id']))
                rec['image_id'].append(int(im['id']))
                rec['img_path'].append(path)
                if args.box_source == 'teacher':
                    rec['teacher_score'].append(float(teacher_scores[kk]))
            rec['bbox'].append(np.array(boxes, dtype=np.float32)[keep])
            if args.box_source == 'teacher':
                rec['bbox_resized'].append(b.copy())
            if (n_done + 1) % 50 == 0:
                print('[%s] %d/%d frames  %.1fs' % (name, n_done + 1, len(plan),
                                                   time.time() - t0), flush=True)
        if not rec['video']:
            print('[warn] %s: no RoIs extracted' % name)
            continue
        arrays = {k: np.concatenate(v, 0) for k, v in feats.items()}
        arrays['bbox'] = np.concatenate(rec['bbox'], 0)
        if args.box_source == 'teacher':
            arrays['teacher_score'] = np.asarray(rec['teacher_score'], dtype=np.float32)
            arrays['bbox_resized'] = np.concatenate(rec['bbox_resized'], 0)
        for k in ('video', 'scene', 'variant', 'img_path'):
            arrays[k] = np.array(rec[k])
        for k in ('instance_id', 'frame_id', 'image_id', 'category'):
            arrays[k] = np.array(rec[k], dtype=np.int64)
        arrays['weather'] = np.array([name] * len(rec['video']))
        for k, v in bank.items():
            arrays['bank_' + k] = v
        np.savez_compressed(osp.join(out_dir, '%s.npz' % name), **arrays)
        domain_info = dict(info, label=args.label, domain=name,
                           views=views, dose_views=DOSE_VIEWS, args=vars(args),
                           rois=len(rec['video']), elapsed_seconds=time.time() - t0)
        with open(osp.join(out_dir, '%s_meta.json' % name), 'w') as f:
            json.dump(domain_info, f, indent=2, default=str)
        print('[done] %s rois=%d dose_rois=%d %.1fs' % (
            name, len(rec['video']),
            int(np.isfinite(arrays[view_key(DOSE_VIEWS[0])][:, 0]).sum()),
            time.time() - t0), flush=True)

    info.update(label=args.label, views=views, dose_views=DOSE_VIEWS,
                domains=[d[0] for d in args.domain], args=vars(args))
    with open(osp.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(info, f, indent=2, default=str)
    if pool is not None:
        pool.shutdown()


if __name__ == '__main__':
    main()
