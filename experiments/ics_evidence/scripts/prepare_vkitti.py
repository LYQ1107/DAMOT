"""Audit six VKITTI domains; freeze frame samples and IoU matches before inference."""
import json
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from evidence_cpu import ROOT, csvout, dump, sha

DOMAINS = ['clone', 'fog', 'morning', 'overcast', 'rain', 'sunset']
REPO = Path('/data1/liuyeqiang/nas/A100/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main')
RAW = Path('/data1/liuyeqiang/nas/A100/testdata/vranlee/.MOTSynth.partial/VKITTI2/vkitti_2.0.3_rgb')
DEST = ROOT / 'data/vkitti2'


def video_parts(name):
    match = re.fullmatch(r'(Scene\d+)_(\w+)_Camera_(\d+)', name)
    if not match:
        raise ValueError(name)
    scene, weather, camera = match.groups()
    return scene, weather, camera


def match_boxes(a, b):
    if not a or not b:
        return []
    aa, bb = np.array([v['bbox'] for v in a]), np.array([v['bbox'] for v in b])
    axy, bxy = aa[:, :2]+aa[:, 2:], bb[:, :2]+bb[:, 2:]
    wh = np.maximum(0, np.minimum(axy[:, None], bxy[None]) - np.maximum(aa[:, None, :2], bb[None, :, :2]))
    inter = wh.prod(2)
    iou = inter / (aa[:, 2:].prod(1)[:, None] + bb[:, 2:].prod(1)[None] - inter + 1e-12)
    # Maximize number of eligible matches first, then IoU, deterministically.
    row, col = linear_sum_assignment(-((iou >= .9).astype(float)*1000 + iou))
    return [(a[i], b[j], float(iou[i,j])) for i,j in zip(row,col) if iou[i,j] >= .9]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    files, snapshots, checks, copy_tasks = {}, {}, [], []
    for domain in DOMAINS:
        source = REPO / ('data/vkitti2/annotations/vkitti2_%s_cocofmt.json' % domain)
        data = json.loads(source.read_text())
        videos = {v['id']:v['name'] for v in data['videos']}
        car = [v['id'] for v in data['categories'] if v['name'].lower()=='car']
        assert len(car)==1
        by_video = defaultdict(list)
        for im in data['images']:
            by_video[im['video_id']].append(im)
        selected = []
        frame_signature = {}
        for vid, items in sorted(by_video.items()):
            scene, _, camera = video_parts(videos[vid])
            chosen = sorted(items, key=lambda im:im['frame_id'])[::3][:60]
            frame_signature[scene+'|Camera_'+camera] = [im['frame_id'] for im in chosen]
            for im in chosen:
                raw = RAW / scene / domain / 'frames/rgb' / ('Camera_'+camera) / Path(im['file_name']).name
                dest = DEST / 'images' / im['file_name']
                copy_tasks.append((raw, dest))
            selected.extend(chosen)
        if snapshots:
            assert frame_signature == snapshots['clone']['signature'], ('frame mismatch',domain)
        ims = {im['id']:im for im in selected}
        annotations = [a for a in data['annotations'] if a['image_id'] in ims and a['category_id']==car[0]
                       and not a.get('ignore',0) and not a.get('iscrowd',0) and min(a['bbox'][2:])>0]
        key_anns = defaultdict(list)
        for a in annotations:
            im = ims[a['image_id']]
            scene, _, camera = video_parts(videos[im['video_id']])
            key_anns[(scene,camera,im['frame_id'])].append(a)
        subset = dict(data, images=selected, annotations=annotations)
        dest_ann = DEST / 'annotations' / (domain+'.json')
        dump(dest_ann, subset)
        snapshots[domain] = dict(annotations=key_anns, signature=frame_signature, videos=videos)
        files[domain] = str(dest_ann)
        checks.append(dict(domain=domain, config=str(Path('/data1/liuyeqiang/damot_rebuttal_20261026/code/configs/_base_/datasets') / ('vkitti2_clone.py' if domain=='clone' else 'vkitti2_%s_eval.py'%domain)),
                           annotation_original=str(source), annotation_sha256=sha(source), annotation_sampled=str(dest_ann),
                           img_prefix_config='/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/vkitti2/images/',
                           resolved_raw_prefix=str(RAW), resolved_sampled_prefix=str(DEST/'images'),
                           videos=len(data['videos']), sampled_frames=len(selected), sampled_car_boxes=len(annotations)))
        print('ANNOTATIONS', domain, len(selected), len(annotations), flush=True)
    matches, counts = [], []
    for domain in DOMAINS[1:]:
        n = 0
        for key, a in sorted(snapshots['clone']['annotations'].items()):
            b = snapshots[domain]['annotations'].get(key,[])
            for aa, bb, iou in match_boxes(a,b):
                matches.append(dict(weather=domain, scene=key[0], camera=int(key[1]), frame_id=key[2],
                     clone_image_id=aa['image_id'], clone_annotation_id=aa['id'], clone_instance_id=aa['instance_id'],
                     variant_image_id=bb['image_id'], variant_annotation_id=bb['id'], variant_instance_id=bb['instance_id'],
                     IoU=iou, clone_bbox=json.dumps(aa['bbox']), variant_bbox=json.dumps(bb['bbox'])))
                n += 1
        counts.append(dict(weather=domain, sampled_pairs=n, below_500=n<500))
        print('MATCHES',domain,n,flush=True)
    csvout(ROOT/'part2/pairing_counts.csv',counts)
    csvout(ROOT/'part2/GT_pairs.csv',matches)
    dump(ROOT/'manifests/vkitti2_inputs.json',dict(domains=checks, pairing_rule='same scene, same camera, same frame; maximum-cardinality IoU>=0.9 matching, then maximum IoU',
         sampling='all five scenes, both cameras, every third frame starting at 0, first60 per video', min_box=0, max_rois='all GT cars',
         paired_identity_definition='clone identity obtained exclusively by IoU matches; no assumption of cross-weather ID consistency'))
    (ROOT/'vkitti2_inputs.txt').write_text('\n\n'.join(json.dumps(v,indent=2) for v in checks)+'\n')
    failures=[]
    def copy(pair):
        raw,dest=pair
        try:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                shutil.copy2(str(raw),str(dest))
            return None
        except Exception as e:
            return dict(source=str(raw),destination=str(dest),error=repr(e))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, failure in enumerate(pool.map(copy,copy_tasks)):
            if failure:failures.append(failure)
            if (i+1)%100==0:print('STAGED',i+1,'/',len(copy_tasks),'failures',len(failures),flush=True)
    dump(ROOT/'manifests/vkitti2_stage_complete.json',dict(status='complete' if not failures else 'failed',images=len(copy_tasks),failures=failures))
    assert not failures, failures[:3]


if __name__=='__main__':
    main()
