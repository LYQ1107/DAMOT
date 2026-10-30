"""Every original annotated frame, native resolution, OpenCV BT.601 grayscale."""
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import cv2
from common import *
cv2.setNumThreads(0)

def video_brightness(task):
    domain,video,images,is_test=task
    values=[]
    for im in sorted(images,key=lambda x:x['frame_id']):
        p=image_prefix(domain)/im['file_name'];img=cv2.imread(str(p))
        if img is None:raise FileNotFoundError(str(p))
        values.append(float(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).mean()))
    out=dict(domain=domain,video=video['name'],video_id=video['id'],test=bool(is_test),frames=len(values),brightness=float(np.mean(values)))
    print('BRIGHTNESS',domain,video['name'],len(values),out['brightness'],flush=True)
    return out

def main():
    tasks=[]
    for domain in ['overcast','rainy','dawn_dusk','night']:
        a=json.loads((RUN/'data/annotations'/(domain+'.json')).read_text());videos=sorted(a['videos'],key=lambda v:v['id'])
        fit=probe.split_mask(np.array([v['name'] for v in videos]),0);by=defaultdict(list)
        for im in a['images']:by[im['video_id']].append(im)
        tasks.extend((domain,v,by[v['id']],not f) for v,f in zip(videos,fit))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=[]
        for row in pool.map(video_brightness,tasks):
            rows.append(row);csvout(ROOT/'tracking/video_brightness.csv',rows)
    dump(ROOT/'manifests/brightness_complete.json',dict(status='complete',videos=len(rows),frames=sum(r['frames'] for r in rows),
         convention='mean over all full-resolution original-frame OpenCV BGR2GRAY pixel means, then equal frame average within video; no resizing or frame sampling'))

if __name__=='__main__':main()
