#!/usr/bin/env python3
"""Read-only NAS inventory, metadata-only checkpoint inspection, local cache."""
import json, os, sys, hashlib, pickle, zipfile, shutil, time, concurrent.futures
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
R = Path('/data1/liuyeqiang/nas/A100/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main')
M = Path('/data1/liuyeqiang/nas/A100/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/bdd/bdd100k/images/track')
F = R/'work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e'
CLASSES = ['pedestrian','car','truck','bus','motorcycle','bicycle']
STAGES = ['overcast','rainy','dawn_dusk','night']

def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)); t.replace(p)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

class MetaUnpickler(pickle.Unpickler):
    def persistent_load(self, pid): return {'storage':str(pid)}
    def find_class(self, module, name):
        if module=='torch._utils' and name.startswith('_rebuild_tensor'):
            return lambda storage,offset,size,stride,*args:dict(shape=size,stride=stride,storage=storage)
        if module=='torch._utils' and name.startswith('_rebuild_parameter'):
            return lambda data,*args:data
        if module=='torch' and name.endswith('Storage'):return name
        return super().find_class(module,name)

def metadata(p):
    with zipfile.ZipFile(p) as z:
        name=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        with z.open(name) as f: ck=MetaUnpickler(f).load()
    return ck

def checkpoint_inventory():
    paths={'S':R/'work_dirs/bdd_4domain/source/clear_20e/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear/epoch_20.pth'}
    for i,d in enumerate(STAGES,1): paths['F_stage%d_e5'%i]=F/('episode/adapt/stage%02d_after_%s/epoch_5.pth'%(i,d))
    for e in [1,3]:paths['F_stage4_e%d'%e]=F/('episode/adapt/stage04_after_night/epoch_%d.pth'%e)
    for label,method in [('N_old','spaP1_causalOld_4domain_5e'),('F_old','spaP1_causalOld_roiV1Weak_4domain_5e')]:
        paths[label]=R/('work_dirs/bdd_4domain/module_combinations/%s/episode/adapt/stage04_after_night/epoch_5.pth'%method)
    records=[]
    for label,p in paths.items():
        print('metadata',label,str(p),flush=True)
        ck=metadata(p);meta=ck.get('meta',{})
        config=meta.get('config','')
        (RUN/'configs'/(label+'_embedded.py')).write_text(config)
        state=ck.get('state_dict',{})
        dump(RUN/'manifests'/(label+'_state_shapes.json'),state)
        records.append(dict(model=label,source=str(p),bytes=p.stat().st_size,meta={k:v for k,v in meta.items() if k!='config'},state_key_count=len(state),teacher_key_count=sum(k.startswith('teacher_detector.') for k in state),config_sha256=hashlib.sha256(config.encode()).hexdigest()))
    dump(RUN/'manifests/checkpoints.json',records)
    return records

def cache_weights(records):
    for rec in records:
        if rec['model'] in ['N_old','F_old']:continue
        target=RUN/'cache/weights'/(rec['model']+'.pth')
        if not target.exists():
            print('cache_weight',rec['model'],rec['bytes'],flush=True)
            h=hashlib.sha256()
            part=target.with_suffix('.part')
            with open(rec['source'],'rb') as src,open(part,'wb') as dst:
                for b in iter(lambda:src.read(8*1024*1024),b''):h.update(b);dst.write(b)
            assert part.stat().st_size==rec['bytes']
            part.replace(target);rec['sha256']=h.hexdigest()
        else:rec['sha256']=sha(target)
        rec['local']=str(target)
        dump(RUN/'manifests/checkpoints.json',records)

def datasets():
    A=R/'data/bdd_da/annotations/box_track_20'
    info={}; namesets={}
    for domain in ['clear_source']+STAGES+['clear_holdout_45v']:
        name='clear' if domain=='clear_source' else domain
        src=A/('box_track_%s_all_cocofmt.json'%name)
        if domain=='clear_holdout_45v':src=A/'backup_all_splits_20260627_160444/box_track_clear_train_cocofmt.json'
        dst=RUN/'data/annotations'/(domain+'.json')
        shutil.copyfile(src,dst); digest=sha(dst)
        if domain=='clear_holdout_45v':assert digest=='5afd6fafb2f9c6f3c8a48852b6841eefafabfc4e8877a5c4ffa52d1b6a6e31bd'
        ann=json.loads(dst.read_text()); videos={v['id']:v['name'] for v in ann['videos']}
        namesets[domain]=set(videos.values())
        print(domain,len(videos),len(ann['images']),len(ann['annotations']),ann['categories'],ann['images'][0],flush=True)
        frame_info=[];splits={};view=RUN/'data/views'/domain;view.mkdir(parents=True,exist_ok=True)
        for vid,name in videos.items():
            found=[(s,M/s/name) for s in ['train','val'] if (M/s/name).is_dir()]
            assert len(found)==1,(name,found)
            split,original=found[0];splits[name]=split
            # All annotations keep their original file_name, normally video/frame.jpg.
            link=view/name
            if not link.exists():link.symlink_to(original,target_is_directory=True)
        # One directory listing per video avoids tens of thousands of remote stats.
        listings={name:{e.name:e for e in os.scandir(M/splits[name]/name)} for name in videos.values()}
        for im in sorted(ann['images'],key=lambda i:(i['video_id'],i['frame_id'])):
            name=videos[im['video_id']];original=M/splits[name]/im['file_name']
            assert Path(im['file_name']).name in listings[name],original
            frame_info.append(dict(image_id=im['id'],video_id=im['video_id'],video=name,frame_id=im['frame_id'],file_name=im['file_name'],split=splits[name],source=str(original)))
        info[domain]=dict(source_annotation=str(src),annotation=str(dst),sha256=digest,videos=len(videos),frames=len(frame_info),annotations=len(ann['annotations']),categories=ann['categories'],image_view=str(view),split_by_video=splits)
        dump(RUN/'manifests'/(domain+'_frames.json'),frame_info)
    assert not namesets['clear_holdout_45v'].intersection(set.union(*(s for k,s in namesets.items() if k!='clear_holdout_45v')))
    dump(RUN/'manifests/datasets.json',info)
    dump(RUN/'results/exp2/clear_holdout_manifest.json',info['clear_holdout_45v'])
    dump(RUN/'results/exp3/frame_manifest.json',dict(dataset=info['night'],frames=json.loads((RUN/'manifests/night_frames.json').read_text()),geometry='fixed, deterministic: base training load pipeline before teacher_pipeline; original xyxy coordinates retained for matching'))

def cache_images(domain):
    frames=json.loads((RUN/'manifests'/(domain+'_frames.json')).read_text())
    import cv2
    def one(rec):
        dst=RUN/'cache/images'/domain/rec['file_name'];dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copyfile(rec['source'],dst)
        im=cv2.imread(str(dst))
        if im is None:raise RuntimeError('decode failed: '+str(dst))
        return im.shape[:2]
    with concurrent.futures.ThreadPoolExecutor(12) as ex:
        shapes=list(ex.map(one,frames))
    assert len(shapes)==len(frames)
    dump(RUN/'manifests'/(domain+'_decode_validation.json'),dict(frames=len(frames),shape_counts={str(k):shapes.count(k) for k in set(shapes)},all_decoded=True))

if __name__=='__main__':
    if sys.argv[1]=='datasets':datasets()
    elif sys.argv[1]=='inventory':
        records=checkpoint_inventory();datasets()
    elif sys.argv[1]=='weights':cache_weights(json.loads((RUN/'manifests/checkpoints.json').read_text()))
    elif sys.argv[1]=='images':
        for domain in sys.argv[2:]:cache_images(domain)
