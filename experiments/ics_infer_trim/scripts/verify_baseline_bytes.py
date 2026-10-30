"""Verify dtype, shape and actual bytes, including jobs launched before this audit."""
import pickle
from common import *
from infer import equal
from mmcv import Config
records=[]
for model in ['C0','C2','C4']:
    for domain in DOMAINS:
        file=ROOT/'jobs'/model/domain/'inference_complete.json';assert file.exists()
        new=ROOT/'results'/(model+'_b0')/domain/'raw_results.pkl'
        source=CROOT/'evaluation'/model/domain/'raw_results.pkl'
        with source.open('rb') as f:a=pickle.load(f)
        with new.open('rb') as f:b=pickle.load(f)
        assert equal(a,b),('Not byte-for-byte equal',model,domain)
        n=len(a['track_bboxes']);del a,b
        cfgfile=CROOT/'evaluation'/model/domain/'used_config.py';cfg=Config.fromfile(str(cfgfile))
        records.append(dict(model=model,domain=domain,frames=n,arrays_dtype_shape_and_bytes_equal=True,
                            original_pickle_sha256=sha(source),new_pickle_sha256=sha(new),
                            original_evaluation_config_sha256=sha(cfgfile),annotation=str(cfg.data.test.ann_file),annotation_sha256=sha(cfg.data.test.ann_file)))
        r=read_json(file);r['bytewise_audit']='PASS: dtype, shape and array bytes; entire domain';dump(file,r)
        print('BASELINE_BYTES_PASS',model,domain,n,flush=True)
dump(ROOT/'manifests/baseline_bytes.json',dict(status='PASS',model_frames=sum(r['frames'] for r in records),jobs=records,
     inference_head_sha256=sha(CODE/'darth/models/track_heads/ics_inference_track_head.py'),
     comparison='All detection/track output array dtypes, shapes and byte representations, independent of pickle metadata or serialization sharing'))
