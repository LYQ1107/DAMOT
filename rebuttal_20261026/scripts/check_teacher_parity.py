#!/usr/bin/env python3
from runtime import *
from darth.models.utils.transforms import detections2bboxes,filter_bboxes_by_confidence
fixed_seed();m,c=load_model('F_stage3_e5',deploy=False)
r=json.loads((RUN/'manifests/night_frames.json').read_text())[0]
dst=RUN/'cache/images/night'/r['file_name']
im=dict(file_name=r['file_name'],frame_id=r['frame_id'],video_id=r['video_id'],height=720,width=1280)
x,meta=preprocessed_frame(dst,im);raw,rawmeta=preprocessed_frame(dst,im,training_base=True)
t=m.teacher_detector
with torch.no_grad():
    f=t.extract_feat(x.cuda());prop=t.rpn_head.simple_test_rpn(f,meta);manual=t.roi_head.simple_test(f,prop,meta,rescale=False)
    orig=t.roi_head.simple_test;captured=[]
    def capture(*a,**kw):
        o=orig(*a,**kw);captured.append(copy.deepcopy(o));return o
    t.roi_head.simple_test=capture
    empty=[torch.empty((0,4),device='cuda')];lab=[torch.empty(0,dtype=torch.long,device='cuda')]
    m._prepare_views(raw.cuda(),rawmeta,empty,lab)
    assert len(captured)==1
    max_coord=max_score=0.
    for a,b in zip(manual[0],captured[0][0]):
        assert a.shape==b.shape
        if a.size:
            max_coord=max(max_coord,float(np.max(np.abs(a[:,:4]-b[:,:4]))));max_score=max(max_score,float(np.max(np.abs(a[:,4]-b[:,4]))))
            np.testing.assert_allclose(a[:,:4],b[:,:4],rtol=0,atol=.001)
            np.testing.assert_allclose(a[:,4],b[:,4],rtol=0,atol=1e-6)
    a,b,s=detections2bboxes(manual);a,b,s=filter_bboxes_by_confidence(a,b,s,c.model.conf_thr)
    aa,bb,ss=detections2bboxes(captured[0]);aa,bb,ss=filter_bboxes_by_confidence(aa,bb,ss,c.model.conf_thr)
    np.testing.assert_array_equal(b[0],bb[0]);assert len(a[0])==len(aa[0])
    dump(RUN/'manifests/teacher_preflight_parity.json',dict(pass_parity=True,checkpoint='F_stage3_e5',image_id=r['image_id'],branch='teacher_detector',manual_flow='extract_feat -> RPN -> RoI -> strict score>0.7, before teacher_pipeline',same_as_archived_prepare_views=True,filtered_count=len(a[0]),maximum_coordinate_error_pixels=max_coord,maximum_score_error=max_score,tolerance='0.001 pixels and 1e-6 confidence (FP32 normalization order)',coordinate='base resized xyxy; invert scale_factor for original GT',scale_factor=meta[0]['scale_factor'].tolist(),teacher_eval=True,no_EMA=True,no_backward=True))
    print('PARITY PASS',len(a[0]),max_coord,max_score)
