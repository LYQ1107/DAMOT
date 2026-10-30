"""Compare saved interventions to independent native forwards without detector reuse."""
import argparse,copy,pickle
from functools import partial
from common import *
from infer import equal
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--domain',required=True);a=p.parse_args()
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import collate,scatter
from torch.utils.data import DataLoader,Subset
from darth.models import build_model
from darth.datasets import build_dataset
from darth.models.track_heads.ics_inference_track_head import ICSInferenceTrackHead
torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35,0);torch.backends.cudnn.benchmark=False
torch.multiprocessing.set_start_method('fork',force=True);seed()
specs=cases(a.model);cfg=Config.fromfile(str(ROOT/'results'/specs[0]['label']/a.domain/'used_config.py'));strip_init(cfg.model)
ds=build_dataset(cfg.data.test);vid=ds.data_infos[0]['video_id'];ids=[i for i,im in enumerate(ds.data_infos) if im['video_id']==vid][:16]
ck=torch.load(str(checkpoint(a.model)),map_location='cpu');m=build_model(copy.deepcopy(cfg.model));target=m.state_dict()
state={k:v for k,v in ck['state_dict'].items() if k in target};assert set(state)==set(target);m.load_state_dict(state,strict=True)
m.track_head.load_checkpoint_bank(ck['state_dict'].get('track_head.causal_bank.pcs'))
del state,ck,target;m.CLASSES=CLASSES;m=fuse_conv_bn(m).cuda().eval()
trackers={c['label']:copy.deepcopy(m.tracker) for c in specs};saved={}
for c in specs:
    with (ROOT/'results'/c['label']/a.domain/'predictions'/('video_%s.pkl'%vid)).open('rb') as f:saved[c['label']]=pickle.load(f)['outputs']
loader=DataLoader(Subset(ds,ids),batch_size=1,shuffle=False,num_workers=2,collate_fn=partial(collate,samples_per_gpu=1))
with torch.no_grad():
    for pos,batch in enumerate(loader):
        data=scatter(batch,[0])[0]
        for c in specs:
            m.tracker=trackers[c['label']]
            m.track_head.set_inference_intervention(beta=c['beta'],basis_source=c['basis_source'],npy_path=c.get('npy_path'),enabled=c['beta']!=0)
            native=m(return_loss=False,rescale=True,**data)
            assert equal(native,saved[c['label']][pos]),('Detector reuse changed predictions',a.model,a.domain,c['label'],pos)
dump(ROOT/'manifests'/('reuse_%s_%s.json'%(a.model,a.domain)),dict(status='PASS',frames_per_case=len(ids),
     cases=[c['label'] for c in specs],comparison='Independent native detector forward per case vs saved detector-reuse outputs; dtype/shape/bytes'))
print('INDEPENDENT_NATIVE_REUSE_CHECK_PASS',a.model,a.domain,len(ids),len(specs),flush=True)
