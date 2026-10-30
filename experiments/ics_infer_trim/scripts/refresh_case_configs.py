"""Document each case's runtime settings, including the original two live jobs."""
from common import *
from mmcv import Config
for m in ['C0','C2','C4']:
    for d in DOMAINS:
        source=ROOT/'jobs'/m/d/'provenance.json'
        if not source.exists():continue
        provenance=json.loads(source.read_text())
        for case in cases(m):
            file=ROOT/'results'/case['label']/d/'used_config.py'
            if not file.exists():continue
            cfg=Config.fromfile(str(file))
            cfg.model.track_head.update(inference_trim_enabled=case['beta']!=0.,inference_trim_beta=case['beta'],
                                        basis_source=case['basis_source'],inference_trim_basis=case.get('npy_path'))
            if provenance['config_alpha'] is not None:cfg.model.track_head.causal_trim_alpha=provenance['config_alpha']
            cfg.dump(str(file))
print('CASE_CONFIGS_MATCH_RUNTIME_SETTINGS',flush=True)
