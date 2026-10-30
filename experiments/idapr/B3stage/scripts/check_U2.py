"""Compare original and full-mode model forwards on one serialized batch."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

Q = Path(__file__).resolve().parents[1]
INPUTS = json.loads((Q / 'manifests/inputs.json').read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def forward(implementation):
    code = Q / ('code_original' if implementation == 'original' else 'code_hf')
    sys.path.insert(0, str(code))
    from common import strip_init, seed, install_checkpointing
    import torch
    import cv2
    from mmcv import Config
    from mmcv.parallel import scatter
    import darth.models
    import darth.datasets
    from darth.models import build_model
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    seed(1790355234)
    cfg = Config.fromfile(INPUTS['HF_base_config'])
    cfg.model.track_head.roi_proto_couple_ics = True
    if implementation == 'modified':
        cfg.model.track_head.roi_proto_space_mode = 'full'
    strip_init(cfg.model)
    install_checkpointing()
    torch.backends.cudnn.benchmark = bool(cfg.get('cudnn_benchmark', False))
    model = build_model(copy.deepcopy(cfg.model))
    ck = torch.load(INPUTS['HF_stage3_snapshot'], map_location='cpu')
    model.load_state_dict(ck['state_dict'], strict=True)
    del ck
    model.set_current_target_domain('night')
    model.roi_proto_start_stage('night')
    history = torch.load(INPUTS['shared_apr_history'], map_location='cpu')
    model.track_head.roi_proto_memory.load_state_dict_memory(history)
    assert len(model.track_head.roi_proto_memory._history) == 3
    model.cuda()
    model.train()
    model.teacher_detector.eval()
    model.teacher_track_head.eval()
    batch = torch.load(str(Q / 'U/U2_batch.pth'), map_location='cpu')
    batch = scatter(batch, [0])[0]
    seed(1790355234)
    with torch.no_grad():
        losses = model.forward_train(**batch)
        total, log_vars = model._parse_losses(losses)
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, (tuple, list)):
            return [cpu(v) for v in value]
        raise TypeError(type(value))
    tensors = {key: cpu(value) for key, value in losses.items()}
    tensors['TOTAL_LOSS'] = total.detach().cpu()
    assert 'loss_roi_proto' in tensors and float(tensors['loss_roi_proto']) != 0
    torch.save(tensors, str(Q / 'U' / ('U2_' + implementation + '_losses.pth')))
    write(Q / 'U' / ('U2_' + implementation + '.json'),
          dict(implementation=implementation, history_domains=3, log_vars=log_vars,
               train_mode=True, teacher_eval=True, no_grad=True,
               source_snapshot=INPUTS['HF_stage3_snapshot'], full_state_strict_load=True))
    print('U2_FORWARD_COMPLETE', implementation, log_vars, flush=True)


def driver():
    sys.path.insert(0, str(Q / 'code_hf'))
    from common import config, seed, training_data
    import torch
    import cv2
    import darth.models
    import darth.datasets
    from darth.datasets import build_dataset
    from mmtrack.datasets import build_dataloader
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    # Retain a small context on the one allocated GPU across the two separate
    # interpreters, so another task does not see a false idle gap between them.
    reservation = torch.zeros(1, device='cuda')
    cfg = config('HF')
    seed(1790355234)
    dataset = build_dataset(training_data(cfg, 'night'))
    loader = build_dataloader(dataset, samples_per_gpu=16, workers_per_gpu=2,
                             num_gpus=1, dist=False, shuffle=True,
                             seed=1790355234, persistent_workers=False)
    batch = next(iter(loader))
    batch_path = Q / 'U/U2_batch.pth'
    torch.save(batch, str(batch_path))
    hasher = hashlib.sha256()
    with batch_path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    batch_info = dict(sha256=hasher.hexdigest(), bytes=batch_path.stat().st_size,
                      samples_per_gpu=16, workers_per_gpu=2, seed=1790355234,
                      pipeline='unaltered archived HF Night train pipeline',
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(Q / 'U/U2_batch_provenance.json', batch_info)
    del batch, loader, dataset
    gc.collect()
    for implementation in ['original', 'modified']:
        with (Q / 'logs' / ('U2_' + implementation + '.log')).open('w') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--forward',
                            implementation], stdout=log, stderr=subprocess.STDOUT, check=True)
    old = torch.load(str(Q / 'U/U2_original_losses.pth'), map_location='cpu')
    new = torch.load(str(Q / 'U/U2_modified_losses.pth'), map_location='cpu')
    checks = []
    def compare(name, a, b):
        if isinstance(a, list):
            assert len(a) == len(b)
            for i, (aa, bb) in enumerate(zip(a, b)):
                compare(name + '[%d]' % i, aa, bb)
        else:
            equal = bool(torch.equal(a, b))
            checks.append(dict(name=name, bitwise_equal=equal,
                               max_abs_delta=float((a - b).abs().max())))
    assert set(old) == set(new)
    for key in old:
        compare(key, old[key], new[key])
    result = dict(status='passed' if all(r['bitwise_equal'] for r in checks) else 'failed',
                  same_serialized_batch=batch_info, full_mode=True,
                  original_HF_coupling=True, checks=checks,
                  APR_loss_original=float(old['loss_roi_proto']),
                  APR_loss_modified=float(new['loss_roi_proto']),
                  total_original=float(old['TOTAL_LOSS']), total_modified=float(new['TOTAL_LOSS']))
    write(Q / 'U/U2.json', result)
    assert result['status'] == 'passed', result
    batch_path.unlink()  # only this task's temporary raw input batch
    print('U2_PASSED', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--forward', choices=['original', 'modified'])
    args = parser.parse_args()
    if args.forward:
        forward(args.forward)
    else:
        driver()
