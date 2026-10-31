#!/usr/bin/env python3
"""Isolated four-GPU F + high-noise + complement-APR experiment.

The source checkpoint and prior rebuttal inputs are read-only. The pilot uses
disposable updates; formal training always starts again from source S.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
import torch
torch.set_num_threads(1)
import numpy as np
import mmcv
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, is_module_wrapper
from mmcv.runner import HOOKS, Hook, get_dist_info, init_dist
from mmtrack.apis import train_model
from mmtrack.datasets import build_dataloader
from mmtrack.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
import darth.models
import darth.datasets
from darth.models import build_model
from darth.datasets import build_dataset
from darth.core.hooks.ema import EMATrainHook

OLD = Path('/data1/liuyeqiang/damot_rebuttal_20261026')
NAS = Path('/data1/liuyeqiang/nas/A100/data1/LWR/vranlee/SERVER_ONLY/avis')
STAGES = ['overcast', 'rainy', 'dawn_dusk', 'night']
CLASSES = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
EXPECTED = {'overcast': (22, 4448), 'rainy': (17, 3150),
            'dawn_dusk': (33, 6476), 'night': (36, 7286)}


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    tmp.replace(path)


def torch_save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, str(tmp))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unwrap(model):
    return model.module if is_module_wrapper(model) else model


def strip_init(obj):
    if isinstance(obj, dict):
        obj.pop('init_cfg', None)
        obj.pop('pretrained', None)
        for val in obj.values():
            strip_init(val)
    elif isinstance(obj, (list, tuple)):
        for val in obj:
            strip_init(val)


def install_checkpointing():
    # F freezes all backbone BN statistics. Non-reentrant recomputation avoids
    # reentrant-checkpoint/DDP unused-parameter restrictions in torch 1.12.
    import torch.utils.checkpoint as cp
    if getattr(cp.checkpoint, '_damot_nonreentrant', False):
        return
    original = cp.checkpoint

    def checkpoint(function, *args, **kwargs):
        kwargs['use_reentrant'] = False
        return original(function, *args, **kwargs)
    checkpoint._damot_nonreentrant = True
    cp.checkpoint = checkpoint


def config(root, stage):
    return Config.fromfile(str(root / 'configs' / ('stage%d.py' % stage)))


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((OLD / 'manifests/datasets.json').read_text())
    frames_by_domain = {}
    for domain in STAGES:
        annotation = Path(inventory[domain]['annotation'])
        data = json.loads(annotation.read_text())
        assert (len(data['videos']), len(data['images'])) == EXPECTED[domain]
        by_video = {}
        for im in data['images']:
            video = im['file_name'].split('/')[0]
            by_video.setdefault(video, []).append(im['file_name'])
        view = root / 'data/images' / domain
        view.mkdir(parents=True, exist_ok=True)
        sources = {}
        for video, files in by_video.items():
            cache = OLD / 'cache/images' / domain
            local = all((cache / f).is_file() for f in files)
            split = inventory[domain]['split_by_video'][video]
            src = (cache / video if local else NAS /
                   'masa/data/bdd/bdd100k/images/track' / split / video)
            assert src.is_dir(), src
            dest = view / video
            if not dest.is_symlink():
                dest.symlink_to(src, target_is_directory=True)
            assert dest.resolve() == src.resolve()
            sources[video] = dict(source=str(src), local_cache=local,
                                  original_split=split, frames=len(files))
        frames_by_domain[domain] = dict(
            annotation=str(annotation), annotation_sha256=sha(annotation),
            frames=len(data['images']), videos=len(data['videos']),
            image_view=str(view), by_video=sources)
    base = Config.fromfile(str(OLD / 'configs/F_stage4_e5_embedded.py'))
    for stage, domain in enumerate(STAGES, 1):
        cfg = Config.fromfile(str(OLD / 'configs' /
                                 ('F_stage%d_e5_embedded.py' % stage)))
        original_model = copy.deepcopy(cfg.model)
        cfg.model.track_head.update(
            roi_proto_couple_ics=True, roi_proto_couple_beta=1.0,
            roi_proto_couple_basis='ics',
            roi_proto_couple_metric='projected_cosine')
        cfg.model.contrastive_pipeline.append(dict(
            type='SeqSPAFrequencyTransform', mode='high_freq_noise',
            apply_noise=True, noise_ratio=0.03))
        assert cfg.model.teacher_pipeline == original_model.teacher_pipeline
        assert cfg.model.student_pipeline == original_model.student_pipeline
        assert cfg.model.contrastive_pipeline[:-1] == original_model.contrastive_pipeline
        assert all(cfg.model.track_head[k] == v
                   for k, v in original_model.track_head.items())
        # Compute-memory option only; preserve 16 genuine samples/GPU, FP32.
        cfg.model.detector.backbone.with_cp = True
        strip_init(cfg.model)
        train = cfg.data.train if isinstance(cfg.data.train, list) else [cfg.data.train]
        for ds in train:
            ds.ann_file = frames_by_domain[domain]['annotation']
            ds.img_prefix = frames_by_domain[domain]['image_view'] + '/'
        cfg.data.samples_per_gpu = 16
        cfg.data.workers_per_gpu = 2
        cfg.data.persistent_workers = True
        cfg.gpu_ids = list(range(4))
        cfg.seed = int(base.seed)
        cfg.cudnn_benchmark = False
        cfg.find_unused_parameters = True
        cfg.load_from = None
        cfg.resume_from = None
        cfg.total_epochs = 5
        cfg.checkpoint_config = dict(interval=1, max_keep_ckpts=2,
                                     save_optimizer=True, sync_buffer=False)
        cfg.log_config.interval = 20
        cfg.work_dir = str(root / 'episode/adapt' /
                           ('stage%02d_after_%s' % (stage, domain)))
        cfg.custom_hooks = list(cfg.custom_hooks) + [
            dict(type='APRFiniteLossHook', priority='HIGH'),
            dict(type='APRExperimentStateHook', root=str(root), stage=stage,
                 priority='LOW')]
        cfg.dump(str(root / 'configs' / ('stage%d.py' % stage)))
    models = json.loads((OLD / 'manifests/checkpoints.json').read_text())
    source = next(x for x in models if x['model'] == 'S')
    assert sha(source['local']) == source['sha256']
    dump(root / 'manifests/data.json', frames_by_domain)
    dump(root / 'manifests/code_hashes.json', {
        str(p.relative_to(CODE)): sha(p) for p in sorted(CODE.rglob('*.py'))})
    dump(root / 'manifests/hardware_preflight.json', dict(
        hostname=subprocess.check_output(['hostname'], text=True).strip(),
        nvidia_smi=subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used',
            '--format=csv'], text=True),
        python=sys.executable, selected_physical_gpus=[3,4,5,6],
        uploaded_patch_sha256=sha(root / 'manifests/user_apr_coupling.patch')))
    dump(root / 'manifests/experiment.json', dict(
        version=1, source=source, target_order=STAGES, epochs_per_stage=5,
        seed=int(base.seed), precision='FP32', gpu_count=4,
        samples_per_gpu=16, nominal_global_batch=64,
        geometry='unchanged F teacher/student/contrastive geometry',
        low_frequency='unchanged F CPU pre-Normalize low_ratio=.10, keep=.90',
        high_frequency='contrastive pre-Normalize existing spatial noise mixture .03; not a Fourier high-pass filter',
        APR='normalize after ICS-complement projection, beta=1, top3, car-only, weight=.005; skip APR until a valid basis exists',
        ICS='unchanged proto_shrinkage_cov_ema, 2 PCs, trim=.10, min96, covEMA=.98, PCEMA=.9, shrink=.15',
        compute_changes=['backbone activation recomputation, non-reentrant',
                         'cuDNN benchmark disabled on V100'],
        checkpoint_retention='latest two epoch checkpoints per stage plus hard-linked stage endpoint; all rank-local banks/candidates saved',
        resume_limit='epoch-boundary recovery; data-loader worker RNG cannot be restored bit-exactly',
        evaluation='10 learned-domain cells; native student, thresholds .05/.30/.20, fusion, all-frame HOTA, no plots',
        comparison='historical F uses the same corrected metric protocol; HF and APR both change, so this is not an APR-only ablation'))
    dump(root / 'progress.json', dict(status='prepared', time=time.time()))
    print('PREPARED', root, flush=True)


def instrument_fap(model):
    transforms = []
    for name in ('student_pipeline', 'contrastive_pipeline'):
        for transform in getattr(model, name).transforms:
            if transform.__class__.__name__ != 'SeqSPAFrequencyTransform':
                continue
            fn = '_low_freq_mask' if transform.mode == 'low_freq_mask' else '_high_freq_noise'
            original = getattr(transform, fn)
            transform.audit_calls = 0

            def call(img, t=transform, orig=original):
                t.audit_calls += 1
                return orig(img)
            setattr(transform, fn, call)
            transforms.append((name, transform))
    model._fap_audit = transforms


def activity(model):
    m = unwrap(model)
    h = m.track_head
    b = h.causal_bank
    return dict(
        bank_initialized=bool(b.initialized.item()),
        ICS_num_updates=int(b.num_updates.item()),
        ICS_last_effective_rois=int(b.last_num_samples.item()),
        ICS_last_update_ok=bool(b.last_update_ok.item()),
        APR_calls=h._apr_couple_calls,
        APR_basis_ready_calls=h._apr_couple_basis_ready,
        APR_coupled_loss_active_calls=h._apr_couple_loss_active,
        APR_memory=h.roi_proto_memory.summary(),
        FAP=[dict(pipeline=n, mode=t.mode, calls=t.audit_calls)
             for n, t in m._fap_audit],
        allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
        reserved_MiB=torch.cuda.max_memory_reserved() / 2**20)


def build_loaded(root, cfg, checkpoint, source=False):
    model = build_model(copy.deepcopy(cfg.model))
    ck = torch.load(str(checkpoint), map_location='cpu')
    assert tuple(ck['meta']['CLASSES']) == CLASSES
    state = ck['state_dict']
    if source:
        params = dict(model.named_parameters())
        required = [n for n, p in params.items() if not n.startswith('teacher_')]
        missing = [n for n in required if n not in state]
        assert not missing, missing
        target = model.state_dict()
        unexpected = [n for n in state if n not in target]
        assert not unexpected, unexpected
        for n, tensor in state.items():
            assert target[n].shape == tensor.shape, n
            target[n] = tensor
        model.load_state_dict(target, strict=True)
        model.teacher_detector.load_state_dict(model.detector.state_dict(), strict=True)
        model.teacher_track_head.load_state_dict(model.track_head.state_dict(), strict=True)
    else:
        model.load_state_dict(state, strict=True)
    for n in ('detector.backbone.conv1.weight',
              'detector.roi_head.bbox_head.fc_cls.weight',
              'track_head.embed_head.fc_embed.weight'):
        assert torch.equal(model.state_dict()[n], state[n]), n
    model.CLASSES = CLASSES
    instrument_fap(model)
    rank, _ = get_dist_info()
    dump(root / 'manifests' / ('load_%s_rank%d.json' %
                             ('source' if source else Path(checkpoint).stem, rank)),
         dict(checkpoint=str(checkpoint), source_initialization=source,
              loaded_keys=len(state), teacher_initialized_from_student=source,
              critical_tensors_exact=True, checkpoint_epoch=ck['meta'].get('epoch')))
    del ck, state
    return model


def save_rank_state(root, model, stage, epoch, final=False):
    rank, _ = get_dist_info()
    m = unwrap(model)
    h = m.track_head
    state = dict(
        stage=stage, epoch=epoch, final=final,
        buffers={n: v.detach().cpu().clone() for n, v in m.named_buffers()},
        APR=h.roi_proto_memory.runtime_state_dict(),
        counters={n: getattr(h, n) for n in (
            '_proto_iter', '_apr_couple_calls', '_apr_couple_basis_ready',
            '_apr_couple_loss_active', '_causal_forward_count')},
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    tag = 'final' if final else 'epoch%d' % epoch
    path = root / 'episode/state' / ('stage%d_%s_rank%d.pth' % (stage, tag, rank))
    torch_save(path, state)
    dump(path.with_suffix('.json'), dict(
        path=str(path), stage=stage, epoch=epoch, rank=rank, final=final,
        activity=activity(m), time=time.time()))


@torch.no_grad()
def restore_rank_state(root, model, stage, epoch, final=False):
    rank, _ = get_dist_info()
    tag = 'final' if final else 'epoch%d' % epoch
    path = root / 'episode/state' / ('stage%d_%s_rank%d.pth' % (stage, tag, rank))
    state = torch.load(str(path), map_location='cpu')
    assert state['stage'] == stage and state['epoch'] == epoch
    m = unwrap(model)
    buffers = dict(m.named_buffers())
    for n, v in state['buffers'].items():
        buffers[n].copy_(v.to(buffers[n].device))
    m.track_head.roi_proto_memory.load_runtime_state_dict(state['APR'])
    for n, v in state['counters'].items():
        setattr(m.track_head, n, v)
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])


@HOOKS.register_module()
class APRFiniteLossHook(Hook):
    def after_train_iter(self, runner):
        if not torch.isfinite(runner.outputs['loss']).all():
            raise FloatingPointError('Non-finite loss before backward/optimizer')


@HOOKS.register_module()
class APRExperimentStateHook(Hook):
    def __init__(self, root, stage, recover_epoch=0):
        self.root, self.stage = Path(root), int(stage)
        self.recover_epoch = int(recover_epoch)

    def before_run(self, runner):
        if self.recover_epoch:
            restore_rank_state(self.root, runner.model, self.stage,
                               self.recover_epoch)

    def after_train_iter(self, runner):
        if runner.iter < 3 or (runner.iter + 1) % 20 == 0:
            rank, _ = get_dist_info()
            record = dict(status='training', stage=self.stage,
                          domain=STAGES[self.stage-1], epoch=runner.epoch+1,
                          iteration=runner.iter+1, pid=os.getpid(), rank=rank,
                          loss=float(runner.outputs['loss'].detach().item()),
                          activity=activity(runner.model), time=time.time())
            dump(self.root / ('progress_rank%d.json' % rank), record)
            if rank == 0:
                dump(self.root / 'progress.json', record)

    def after_train_epoch(self, runner):
        save_rank_state(self.root, runner.model, self.stage, runner.epoch+1)
        torch.distributed.barrier()
        if shutil.disk_usage(self.root).free < 1.5 * 2**30:
            raise RuntimeError('Less than 1.5 GiB local space remains after epoch save')


def init_training(root):
    from mmtrack.core import setup_multi_processes
    setup_multi_processes(config(root, 1))
    torch.backends.cudnn.benchmark = False
    init_dist('pytorch', backend='nccl')
    rank, world = get_dist_info()
    assert world == 4, world
    install_checkpointing()
    return rank, world


def pilot(root, steps):
    rank, world = init_training(root)
    cfg = config(root, 1)
    set_random_seed(int(cfg.seed))
    logger = get_root_logger(log_level='INFO')
    model = build_loaded(root, cfg, OLD / 'cache/weights/F_stage3_e5.pth')
    with open(OLD / 'results/exp4/reconstructed_history.pkl', 'rb') as f:
        history = pickle.load(f)
    model.set_current_target_domain('night')
    model.roi_proto_start_stage('night')
    model.track_head.roi_proto_memory.load_state_dict_memory(history)
    model.train()
    model = model.cuda()
    ddp = MMDistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                    broadcast_buffers=False,
                                    find_unused_parameters=True)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                **{k:v for k,v in cfg.optimizer.items() if k!='type'})
    ema = EMATrainHook(momentum=.002, interval=1, warm_up=100)
    runner = SimpleNamespace(model=ddp, iter=0)
    ema.before_run(runner)
    ds = build_dataset(cfg.data.train)
    loader = build_dataloader(ds, 16, 2, num_gpus=4, dist=True,
                              seed=int(cfg.seed), persistent_workers=False)
    torch.cuda.reset_peak_memory_stats()
    before = activity(model)
    for it, batch in enumerate(loader):
        if it >= steps:
            break
        runner.iter = it
        outputs = ddp.train_step(batch, optimizer)
        loss = outputs['loss']
        assert torch.isfinite(loss).all()
        optimizer.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), n
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 35.0)
        assert torch.isfinite(grad)
        optimizer.step()
        ema.after_train_iter(runner)
        torch.cuda.synchronize()
        after = activity(model)
        dump(root / 'manifests' / ('pilot_rank%d_step%d.json' % (rank,it+1)),
             dict(status='pilot_step_complete', rank=rank, loss=float(loss),
                  grad_norm=float(grad), activity=after,
                  disposable_weights=True, source='F_stage3_e5 + reconstructed real history',
                  input_domain='overcast', samples_per_gpu=16,
                  actual_num_samples=outputs['num_samples'],
                  time=time.time()))
        logger.info('PILOT rank=%d step=%d loss=%.6f allocated=%.1fMiB APR=%d',
                    rank, it+1, loss.item(), after['allocated_MiB'],
                    after['APR_coupled_loss_active_calls'])
    after = activity(model)
    assert after['APR_coupled_loss_active_calls'] > before['APR_coupled_loss_active_calls']
    assert all(x['calls'] > 0 for x in after['FAP'])
    torch.distributed.barrier()
    if rank == 0:
        dump(root / 'manifests/pilot_complete.json', dict(
            status='complete', steps=steps, ranks=4, formal_weights_saved=False,
            checks=['forward/loss/backward/optimizer/EMA finite',
                    'nonempty real APR history and active coupled loss',
                    'both low-frequency and high-noise transforms called',
                    'batch16/GPU, FP32, activation recomputation'], time=time.time()))
    torch.distributed.destroy_process_group()


def train(root):
    rank, world = init_training(root)
    intervention = (config(root, 1).model.track_head.causal_pca_mode
                    == 'intervention_residual')
    if intervention:
        assert json.loads((root / 'manifests/preflight.json').read_text())['status'] == 'passed'
    else:
        assert (root / 'manifests/pilot_complete.json').is_file()
    cfg0 = config(root, 1)
    set_random_seed(int(cfg0.seed))
    logger = get_root_logger(log_file=str(root / 'logs/episode_rank0.log'),
                             log_level='INFO')
    env = collect_env()
    if rank == 0:
        dump(root / 'manifests/environment.json', env)
    manifest = json.loads((root / 'manifests/experiment.json').read_text())
    model = build_loaded(root, cfg0, manifest['source']['local'], source=True)
    last = 0
    while (root / 'episode/state' / ('stage%d_complete.json' % (last+1))).exists():
        last += 1
    if last:
        ck = torch.load(str(root / 'episode/stage_snapshots' /
                           ('stage%02d_after_%s.pth' % (last, STAGES[last-1]))),
                        map_location='cpu')
        model.load_state_dict(ck['state_dict'], strict=True)
        del ck
        restore_rank_state(root, model, last, 5, final=True)
    for stage in range(last+1, 5):
        domain = STAGES[stage-1]
        cfg = config(root, stage)
        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
        recover = 0
        for ep in range(1, 6):
            ckpt = Path(cfg.work_dir) / ('epoch_%d.pth' % ep)
            states = [root / 'episode/state' /
                      ('stage%d_epoch%d_rank%d.pth' % (stage,ep,r)) for r in range(world)]
            if ckpt.exists() and all(p.exists() for p in states):
                recover = ep
        model.set_current_target_domain(domain)
        model.roi_proto_start_stage(domain)
        active_before = model.track_head._apr_couple_loss_active
        if recover:
            cfg.resume_from = str(Path(cfg.work_dir) / ('epoch_%d.pth' % recover))
            cfg.custom_hooks[-1]['recover_epoch'] = recover
        ds = build_dataset(cfg.data.train)
        model.CLASSES = CLASSES
        cfg.checkpoint_config.meta = dict(
            config=cfg.pretty_text, CLASSES=CLASSES, stage=stage,
            target_domain=domain, experiment='F_HF_APR_complement_cosine',
            seed=int(cfg.seed))
        if rank == 0:
            dump(root / 'manifests' / ('stage%d_dataset.json' % stage), dict(
                stage=stage, domain=domain, actual_training_samples=len(ds),
                original_annotation_counts=EXPECTED[domain],
                global_batch=64, resume_epoch=recover))
        logger.info('FORMAL STAGE %d domain=%s source_once=True samples=%d resume_epoch=%d',
                    stage, domain, len(ds), recover)
        if recover < 5:
            train_model(model, [ds], cfg, distributed=True, validate=False,
                        timestamp='apr_complement_20261028',
                        meta=dict(env_info=str(env), seed=int(cfg.seed)))
        else:
            ck = torch.load(cfg.resume_from, map_location='cpu')
            model.load_state_dict(ck['state_dict'], strict=True)
            del ck
            model.cuda()
            restore_rank_state(root, model, stage, recover)
        assert bool(model.track_head.causal_bank.initialized.item()), (
            'ICS never initialized', stage, rank)
        if stage > 1:
            assert model.track_head._apr_couple_loss_active > active_before, (
                'No coupled APR loss in this stage', stage, rank)
        model.roi_proto_finalize_stage(domain)
        save_rank_state(root, model, stage, 5, final=True)
        torch.distributed.barrier()
        if rank == 0:
            endpoint = root / 'episode/stage_snapshots' / (
                'stage%02d_after_%s.pth' % (stage,domain))
            endpoint.parent.mkdir(parents=True, exist_ok=True)
            source = Path(cfg.work_dir) / 'epoch_5.pth'
            if not endpoint.exists():
                os.link(str(source), str(endpoint))
            dump(root / 'episode/state' / ('stage%d_complete.json' % stage), dict(
                status='complete', stage=stage, domain=domain, epoch=5,
                checkpoint=str(endpoint), checkpoint_sha256=sha(endpoint),
                rank_local_memory_saved=True, time=time.time()))
        torch.distributed.barrier()
        del ds
        gc.collect()
        torch.cuda.empty_cache()
        if intervention and stage == 1:
            # Keep the exact in-process model/RNG state while the predeclared
            # frozen inference gate runs. No stage2 batch is built before PASS.
            torch.distributed.barrier()
            if rank == 0:
                dump(root / 'episode/stage1_gate_waiting.json', dict(
                    status='waiting', stage=1, physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    checkpoint=str(root / 'episode/stage_snapshots/stage01_after_overcast.pth'),
                    time=time.time()))
            decision_path = root / 'episode/stage1_gate.json'
            while not decision_path.exists():
                time.sleep(2)
            decision = json.loads(decision_path.read_text())
            assert decision['status'] in ('pass', 'stop')
            torch.distributed.barrier()
            if decision['status'] == 'stop':
                if rank == 0:
                    dump(root / 'episode/train_stopped.json', dict(
                        status='stopped_by_prespecified_stage1_gate',
                        completed_stages=1, decision=decision, time=time.time()))
                torch.distributed.destroy_process_group()
                return
    if rank == 0:
        dump(root / 'episode/train_complete.json', dict(
            status='complete', stages=4, epochs_per_stage=5,
            source_loaded_from=manifest['source']['local'], time=time.time()))
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['prepare', 'pilot', 'train'], required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    root = args.run_root.resolve()
    try:
        if args.mode == 'prepare':
            prepare(root)
        elif args.mode == 'pilot':
            pilot(root, args.steps)
        else:
            train(root)
    except BaseException as exc:
        rank = int(os.environ.get('RANK', 0))
        dump(root / 'logs' / ('%s_failure_rank%d_%d.json' %
                             (args.mode, rank, int(time.time()))), dict(
            error=repr(exc), argv=sys.argv, pid=os.getpid(), time=time.time()))
        raise


if __name__ == '__main__':
    main()
