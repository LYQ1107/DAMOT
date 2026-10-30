"""Twelve missing legacy-protocol cells, native original four-GPU evaluation."""
from common import *
import concurrent.futures
import socket
import subprocess
import time
from mmcv import Config

PY = str(RUN / 'env/bin/python')
B = Path('/data1/liuyeqiang/idapr_20261030_153059/B3stage/nas/A100/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/work_dirs/bdd_4domain')
roots = {'SPA_pipeline': B / 'spa_fix/spa_pipeline_mild_4domain_5e',
         'SPA_APR': B / 'module_combinations/spaP1_roiV1_4domain_5e'}
domains = ['overcast', 'rainy', 'dawn_dusk', 'night']
for domain in domains:
    cfg = Config.fromfile(str(RUN / 'code/configs/mot/qdtrack/bdd_7domain' / ('qdtrack_faster-rcnn_r50_fpn_20e_bdd_%s.py' % domain)))
    strip_init(cfg.model)
    cfg.cudnn_benchmark = True
    cfg.data.workers_per_gpu = 6
    cfg.model.tracker.init_score_thr = .30
    cfg.data.test.ann_file = str(RUN / 'data/annotations' / (domain + '.json'))
    cfg.data.test.img_prefix = str(Q / 'data_views' / domain) + '/'
    cfg.data.test.classes = CLASSES
    cfg.data.test.test_mode = True
    cfg.dump(str(Q / 'configs' / ('historical_eval_%s.py' % domain)))
tasks = [(stage, domain) for stage in range(2, 5) for domain in domains[:stage - 1]]


def run_group(group, preference):
    for stage, domain in tasks:
        name = 'historical_%s_s%d_%s' % (group, stage, domain)
        target = Q / 'historical_inference' / group / ('stage%d' % stage) / domain
        if (target / 'complete.json').exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        with socket.socket() as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        checkpoint = roots[group] / 'episode/adapt' / ('stage%02d_after_%s' % (stage, domains[stage - 1])) / 'epoch_5.pth'
        args = [PY, '-m', 'torch.distributed.launch', '--nproc_per_node=4', '--master_port=' + str(port),
                str(Q / 'code_hf/tools/run/test.py'), str(Q / 'configs' / ('historical_eval_%s.py' % domain)),
                '--checkpoint', str(checkpoint), '--launcher', 'pytorch', '--work-dir', str(target),
                '--exp-name', 'native', '--version', 'epoch5', '--eval', 'bbox', 'track', '--fuse-conv-bn']
        cmd = [sys.executable, str(Q / 'scripts/gpu_job.py'), '--name', name, '--gpus', '4',
               '--prefer', preference, '--'] + args
        dump(target / 'job.json', dict(command=cmd, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
             stage=stage, epoch=5, domain=domain, group=group, protocol='original legacy evaluator, native four-GPU test.py, cudnn_benchmark=True, fusion, init=.30'))
        with (Q / 'logs' / (name + '.log')).open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        output = target / 'native/epoch5'
        summary = next(r for r in csv.DictReader((output / 'summary_metrics.csv').open()) if r[''] == 'OVERALL')
        assert all(float(summary[m]) == float(summary[m]) for m in ['HOTA', 'DetA', 'IDF1', 'AssA'])
        dump(target / 'complete.json', dict(status='complete', stage=stage, epoch=5, domain=domain,
             group=group, summary=str(output / 'summary_metrics.csv'), time=time.time()))
        print('HISTORICAL_CELL_COMPLETE', group, stage, domain, flush=True)


if __name__ == '__main__':
    dump(Q / 'manifests/historical_inference_state.json', dict(status='running', pid=os.getpid(), cells=12))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run_group, 'SPA_pipeline', '2,3,4,5,6,7,8,9'),
                       pool.submit(run_group, 'SPA_APR', '6,7,8,9,2,3,4,5')]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        dump(Q / 'manifests/historical_inference_state.json', dict(status='complete', cells=12))
    except Exception:
        import traceback
        dump(Q / 'manifests/historical_inference_state.json', dict(status='failed', error=traceback.format_exc()))
        raise
