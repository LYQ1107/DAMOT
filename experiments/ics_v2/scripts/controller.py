"""Finish historical inference, then five APR-off, single-seed Night runs."""
from common import *
import concurrent.futures
import socket
import subprocess
import time
import traceback

PY = str(RUN / 'env/bin/python')
GROUPS = ['C0', 'C1', 'C4', 'C2', 'C3']


def state(status, **kwargs):
    dump(Q / 'manifests/controller.json', dict(status=status, pid=os.getpid(), time=time.time(), **kwargs))


def job(name, args, count, preference):
    command = [sys.executable, str(Q / 'scripts/gpu_job.py'), '--name', name, '--gpus', str(count),
               '--prefer', preference, '--'] + args
    with (Q / 'logs' / (name + '.log')).open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        dump(Q / 'manifests' / (name + '_dispatch.json'), dict(pid=process.pid, command=command, time=time.time()))
        code = process.wait()
    if code:
        raise RuntimeError('%s failed: %s' % (name, code))


def run_group(group):
    root = Q / 'runs' / group
    for record in json.loads((Q / 'manifests/code_seal.json').read_text())['files']:
        assert sha(Q / 'code_hf' / record['relative_path']) == record['sha256']
    with socket.socket() as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    args = [PY, '-m', 'torch.distributed.launch', '--nproc_per_node=4', '--master_port=' + str(port),
            str(Q / 'code_hf/tools/run/train_seq3_episode.py'), '--base-config', str(root / 'base.py'),
            '--source-ckpt', INPUTS['source_checkpoint'], '--target-order', 'overcast', 'rainy', 'dawn_dusk', 'night',
            '--eval-domains', 'overcast', 'rainy', 'dawn_dusk', 'night', '--work-dir', str(root / 'episode'),
            '--epochs-per-stage', '2', '--resume-episode-from-stage', '4', '--samples-per-gpu', '16',
            '--workers-per-gpu', '2', '--launcher', 'pytorch', '--no-validate', '--seed', '1790355234',
            '--target-config-template', str(Q / 'configs/night_target.py')]
    preference = '2,3,4,5,6,7,8,9' if group in ['C0','C4','C3'] else '6,7,8,9,2,3,4,5'
    job(group + '_train', args, 4, preference)
    assert (root / 'episode/adapt/stage04_after_night/epoch_2.pth').is_file()
    assert all((root / ('v2_end_rank%d.json' % rank)).is_file() for rank in range(4))
    removed = []
    # Only redundant files created by this task; final weights are already saved.
    for rel in ['episode/stage_snapshots/stage00_source_only.pth', 'episode/adapt/stage04_after_night/epoch_1.pth']:
        path = root / rel
        if path.is_file() and not path.is_symlink():
            removed.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
            path.unlink()
    dump(root / 'redundant_weight_cleanup.json', removed)
    def evaluate(domain):
        job(group + '_eval_' + domain, [PY, str(Q / 'scripts/evaluate.py'), '--model', 'HF',
            '--domain', domain, '--checkpoint', str(root / 'episode/adapt/stage04_after_night/epoch_2.pth'),
            '--out', str(Q / 'evaluation' / group / domain), '--beta', '0'], 1, preference)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(evaluate, d) for d in ['night','dawn_dusk']]):
            future.result()


def main():
    assert json.loads((Q / 'manifests/preflight.json').read_text())['status'] == 'passed'
    while True:
        current = json.loads((Q / 'manifests/historical_inference_state.json').read_text())
        if current['status'] == 'failed':
            raise RuntimeError('Historical inference failed; inspect its preserved logs')
        if current['status'] == 'complete':
            break
        state('waiting_for_historical_inference', groups=GROUPS, seed=1790355234, APR_enabled=False)
        time.sleep(15)
    subprocess.run([sys.executable, str(Q / 'scripts/summarize_historical_completed.py')], check=True)
    state('formal_training_and_evaluation', groups=GROUPS, seed=1790355234, APR_enabled=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run_group, g) for g in GROUPS]):
            future.result()
    subprocess.run([PY, str(Q / 'scripts/validate_completed.py')], check=True)
    subprocess.run([sys.executable, str(Q / 'scripts/summarize.py')], check=True)
    state('experiments_complete_awaiting_publication', groups=GROUPS, seed=1790355234)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        state('failed', error=traceback.format_exc())
        raise
