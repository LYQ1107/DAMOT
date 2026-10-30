"""U gates, two single-seed four-card runs, then full-domain evaluations."""
from common import *
import concurrent.futures
import socket
import subprocess
import time
import traceback

PY = str(RUN / 'env/bin/python')


def state(status, **kwargs):
    dump(Q / 'manifests/controller.json', dict(status=status, pid=os.getpid(), time=time.time(), **kwargs))


def wait_validation():
    required = ['U1', 'U2', 'U3', 'extractor_parity', 'warmstart']
    while True:
        for name in ['U2', 'U3_smoke', 'extractor_parity', 'teacher_extract']:
            p = Q / 'manifests' / (name + '_job.json')
            if p.exists() and json.loads(p.read_text())['status'] == 'failed':
                raise RuntimeError('Quick verification job failed: ' + name)
        if all((Q / 'U' / (name + '.json')).exists() for name in required):
            checks = {name: json.loads((Q / 'U' / (name + '.json')).read_text()) for name in required}
            assert all(r['status'] == 'passed' for r in checks.values()), checks
            jobs_done = all(json.loads((Q / 'manifests' / (name + '_job.json')).read_text())['status'] == 'complete'
                            for name in ['U2', 'U3_smoke', 'extractor_parity', 'teacher_extract'])
            if jobs_done:
                dump(Q / 'U/all_passed.json', dict(status='passed', names=required,
                     seed=1790355234, formal_training_may_start=True, time=time.time()))
                return
        state('waiting_for_U_checks', required=required)
        time.sleep(10)


def job(name, args, gpus=1, prefer='2,3,4,5,6,7,8,9'):
    cmd = [sys.executable, str(Q / 'scripts/gpu_job.py'), '--name', name,
           '--gpus', str(gpus), '--prefer', prefer, '--'] + args
    with (Q / 'logs' / (name + '.log')).open('w') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        dump(Q / 'manifests' / (name + '_dispatch.json'), dict(pid=proc.pid, command=cmd, time=time.time()))
        code = proc.wait()
    if code:
        raise RuntimeError('%s failed, exit=%s' % (name, code))


def run_group(group):
    root = Q / 'runs' / group
    with socket.socket() as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    args = [PY, '-m', 'torch.distributed.launch', '--nproc_per_node=4', '--master_port=' + str(port),
            str(Q / 'code_hf/tools/run/train_seq3_episode.py'), '--base-config', str(root / 'base.py'),
            '--source-ckpt', INPUTS['source_checkpoint'], '--target-order', 'overcast', 'rainy', 'dawn_dusk', 'night',
            '--eval-domains', 'overcast', 'rainy', 'dawn_dusk', 'night', '--work-dir', str(root / 'episode'),
            '--epochs-per-stage', '2', '--resume-episode-from-stage', '4', '--samples-per-gpu', '16',
            '--workers-per-gpu', '2', '--launcher', 'pytorch', '--no-validate', '--seed', '1790355234',
            '--target-config-template', str(Q / 'configs/night_target.py'),
            '--apr-history', str(Q / 'shared_apr_history.pth')]
    preference = '2,3,4,5,6,7,8,9' if group == 'B3' else '6,7,8,9,2,3,4,5'
    job(group + '_train', args, gpus=4, prefer=preference)
    def evaluate(domain):
        job(group + '_eval_' + domain, [PY, str(Q / 'scripts/evaluate.py'), '--model', 'HF',
            '--domain', domain, '--checkpoint', str(root / 'episode/adapt/stage04_after_night/epoch_2.pth'),
            '--out', str(Q / 'evaluation' / group / domain), '--beta', '0'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for future in concurrent.futures.as_completed([pool.submit(evaluate, d)
                for d in ['night', 'overcast', 'rainy', 'dawn_dusk']]):
            future.result()
    # Only this task's redundant generated weights; all four evaluations done.
    cleanup = []
    for rel in ['episode/stage_snapshots/stage00_source_only.pth',
                'episode/adapt/stage04_after_night/epoch_1.pth']:
        path = root / rel
        if path.is_file() and not path.is_symlink():
            cleanup.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
            path.unlink()
    dump(root / 'redundant_weight_cleanup.json', cleanup)


def main():
    wait_validation()
    state('formal_training_and_evaluation', groups=['B3', 'R'], single_seed=1790355234,
          policy='four wholly idle V100s per group; two concurrent groups only when eight are idle')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run_group, group) for group in ['B3', 'R']]):
            future.result()
    for name in ['B0_eval_overcast', 'B0_eval_rainy', 'HF_stage3_eval_overcast', 'HF_stage3_eval_rainy']:
        assert json.loads((Q / 'manifests' / (name + '_job.json')).read_text())['status'] == 'complete'
    subprocess.run([PY, str(Q / 'scripts/validate_completed.py')], check=True,
                   env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MKL_THREADING_LAYER='GNU'))
    subprocess.run([sys.executable, str(Q / 'scripts/summarize.py')], check=True)
    state('experiments_complete_awaiting_publication')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        state('failed', error=traceback.format_exc())
        raise
