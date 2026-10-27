#!/usr/bin/env python3
"""Interleaved same-process retest of FAP and FAP+ICS throughput.

This deliberately does not overwrite the primary exp4 results.  It reuses the
same fixed Night frame stream and the same profile_damot setup/train_step path,
but alternates the two groups in one process so process-restart and group-order
effects are not folded into the module comparison.
"""
import csv
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / "scripts"))

import profile_damot as p  # noqa: E402


OUT = RUN / "results" / "exp4" / "retest_fap_ics_20261027"
RAW = OUT / "raw_windows"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False))
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def gpu_snapshot():
    # GPU1 currently has a driver-side stale NVML row for a dead process
    # ([Not Found], PID absent from /proc).  It is recorded explicitly and is
    # allowed; any live peer process still aborts the timing window.
    apps_raw = subprocess.check_output(
        ["nvidia-smi", "--id=" + os.environ["CUDA_VISIBLE_DEVICES"],
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"], text=True)
    stale = []
    live = []
    for line in apps_raw.splitlines():
        parts = [x.strip() for x in line.split(",", 2)]
        if len(parts) != 3:
            continue
        pid, name, memory = parts
        if pid == str(os.getpid()):
            continue
        if name == "[Not Found]" and not Path("/proc") .joinpath(pid).exists():
            stale.append(dict(pid=pid, process_name=name, used_memory=memory))
        else:
            live.append(dict(pid=pid, process_name=name, used_memory=memory))
    assert not live, ("GPU acquired by another live process", live)
    return {
        "nvidia_smi": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
             "--format=csv,noheader"], text=True),
        "compute_apps_raw": apps_raw,
        "stale_not_found_apps": stale,
        "live_peer_apps": live,
    }


def load_records():
    src = RUN / "results" / "exp4" / "adaptation_frame_manifest.json"
    data = json.loads(src.read_text())
    records = data["frames"]
    assert len(records) == 512
    return records, src


def one_window(group, repeat, order_index, records, batch=4,
               warmup=20, measured=100):
    # setup() loads identical F_stage3_e5 weights and creates a fresh optimizer
    # and EMA state for every window.  FAP is retained for both groups; only the
    # causal direction bank differs.
    model, opt, ema, runner, stats = p.setup(group)
    loader = p.loader_for(records, "night", batch, 2000)
    it = iter(loader)
    try:
        for i in range(warmup):
            p.train_step(model, opt, ema, runner, next(it), i)
        p.torch.cuda.synchronize()
        p.torch.cuda.reset_peak_memory_stats()
        before = {k: (len(v) if isinstance(v, list) else v)
                  for k, v in stats.items()}
        hardware_before = gpu_snapshot()
        load_before = os.getloadavg()
        start = time.perf_counter()
        losses = []
        for i in range(warmup, warmup + measured):
            losses.append(p.train_step(model, opt, ema, runner, next(it), i))
        p.torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert elapsed >= 30.0, elapsed
        hardware_after = gpu_snapshot()
        activity = {
            k: (v[before[k]:] if isinstance(v, list) else v - before[k])
            for k, v in stats.items()
        }
        finite = all(p.np.isfinite(v) for row in losses for v in row.values())
        assert finite
        row = dict(
            mode="adaptation_retest_interleaved",
            group=group,
            repeat=repeat,
            order_index=order_index,
            batch_pairs=batch,
            warmup_steps=warmup,
            measured_steps=measured,
            window_seconds=elapsed,
            step_per_second=measured / elapsed,
            key_reference_pair_per_second=measured * batch / elapsed,
            key_frame_per_second=measured * batch / elapsed,
            peak_allocated_MiB=p.torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_MiB=p.torch.cuda.max_memory_reserved() / 2**20,
            FP32=True,
            gpu="Tesla V100-DGXS-32GB",
            conv_bn_fusion=False,
            finite_losses=finite,
            ics_covariance_calls=activity["covariance_calls"],
            ics_eigh_calls=activity["eigh_calls"],
        )
        write_json(RAW / ("%02d_%s_repeat%d.json" %
                          (order_index, group, repeat)), dict(
            summary=row,
            activity=activity,
            hardware_before=hardware_before,
            hardware_after=hardware_after,
            load_average_before=load_before,
            load_average_after=os.getloadavg(),
            timing_boundary=("next(data) through H2D, teacher/student/loss, "
                             "backward, clip, SGD and EMA; CUDA synchronized "
                             "only at window boundaries"),
            losses=losses,
        ))
        return row
    finally:
        del model, opt, ema, runner, loader, it
        gc.collect()
        p.torch.cuda.empty_cache()


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None
    p.install_fixed_fap_rng()
    p.torch.backends.cudnn.benchmark = False
    records, manifest = load_records()
    schedule = ["FAP", "FAP_ICS", "FAP", "FAP_ICS", "FAP", "FAP_ICS"]
    write_json(OUT / "manifest.json", dict(
        exit_code=None,
        host=os.uname().nodename,
        pid=os.getpid(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allow_dead_nvml_row=True,
        input_manifest=str(manifest),
        input_frame_count=len(records),
        schedule=schedule,
        same_process=True,
        batch_pairs=4,
        warmup_steps=20,
        measured_steps=100,
        repetitions_per_group=3,
        fp32=True,
        cudnn_benchmark=False,
        no_primary_results_overwritten=True,
    ))
    rows = []
    for idx, group in enumerate(schedule):
        rep = sum(1 for r in rows if r["group"] == group)
        row = one_window(group, rep, idx, records)
        rows.append(row)
        write_csv(OUT / "raw_summary.csv", rows)
        print("RETEST", row, flush=True)
    summary = []
    for group in ["FAP", "FAP_ICS"]:
        rs = [r for r in rows if r["group"] == group]
        summary.append(dict(
            group=group,
            repetitions=len(rs),
            window_seconds_mean=statistics.mean(r["window_seconds"] for r in rs),
            window_seconds_sample_std=statistics.stdev(r["window_seconds"] for r in rs),
            step_per_second_mean=statistics.mean(r["step_per_second"] for r in rs),
            step_per_second_sample_std=statistics.stdev(r["step_per_second"] for r in rs),
            pair_per_second_mean=statistics.mean(r["key_reference_pair_per_second"] for r in rs),
            pair_per_second_sample_std=statistics.stdev(r["key_reference_pair_per_second"] for r in rs),
            peak_allocated_MiB_max=max(r["peak_allocated_MiB"] for r in rs),
            peak_reserved_MiB_max=max(r["peak_reserved_MiB"] for r in rs),
            covariance_calls_total=sum(r["ics_covariance_calls"] for r in rs),
            eigh_calls_total=sum(r["ics_eigh_calls"] for r in rs),
        ))
    write_csv(OUT / "summary.csv", summary)
    write_json(OUT / "manifest.json", dict(
        exit_code=0,
        host=os.uname().nodename,
        pid=os.getpid(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        input_manifest=str(manifest),
        input_frame_count=len(records),
        schedule=schedule,
        same_process=True,
        batch_pairs=4,
        warmup_steps=20,
        measured_steps=100,
        repetitions_per_group=3,
        fp32=True,
        cudnn_benchmark=False,
        no_primary_results_overwritten=True,
    ))
    print("RETEST COMPLETE", flush=True)


if __name__ == "__main__":
    main()
