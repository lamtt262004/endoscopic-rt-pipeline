"""diag_01_async.py — Lab Ngay 1. Khong can video, khong can anh."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import (bench, describe_env, load_pmfnet, query_gpu_state,
                         save_result, setup_measurement_env, summarize)

DEVICE = "cuda"
SHAPE = (1, 3, 256, 256)
BUDGET_MS = 16.7


@torch.inference_mode()
def exp_a_why_sync(model, x, n=100):
    print("\n" + "=" * 72)
    print("thi nghiem A — vi sao phai synchronize()")
    print("=" * 72)

    for _ in range(20):
        model(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        y = model(x)
    t_no_sync = (time.perf_counter() - t0) * 1000 / n
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n):
        y = model(x)
    torch.cuda.synchronize()
    t_tail_sync = (time.perf_counter() - t0) * 1000 / n

    per_iter = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x)
        torch.cuda.synchronize()
        per_iter.append((time.perf_counter() - t0) * 1000)
    t_full_sync = sum(per_iter) / n

    print(f"  1. Khong sync         : {t_no_sync:7.2f} ms/iter   <-- sai")
    print(f"  2. Sync cuoi vong lap : {t_tail_sync:7.2f} ms/iter")
    print(f"  3. Sync tung vong     : {t_full_sync:7.2f} ms/iter   <-- dung")
    print()
    ratio = t_full_sync / t_no_sync
    print(f"  Cach 1 nho hon cach 3 {ratio:.2f} lan.")
    print("Cach 1 chi do thoi gian CPU day lenh vao hang doi, chua phai thoi gian tinh.")
    print()
    if ratio > 3:
        print("=> Cach 1 nho hon han: GPU la nut co chai.")
    else:
        print("  => bat thuong, va day chinh la phat hien:")
        print("Khong thay trieu chung 'thieu sync -> so nho vo ly':")
        print("      CPU day lenh mat gan bang GPU tinh.")
    print()
    print(f"  Cach 2 vs cach 3: {t_tail_sync:.2f} vs {t_full_sync:.2f} ms")
    print("Cach 3 lon hon vi moi vong bi chan -> mat overlap CPU/GPU.")
    print("        do rieng >= latency do lien mach'.")
    return t_no_sync, t_tail_sync, t_full_sync


@torch.inference_mode()
def exp_b_launch_bound(model, x, n_list=(1, 2, 5, 10, 50, 200)):
    print("\n" + "=" * 72)
    print("thi nghiem B — nghen o CPU hay GPU?")
    print("=" * 72)

    for _ in range(50):
        model(x)
    torch.cuda.synchronize()

    print(f"  {'n':>5} | {'t_enqueue':>10} | {'t_total':>10} | {'r':>6} | {'ms/iter':>8}")
    print("  " + "-" * 56)

    ratios, cpu_ms_n1 = [], None
    for n in n_list:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            y = model(x)
        t_enq = (time.perf_counter() - t0) * 1000
        torch.cuda.synchronize()
        t_tot = (time.perf_counter() - t0) * 1000

        r = t_enq / t_tot
        ratios.append(r)
        if n == 1:
            cpu_ms_n1 = t_enq
        print(f"  {n:>5} | {t_enq:9.1f}ms | {t_tot:9.1f}ms | {r:6.2f} | {t_tot/n:7.2f}")

    print()
    r1 = ratios[0]
    print(f"  Chi doc r(n=1) = {r1:.2f} — cac n lon hon deu qua tran hang doi (~1024).")
    if r1 > 0.8:
        print("goi Y: launch-bound. Nhung n=1 (~700 launch) da gan tran -> cho Thi nghiem E.")
    elif r1 < 0.3:
        print("  goi Y: GPU-bound. Cho Thi nghiem E xac nhan.")
    else:
        print("  goi Y: o giua. Cho Thi nghiem E.")
    return ratios, cpu_ms_n1


@torch.inference_mode()
def exp_e_gpu_busy(model, x, wall_ms_clean, cpu_ms, n=20):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    print("\n" + "=" * 72)
    print("thi nghiem E — GPU ban bao nhieu % thoi gian? (bang chung dut diem)")
    print("=" * 72)

    for _ in range(20):
        model(x)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
        wall_profiled = (time.perf_counter() - t0) * 1000 / n

    evs = prof.key_averages()

    def self_dev(e):
        return getattr(e, "self_device_time_total", None) or \
               getattr(e, "self_cuda_time_total", 0) or 0

    kernels = [e for e in evs if e.device_type == DeviceType.CUDA]
    aten_ops = [e for e in evs if e.device_type == DeviceType.CPU]

    gpu_ms = sum(self_dev(e) for e in kernels) / 1000 / n
    n_kernels = sum(e.count for e in kernels) / n
    n_aten = sum(e.count for e in aten_ops) / n
    busy = gpu_ms / wall_ms_clean

    print(f"  wall-clock sach     : {wall_ms_clean:7.2f} ms/iter  (Thi nghiem A, cach 3)")
    print(f"  wall-clock co prof  : {wall_profiled:7.2f} ms/iter  (phinh ~{wall_profiled/wall_ms_clean:.0%} — dung de chia!)")
    print(f"  GPU thuc su ban     : {gpu_ms:7.2f} ms/iter")
    print(f"  GPU ranh (gap)      : {wall_ms_clean - gpu_ms:7.2f} ms/iter")
    print(f"  => ty le GPU ban    : {busy*100:5.1f} %")
    print()
    print(f"  so kernel GPU       : {n_kernels:.0f} / forward")
    print(f"  so ATen op          : {n_aten:.0f} / forward (ke ca op long nhau)")
    print()
    print("  --- mo hinh: wall ~ max(CPU, GPU), khong phai tong ---")
    print(f"    chi phi CPU (enqueue, Thi nghiem B n=1) : {cpu_ms:6.2f} ms")
    print(f"    chi phi GPU (kernel time)               : {gpu_ms:6.2f} ms")
    print(f"    wall-clock thuc te                      : {wall_ms_clean:6.2f} ms")
    print(f"    max(CPU, GPU)                           : {max(cpu_ms, gpu_ms):6.2f} ms  <-- khop")
    print("CPU va GPU chong lan nhau -> wall = max(CPU, GPU), khong phai tong.")

    print("\n  Top 10 kernel an thoi gian GPU nhieu nhat:")
    print(f"    {'kernel':<50} {'ms/iter':>8} {'%GPU':>6} {'x':>5}")
    for e in sorted(kernels, key=self_dev, reverse=True)[:10]:
        ms = self_dev(e) / 1000 / n
        print(f"    {e.key[:50]:<50} {ms:8.3f} {ms/gpu_ms*100:5.1f}% {e.count/n:5.0f}")

    print()
    if busy < 0.5:
        print(f"  ket luan: launch-bound. GPU ranh {(1-busy)*100:.0f}% thoi gian.")
        print("  -> Ngay 2: CUDA Graphs / torch.compile('reduce-overhead') truoc TensorRT.")
    else:
        print(f"  ket luan: GPU ban {busy*100:.0f}% — chu yeu GPU-bound.")
        print("  Gia thuyet 'launch-bound' ban dau bi bac bo bang so do.")
        print()
        print(f"  Nhung CPU ({cpu_ms:.1f} ms) van hoi lon hon GPU ({gpu_ms:.1f} ms)")
        print("  -> CPU la nut co chai bien, du GPU gan nhu luc nao cung ban.")
        print(f"  -> CUDA Graphs chi thu hoi toi da {wall_ms_clean-gpu_ms:.1f} ms "
              f"=> {gpu_ms:.1f} ms = {1000/gpu_ms:.0f} FPS. Dang lam vi re, nhung khong du.")
        print(f"  -> Phan chinh nam o {gpu_ms:.1f} ms kernel time. Do la viec cua TensorRT.")
    print()
    print("Doc bang kernel de biet TensorRT an diem o dau:")
    print("    elementwise / layer_norm nhieu -> memory-bound -> fusion")
    print("    nchwToNhwcKernel               -> cuDNN doi layout thua")
    print("    sgemm / conv                   -> compute-bound -> fp16/int8")
    return busy, gpu_ms, n_kernels, n_aten


def exp_c_baseline(model, x):
    print("\n" + "=" * 72)
    print("thi nghiem C — baseline dung luat (50 warm-up + 500 iter)")
    print("=" * 72)

    stats, samples = bench(lambda: model(x), warmup=50, iters=500,
                           budget_ms=BUDGET_MS)
    print(f"  {stats}")
    print(f"  FPS (infer-only) = {1000/stats.mean_ms:.1f}")
    print()
    print("Day la FPS infer-only, khong phai latency ca pipeline.")

    if stats.jitter > 1.5:
        print(f"\n  [!] jitter = {stats.jitter:.2f}x (>1.5) -> co van de that su,")
        print("      khong phai nhieu ngau nhien. Xem Thi nghiem D.")

    import numpy as np
    a = np.asarray(samples)
    slow_idx = np.where(a > np.percentile(a, 99))[0]
    if len(slow_idx):
        first_half = (slow_idx < len(a) / 2).sum()
        print(f"\n  Vi tri {len(slow_idx)} frame cham nhat (top 1%):")
        print(f"    nua dau vong lap: {first_half} | nua sau: {len(slow_idx)-first_half}")
        if len(slow_idx) - first_half > 2 * max(first_half, 1):
            print("    -> Don ve cuoi => nghi throttling (GPU nong dan).")
        else:
            print("    -> Rai deu => nghi allocator / OS scheduling / Python gc.")
    return stats


def exp_d_thermal(model, x, seconds=30):
    import threading

    print("\n" + "=" * 72)
    print(f"thi nghiem D — chay lien tuc {seconds}s, probe o thread rieng")
    print("=" * 72)
    print(f"  {'t(s)':>5} | {'sm_mhz':>7} | {'temp':>5} | {'power':>6} | {'util':>5} | throttle")
    print("  " + "-" * 68)

    stop = threading.Event()
    t_start = time.perf_counter()

    def probe():
        while not stop.is_set():
            s = query_gpu_state()
            print(f"  {time.perf_counter()-t_start:5.1f} | "
                  f"{s.get('clocks_sm_mhz','?'):>7} | {s.get('temp_c','?'):>5} | "
                  f"{s.get('power_w','?'):>6} | {s.get('util_pct','?'):>5} | "
                  f"{s.get('throttle','?')}")
            stop.wait(5.0)

    th = threading.Thread(target=probe, daemon=True)
    th.start()
    with torch.inference_mode():
        while time.perf_counter() - t_start < seconds:
            model(x)
    torch.cuda.synchronize()
    stop.set()
    th.join(timeout=2)

    print("\n  Doc bang tren:")
    print("- clocks.sm tut dan + SwThermalSlowdown/SwPowerCap => throttling nhiet.")
    print("    - clocks.sm thap nhung on dinh, nhiet thap => power state thap.")
    print("    - utilization.gpu chi noi 'co kernel dang chay', khong noi hieu qua.")

    s = query_gpu_state()
    try:
        sm = float(s.get("clocks_sm_mhz", 0))
        temp = float(s.get("temp_c", 0))
    except (TypeError, ValueError):
        sm = temp = 0
    if 0 < sm < 1100 and temp < 80:
        print()
        print(f"  [!!] clocks.sm = {sm:.0f} MHz nhung nhiet chi {temp:.0f}C.")
        print("rtx 3050 Laptop boost duoc ~1400-1800 MHz => day la power management,")
        print("       khong phai gioi han vat ly. Kiem tra:")
        print("         1. Da cam sac chua?")
        print("         2. Windows Power > 'Best performance'")
        print("         3. nvidia Control Panel > Power management = max performance")
        print("         4. Laptop co mux/Optimus: dam bao dang dung dGPU")


def main():
    if not torch.cuda.is_available():
        sys.exit("Khong thay CUDA.")

    setup_measurement_env()
    env = describe_env()
    print("=" * 72)
    print("dieu kien do")
    print("=" * 72)
    for k, v in env.items():
        print(f"  {k:20s} = {v}")
    print("\n  tf32_cudnn=True: conv da chay tf32, baseline nay khong phai fp32 thuan.")

    model = load_pmfnet(device=DEVICE)
    x = torch.randn(*SHAPE, device=DEVICE)

    _, _, wall_clean = exp_a_why_sync(model, x)
    ratios, cpu_ms = exp_b_launch_bound(model, x)
    busy, gpu_ms, n_kern, n_aten = exp_e_gpu_busy(
        model, x, wall_ms_clean=wall_clean, cpu_ms=cpu_ms)
    stats = exp_c_baseline(model, x)
    exp_d_thermal(model, x, seconds=30)

    save_result({
        "config": "pytorch_fp32_eager",
        "stage": "infer_only",
        "shape": list(SHAPE),
        "enqueue_ratio_n1": round(ratios[0], 3),
        "gpu_busy_pct": round(busy * 100, 1),
        "gpu_ms_per_iter": round(gpu_ms, 3),
        "n_kernels": round(n_kern),
        "n_aten_ops": round(n_aten),
        **{k: v for k, v in vars(stats).items()},
        "env": env,
        "env_after": query_gpu_state(),
    })
    print("\n  [xong] Con so quyet dinh Ngay 2 nam o Thi nghiem E: ty le GPU ban.")


if __name__ == "__main__":
    main()
