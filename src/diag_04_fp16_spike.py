"""
diag_04_fp16_spike.py — Ngay 4b. Truy cai spike cua TRT fp16.

    python src/diag_04_fp16_spike.py
    python src/diag_04_fp16_spike.py --iters 400 --engines fp16,tf32,fp32
"""
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import save_result, setup_measurement_env
from trt_utils import TRTRunner

ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT / "engines"
DELAYS_MS = [0, 1, 2, 4, 6, 8, 12]


class ClockLog:
    def __init__(self, period=0.05):
        self.rows, self.period = [], period
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        def run():
            while not self._stop.is_set():
                try:
                    o = subprocess.run(
                        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5).stdout.strip()
                    sm, pw, tp = [float(v) for v in o.split(",")]
                    self.rows.append((time.perf_counter(), sm, pw, tp))
                except Exception:
                    pass
                time.sleep(self.period)
        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._th.join(timeout=2)

    def near(self, t):
        if not self.rows:
            return None
        i = min(range(len(self.rows)), key=lambda k: abs(self.rows[k][0] - t))
        return self.rows[i]


def measure(runner, x, iters, delay_ms, clog):
    ts, lat = [], []
    for _ in range(iters):
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runner(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ts.append(t0)
        lat.append((t1 - t0) * 1000)
    return np.array(ts), np.array(lat)


def stats(a):
    return dict(mean=a.mean(), p50=np.percentile(a, 50), p90=np.percentile(a, 90),
                p99=np.percentile(a, 99), mx=a.max(),
                jitter=np.percentile(a, 99) / np.percentile(a, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--engines", default="fp16,tf32")
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()

    setup_measurement_env(tf32_matmul=True)
    tags = args.engines.split(",")
    runners = {t: TRTRunner(ENGINE_DIR / f"pmfnet_trt_{t}.engine") for t in tags}
    x = torch.randn(1, 3, 256, 256, device="cuda")

    print("=" * 84)
    print("ngay 4b — tiem khoang nghi CPU, xem spike co moc ra khong")
    print("=" * 84)
    print("Gia thuyet: fp16 xong som -> GPU ranh lau hon -> ha xung -> frame sau cham.")
    print(f"\nMoi o: {args.iters} iter. Hai engine do xen ke trong cung mot delay.")

    with torch.inference_mode():
        for r in runners.values():
            for _ in range(args.warmup):
                r(x)
        torch.cuda.synchronize()

        results = {t: {} for t in tags}
        raw = {t: {} for t in tags}
        with ClockLog() as clog:
            print(f"\n{'delay':>7}  " + "  ".join(
                f"{t.upper():>34}" for t in tags))
            print(f"{'(ms)':>7}  " + "  ".join(
                f"{'p50':>8}{'p99':>9}{'jitter':>9}{'>1.5xp50':>8}" for _ in tags))
            print("-" * 84)
            for d in DELAYS_MS:
                line = []
                for t in tags:
                    ts, lat = measure(runners[t], x, args.iters, d, clog)
                    s = stats(lat)
                    slow = (lat > 1.5 * s["p50"]).mean() * 100
                    s["slow_pct"] = slow
                    results[t][d] = s
                    raw[t][d] = (ts, lat)
                    line.append(f"{s['p50']:8.2f}{s['p99']:9.2f}"
                                f"{s['jitter']:8.2f}x{slow:7.1f}%")
                print(f"{d:7d}  " + "  ".join(line))

        print("\n" + "=" * 84)
        print("1. spike co moc ra theo DELAY khong?")
        print("=" * 84)
        verdict = {}
        for t in tags:
            j0 = results[t][0]["jitter"]
            jmax = max(results[t][d]["jitter"] for d in DELAYS_MS)
            dmax = max(DELAYS_MS, key=lambda d: results[t][d]["jitter"])
            grew = jmax / j0
            verdict[t] = grew
            print(f"  {t.upper():6} jitter: delay=0 -> {j0:.2f}x | "
                  f"toi da {jmax:.2f}x tai delay={dmax}ms | tang {grew:.2f} lan")
        if len(tags) >= 2:
            a, b = tags[0], tags[1]
            print(f"\n  => {a.upper()} jitter tang {verdict[a]:.2f}x, "
                  f"{b.upper()} tang {verdict[b]:.2f}x theo delay")
            if verdict[a] > 1.4 and verdict[a] > verdict[b] * 1.3:
                print(f"     gia thuyet dung: khoang nghi lam spike moc ra, "
                      f"va chi moc o {a.upper()}")
            elif verdict[a] < 1.3 and verdict[b] < 1.3:
                print(f"     gia thuyet sai: tiem khoang nghi khong tao ra spike. "
                      f"Nguyen nhan nam o cho khac.")
            else:
                print(f"     khong ket luan duoc — ca hai deu doi, hoac doi it")

        print("\n" + "=" * 84)
        print("2. FRAME cham co trung voi luc clock tut khong?")
        print("=" * 84)
        if not clog.rows:
            print("  (khong lay duoc mau clock)")
        else:
            sm_all = np.array([r[1] for r in clog.rows])
            print(f"  {len(clog.rows)} mau clock: {sm_all.min():.0f}-{sm_all.max():.0f} MHz "
                  f"(trung binh {sm_all.mean():.0f})")
            print(f"\n  {'engine':7} {'delay':>6} {'clock luc nhanh':>17} "
                  f"{'clock luc cham':>16} {'chenh':>8}")
            print("  " + "-" * 60)
            for t in tags:
                for d in DELAYS_MS:
                    ts, lat = raw[t][d]
                    thr = 1.5 * np.percentile(lat, 50)
                    slow_i = np.where(lat > thr)[0]
                    if len(slow_i) < 3:
                        continue
                    fast_i = np.where(lat <= np.percentile(lat, 50))[0]
                    cs = [clog.near(ts[i])[1] for i in slow_i]
                    cf = [clog.near(ts[i])[1] for i in fast_i[:len(slow_i) * 5]]
                    print(f"  {t.upper():7} {d:6d} {np.mean(cf):14.0f} MHz "
                          f"{np.mean(cs):13.0f} MHz {np.mean(cs)-np.mean(cf):+7.0f}")

        print("\n" + "=" * 84)
        print("3. gia / loi cua viec tiem khoang nghi")
        print("=" * 84)
        print(f"  {'engine':7} {'delay':>6} {'p50':>8} {'so voi delay=0':>16}")
        print("  " + "-" * 44)
        for t in tags:
            base = results[t][0]["p50"]
            for d in DELAYS_MS:
                p = results[t][d]["p50"]
                print(f"  {t.upper():7} {d:6d} {p:8.2f} {p/base:15.3f}x")

    save_result({
        "config": "fp16_spike_delay_sweep", "stage": "diag",
        "iters": args.iters, "delays_ms": DELAYS_MS,
        "results": {t: {str(d): {k: float(v) for k, v in s.items()}
                        for d, s in results[t].items()} for t in tags},
        "jitter_growth": {t: float(v) for t, v in verdict.items()},
    })


if __name__ == "__main__":
    main()
