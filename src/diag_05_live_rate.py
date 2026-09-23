"""diem van hanh that chua he duoc do."""
import subprocess, sys, threading, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bench_utils import setup_measurement_env
from trt_utils import TRTRunner

ITERS, WARM = 300, 100
CASES = [(0, "doc file toi da"), (8, "nguon 60 fps"), (18, "nguon 40 fps"),
         (25, "nguon 30 fps"), (33, "nguon 25 fps (hyper-kvasir)")]

setup_measurement_env(tf32_matmul=True)
x = torch.randn(1, 3, 256, 256, device="cuda")

clk, stop = [], threading.Event()
def watch():
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            clk.append((time.perf_counter(), *[float(v) for v in o.split(",")]))
        except Exception:
            pass
        time.sleep(0.05)

th = threading.Thread(target=watch, daemon=True); th.start()
res = {}
with torch.inference_mode():
    for tag in ("fp16", "tf32"):
        r = TRTRunner(ROOT / "engines" / f"pmfnet_trt_{tag}.engine")
        for _ in range(WARM):
            r(x)
        torch.cuda.synchronize()
        for d, name in CASES:
            t_lo = time.perf_counter()
            lat = []
            for _ in range(ITERS):
                if d:
                    time.sleep(d / 1000)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                r(x)
                torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1000)
            t_hi = time.perf_counter()
            a = np.array(lat)
            c = [v[1] for v in clk if t_lo <= v[0] <= t_hi]
            res[(tag, d)] = dict(
                p50=np.percentile(a, 50), p99=np.percentile(a, 99), mean=a.mean(),
                clk=np.mean(c) if c else float("nan"), name=name)
        del r
        torch.cuda.empty_cache()
stop.set(); th.join(timeout=2)

print(f"\n{'='*84}")
print("LATENCY infer theo toc do nguon (khe nghi giua cac frame)")
print(f"{'='*84}")
print(f"{'khe nghi':>9} {'tinh huong':28} " + "  ".join(
    f"{t.upper():>22}" for t in ("fp16", "tf32")))
print(f"{'':>9} {'':28} " + "  ".join(f"{'p50':>7}{'p99':>8}{'clock':>7}"
                                       for _ in range(2)))
print("-" * 84)
for d, name in CASES:
    row = []
    for tag in ("fp16", "tf32"):
        s = res[(tag, d)]
        row.append(f"{s['p50']:7.2f}{s['p99']:8.2f}{s['clk']:7.0f}")
    print(f"{d:7d}ms {name:28} " + "  ".join(row))

print(f"\n{'='*84}\nSO voi dieu kien 'doc file toi da'\n{'='*84}")
for tag in ("fp16", "tf32"):
    b = res[(tag, 0)]
    print(f"  {tag.upper()}:")
    for d, name in CASES:
        s = res[(tag, d)]
        print(f"    {d:3d}ms  p50 {s['p50']:6.2f} ({s['p50']/b['p50']:5.2f}x)  "
              f"p99 {s['p99']:6.2f} ({s['p99']/b['p99']:5.2f}x)  "
              f"clock {s['clk']:.0f} MHz")

print(f"\n{'='*84}\nY nghia: e2e = infer + ~7.7ms cac khau khac\n{'='*84}")
print(f"  {'khe nghi':>9} {'e2e p50 (fp16)':>16} {'e2e p99':>10} {'con kip 16.7ms?':>18}")
for d, name in CASES:
    s = res[("fp16", d)]
    e50, e99 = s["p50"] + 7.7, s["p99"] + 7.7
    print(f"  {d:7d}ms {e50:15.2f} {e99:10.2f} {'co' if e50 <= 16.7 else 'khong':>18}")
