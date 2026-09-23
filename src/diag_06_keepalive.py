"""Da loai: stream rieng, cach cap phat, transient luc danh thuc, ha xung ben vung."""
import sys, threading, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bench_utils import setup_measurement_env
from trt_utils import TRTRunner

DELAY_S, ITERS, WARM = 0.002, 400, 100
setup_measurement_env(tf32_matmul=True)
x = torch.randn(1, 3, 256, 256, device="cuda")


class KeepAlive:
    def __init__(self, size):
        self.size = size
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        if self.size == 0:
            return self
        def run():
            lo, _ = torch.cuda.Stream.priority_range()
            s = torch.cuda.Stream(priority=lo)
            a = torch.randn(self.size, self.size, device="cuda", dtype=torch.float16)
            with torch.cuda.stream(s):
                while not self._stop.is_set():
                    for _ in range(50):
                        a.mul_(1.0001)
                    s.synchronize()
        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()
        time.sleep(0.3)
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._th:
            self._th.join(timeout=3)
        torch.cuda.synchronize()


def run(r):
    for _ in range(WARM):
        r(x)
    torch.cuda.synchronize()
    lat = []
    for _ in range(ITERS):
        time.sleep(DELAY_S)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r(x)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    a = np.array(lat)
    p50 = np.percentile(a, 50)
    return dict(p50=p50, p99=np.percentile(a, 99), mean=a.mean(),
                jit=np.percentile(a, 99) / p50, slow=(a > 1.5 * p50).mean() * 100)


print(f"delay {DELAY_S*1000:.0f}ms, {ITERS} iter/o\n")
print(f"{'keep-alive':22} {'p50':>8} {'mean':>8} {'p99':>8} {'jitter':>8} {'%cham':>7}")
print("-" * 68)
res = {}
with torch.inference_mode():
    r = TRTRunner(ROOT / "engines" / "pmfnet_trt_fp16.engine")
    for size in (0, 128, 512, 1024, 2048):
        with KeepAlive(size):
            s = run(r)
        res[size] = s
        name = "tat (hien tai)" if size == 0 else f"matmul {size}x{size}"
        print(f"{name:22} {s['p50']:8.2f} {s['mean']:8.2f} {s['p99']:8.2f} "
              f"{s['jit']:7.2f}x {s['slow']:6.1f}%")

print("\n" + "=" * 68)
print("danh doi")
print("=" * 68)
b = res[0]
print(f"  moc (tat): p50 {b['p50']:.2f}  p99 {b['p99']:.2f}  {b['slow']:.1f}% cham")
for size, s in res.items():
    if size == 0:
        continue
    print(f"  {size:5d}: p50 {s['p50']:+6.2f}%  p99 {s['p99']/b['p99']*100-100:+6.1f}%  "
          f"cham {s['slow']:5.1f}% (tu {b['slow']:.1f}%)  "
          f"-> {'dang' if s['slow'] < b['slow']*0.5 and s['p50'] < b['p50']*1.15 else 'khong dang'}"
          .replace(f"p50 {s['p50']:+6.2f}%", f"p50 {s['p50']/b['p50']*100-100:+6.1f}%"))
