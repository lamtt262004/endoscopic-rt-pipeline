"""diag_02_cudagraph.py — CUDA Graph vs eager, do paired trong cung mot lan chay."""
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import (load_pmfnet, query_gpu_state, save_result,
                         setup_measurement_env, summarize)

SHAPE = (1, 3, 256, 256)


class ClockWatcher:
    def __init__(self):
        self._stop = threading.Event()
        self.clocks, self.throttles = [], []

    def __enter__(self):
        def run():
            while not self._stop.is_set():
                s = query_gpu_state()
                try:
                    self.clocks.append(float(s.get("clocks_sm_mhz", 0)))
                except (TypeError, ValueError):
                    pass
                self.throttles.append(s.get("throttle", "?"))
                self._stop.wait(1.0)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)

    def report(self):
        if not self.clocks:
            return "clock: n/a"
        idle_pct = 100 * sum("GpuIdle" in t for t in self.throttles) / len(self.throttles)
        return (f"clock sm: mean={statistics.mean(self.clocks):.0f} "
                f"min={min(self.clocks):.0f} max={max(self.clocks):.0f} MHz | "
                f"GpuIdle {idle_pct:.0f}% thoi gian")


def measure(run_once, warmup=50, iters=500, label=""):
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    lat = []
    with ClockWatcher() as w:
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_once()
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
    st = summarize(lat)
    print(f"  [{label}] {st}")
    print(f"           {w.report()}")
    return st, w


def measure_cpu_cost(run_once, n=1, warmup=30):
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        run_once()
    cpu_ms = (time.perf_counter() - t0) * 1000 / n
    torch.cuda.synchronize()
    return cpu_ms


def main():
    setup_measurement_env()
    model = load_pmfnet(device="cuda")

    print("\n" + "=" * 72)
    print("A. eager (moc so sanh)")
    print("=" * 72)
    x = torch.randn(*SHAPE, device="cuda")
    with torch.inference_mode():
        eager_cpu = measure_cpu_cost(lambda: model(x))
        st_eager, w_eager = measure(lambda: model(x), label="eager  ")
    print(f"           chi phi CPU/forward = {eager_cpu:.2f} ms")

    print("\n" + "=" * 72)
    print("B. CUDA graph")
    print("=" * 72)
    print("""  Ba rang buoc cua CUDA Graph, va vi sao PMFNet thoa het:

    1. Shape tinh. Graph ghi lai dia chi bo nho + cau hinh launch cu the.
       -> PMFNet dung co dinh 1x3x256x256. OK.

    2. Dia chi buffer co dinh. Replay ghi vao dung vung nho luc capture.
       -> Phai co static_input / static_output, moi frame copy du lieu vao
          static_input roi replay. Khong the truyen tensor moi.

    3. khong duoc co dong bo hoa hay control flow phu thuoc du lieu trong
       forward (.item(), nonzero(), if x.max()>0...).
       -> Da kiem tra: backbone.py:222 dung `_, _, H, W = x.shape` -> Python
          int, khong phai tensor. `.item()` duy nhat nam trong __init__.
          Forward sach. OK.

    [luu Y] Dung torch.no_grad() chu khong phai inference_mode() khi capture:
    tensor tao trong inference_mode co the gay van de luc capture graph.""")

    static_in = torch.randn(*SHAPE, device="cuda")

    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                static_out = model(static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(g):
            static_out = model(static_in)
        torch.cuda.synchronize()
        print(f"\n  capture xong trong {(time.perf_counter()-t0)*1000:.0f} ms "
              f"(mot lan duy nhat)")

        graph_cpu = measure_cpu_cost(g.replay)
        st_graph, w_graph = measure(g.replay, label="graph  ")
        print(f"           chi phi CPU/forward = {graph_cpu:.2f} ms")

        with torch.inference_mode():
            ref = model(static_in)
        g.replay()
        torch.cuda.synchronize()
        max_diff = (static_out - ref).abs().max().item()
        print(f"           sai khac so voi eager: {max_diff:.2e} "
              f"({'OK' if max_diff < 1e-4 else 'co van de!'})")

    print("\n" + "=" * 72)
    print("ket luan")
    print("=" * 72)
    speedup = st_eager.mean_ms / st_graph.mean_ms
    print(f"  {'':22} {'eager':>10} {'graph':>10} {'thay doi':>12}")
    print(f"  {'-'*58}")
    print(f"  {'chi phi CPU (ms)':22} {eager_cpu:10.2f} {graph_cpu:10.2f} "
          f"{graph_cpu/eager_cpu-1:11.0%}")
    print(f"  {'mean (ms)':22} {st_eager.mean_ms:10.2f} {st_graph.mean_ms:10.2f} "
          f"{st_graph.mean_ms/st_eager.mean_ms-1:11.0%}")
    print(f"  {'p50 (ms)':22} {st_eager.p50_ms:10.2f} {st_graph.p50_ms:10.2f} "
          f"{st_graph.p50_ms/st_eager.p50_ms-1:11.0%}")
    print(f"  {'p99 (ms)':22} {st_eager.p99_ms:10.2f} {st_graph.p99_ms:10.2f} "
          f"{st_graph.p99_ms/st_eager.p99_ms-1:11.0%}")
    print(f"  {'jitter (p99/p50)':22} {st_eager.jitter:10.2f} {st_graph.jitter:10.2f}")
    print(f"  {'FPS infer-only':22} {1000/st_eager.mean_ms:10.1f} "
          f"{1000/st_graph.mean_ms:10.1f} {speedup:10.2f}x")
    print()
    print(f"  clock eager: {w_eager.report()}")
    print(f"  clock graph: {w_graph.report()}")
    print()

    if speedup > 1.5:
        print(f"  => CUDA Graph an {speedup:.2f}x. Lon hon nhieu so voi du doan 'chi thu")
        print("Ly do: bo CPU bottleneck -> GPU giu boost clock -> kernel time tu no giam.")
        print("     Hai hieu ung nhan nhau, khong phai cong.")
    elif speedup > 1.15:
        print(f"  => CUDA Graph an {speedup:.2f}x. Dung nhu du doan ban dau (bo gap).")
        print("     Vong lap phan hoi clock khong bi pha vo -> xem lai bang clock.")
    else:
        print(f"  => CUDA Graph gan nhu khong giup ({speedup:.2f}x).")
        print("     => That su GPU-bound. Di thang TensorRT.")

    print("\n  [pipeline video] Graph doc tu static_in: moi frame copy_ roi replay, ton ~0.1-0.3 ms.")

    save_result({
        "config": "pytorch_cuda_graph",
        "stage": "infer_only",
        "shape": list(SHAPE),
        "eager": vars(st_eager), "graph": vars(st_graph),
        "cpu_ms_eager": round(eager_cpu, 3), "cpu_ms_graph": round(graph_cpu, 3),
        "speedup": round(speedup, 3),
        "clock_eager": w_eager.report(), "clock_graph": w_graph.report(),
    })


if __name__ == "__main__":
    main()
