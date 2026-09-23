"""diag_03_tf32.py — matmul tf32 co dang bat khong? An duoc bao nhieu?"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import load_pmfnet, query_gpu_state, save_result, summarize

SHAPE = (1, 3, 256, 256)
ITERS = 500


def kernel_breakdown(model_call, n=20):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(20):
        model_call()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            model_call()
        torch.cuda.synchronize()

    def sd(e):
        return getattr(e, "self_device_time_total", None) or \
               getattr(e, "self_cuda_time_total", 0) or 0

    kernels = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
    total = sum(sd(e) for e in kernels) / 1000 / n

    groups = {"GEMM: sgemm (CUDA core, fp32)": 0.0,
              "GEMM: tensorop (Tensor Core)": 0.0,
              "conv: tf32 (Tensor Core)": 0.0,
              "khac (elementwise/norm/layout...)": 0.0}
    for e in kernels:
        ms, k = sd(e) / 1000 / n, e.key.lower()
        if "sgemm" in k:
            groups["GEMM: sgemm (CUDA core, fp32)"] += ms
        elif "cutlass" in k or "tensorop" in k or "s1688" in k or "h1688" in k:
            groups["GEMM: tensorop (Tensor Core)"] += ms
        elif "tf32" in k:
            groups["conv: tf32 (Tensor Core)"] += ms
        else:
            groups["khac (elementwise/norm/layout...)"] += ms
    top = sorted(kernels, key=sd, reverse=True)[:6]
    return total, groups, [(e.key[:52], sd(e) / 1000 / n, e.count / n) for e in top]


def measure_cpu_cost(run_once, warmup=50):
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_once()
    cpu_ms = (time.perf_counter() - t0) * 1000
    torch.cuda.synchronize()
    return cpu_ms


def measure(run_once, warmup=50, iters=ITERS):
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    lat = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return summarize(lat)


def capture_graph(model, static_in):
    with torch.no_grad():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                out = model(static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(static_in)
        torch.cuda.synchronize()
    return g, out


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_pmfnet(device="cuda")
    x = torch.randn(*SHAPE, device="cuda")

    results = {}
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        tag = "matmul tf32 on " if tf32 else "matmul tf32 off"

        print("\n" + "=" * 72)
        print(f"{tag}   (cudnn.allow_tf32 = True o ca hai truong hop)")
        print("=" * 72)

        with torch.inference_mode():
            cpu_ms = measure_cpu_cost(lambda: model(x))
            st_eager = measure(lambda: model(x))
            print(f"  eager : {st_eager}")
            print(f"          chi phi CPU/forward = {cpu_ms:.2f} ms")
            gpu_ms, groups, top = kernel_breakdown(lambda: model(x))

        static_in = torch.randn(*SHAPE, device="cuda")
        g, _ = capture_graph(model, static_in)
        with torch.no_grad():
            st_graph = measure(g.replay)
        print(f"  graph : {st_graph}")

        print(f"\n  Kernel GPU time: {gpu_ms:.2f} ms/forward")
        for name, ms in groups.items():
            if ms > 0.001:
                print(f"    {name:26s} {ms:6.2f} ms  ({ms/gpu_ms*100:4.1f}%)")
        print("  Top kernel:")
        for k, ms, c in top:
            print(f"    {k:<52} {ms:6.3f} ms x{c:.0f}")

        results[tf32] = dict(eager=st_eager, graph=st_graph, cpu_ms=cpu_ms,
                             gpu_ms=gpu_ms, groups=groups)
        del g, static_in
        torch.cuda.empty_cache()

    off, on = results[False], results[True]
    print("\n" + "=" * 72)
    print("so sanh  (paired — cung mot lan chay, cung nhiet do, cung power state)")
    print("=" * 72)
    print(f"  {'':16} {'tf32 off':>10} {'tf32 on':>10} {'thay doi':>10}")
    print("  " + "-" * 50)
    for k in ("eager", "graph"):
        a, b = off[k].mean_ms, on[k].mean_ms
        print(f"  {k+' mean (ms)':16} {a:10.2f} {b:10.2f} {b/a-1:9.1%}")
    for k in ("eager", "graph"):
        a, b = off[k].p99_ms, on[k].p99_ms
        print(f"  {k+' p99 (ms)':16} {a:10.2f} {b:10.2f} {b/a-1:9.1%}")
    print(f"  {'chi phi CPU (ms)':16} {off['cpu_ms']:10.2f} {on['cpu_ms']:10.2f} "
          f"{on['cpu_ms']/off['cpu_ms']-1:9.1%}")
    print(f"  {'GPU time (ms)':16} {off['gpu_ms']:10.2f} {on['gpu_ms']:10.2f} "
          f"{on['gpu_ms']/off['gpu_ms']-1:9.1%}")
    print("\n  Phan ra GEMM:")
    for gname in off["groups"]:
        a, b = off["groups"][gname], on["groups"][gname]
        if a > 0.01 or b > 0.01:
            print(f"    {gname:36} {a:7.2f} -> {b:7.2f} ms")

    gemm_off = (off["groups"]["GEMM: sgemm (CUDA core, fp32)"]
                + off["groups"]["GEMM: tensorop (Tensor Core)"])
    gemm_on = (on["groups"]["GEMM: sgemm (CUDA core, fp32)"]
               + on["groups"]["GEMM: tensorop (Tensor Core)"])
    print(f"    {'=> tong GEMM':36} {gemm_off:7.2f} -> {gemm_on:7.2f} ms "
          f"({gemm_on/gemm_off-1:+.0%})")

    print("\n  doc ket qua:")
    print(f"    GPU time {off['gpu_ms']:.2f} -> {on['gpu_ms']:.2f} ms "
          f"({on['gpu_ms']/off['gpu_ms']-1:+.1%})")
    print(f"    eager    {off['eager'].mean_ms:.2f} -> {on['eager'].mean_ms:.2f} ms "
          f"({on['eager'].mean_ms/off['eager'].mean_ms-1:+.1%})")
    print(f"    graph    {off['graph'].mean_ms:.2f} -> {on['graph'].mean_ms:.2f} ms "
          f"({on['graph'].mean_ms/off['graph'].mean_ms-1:+.1%})")
    print()
    if on["eager"].mean_ms > off["eager"].mean_ms and \
       on["graph"].mean_ms < off["graph"].mean_ms:
        print("=> eager cham di trong khi graph nhanh len: eager dang bi chan boi CPU.")
        print("       thi khong giup gi, tham chi phan tac dung'.")
    print(f"\n    ket luan: dung tf32 matmul chung voi CUDA Graph "
          f"-> {off['graph'].mean_ms:.2f} -> {on['graph'].mean_ms:.2f} ms.")
    print("    Va nho ghi vao bang benchmark: day khong con la 'fp32' nua.")

    save_result({
        "config": "tf32_matmul_ablation", "stage": "infer_only",
        "shape": list(SHAPE),
        "tf32_off": {"eager": vars(off["eager"]), "graph": vars(off["graph"]),
                     "gpu_ms": off["gpu_ms"], "groups": off["groups"]},
        "tf32_on": {"eager": vars(on["eager"]), "graph": vars(on["graph"]),
                    "gpu_ms": on["gpu_ms"], "groups": on["groups"]},
        "env_after": query_gpu_state(),
    })


if __name__ == "__main__":
    main()
