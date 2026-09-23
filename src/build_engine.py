"""
build_engine.py — Ngay 4. ONNX -> TensorRT engine, do fusion / toc do / do chinh xac.

    python src/build_engine.py                 # build + do het (mac dinh)
    python src/build_engine.py --skip-build     # dung .engine da co, chi do lai
    python src/build_engine.py --iters 500 --rounds 4
    python src/build_engine.py --n-images 200   # so anh de do Dice
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import (load_pmfnet, query_gpu_state, save_result,
                         setup_measurement_env, summarize)
from trt_utils import TRTRunner, build_engine, engine_layers

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "onnx" / "pmfnet_legacy.onnx"
ENGINE_DIR = ROOT / "engines"
SIZE = 256
BUDGET_MS = 16.7

CONFIGS = {
    "trt_fp32": (False, False),
    "trt_tf32": (False, True),
    "trt_fp16": (True, True),
}


def count_onnx_nodes(path):
    import onnx
    m = onnx.load(str(path))
    return len(m.graph.node), Counter(n.op_type for n in m.graph.node)


def classify_layers(names):
    g = Counter()
    for s in names:
        low = s.lower()
        if low.startswith("__myl") or low.startswith("__mye") or "myelin" in low:
            g["Myelin (fusion subgraph)"] += 1
        elif "pwn" in low:
            g["PWN (pointwise fusion)"] += 1
        elif "conv" in low or "cask" in low:
            g["Convolution"] += 1
        elif "gemm" in low or "matmul" in low:
            g["GEMM / MatMul"] += 1
        elif "reformat" in low or "copy" in low or "transpose" in low or "shuffle" in low:
            g["Reformat / Shuffle"] += 1
        elif "pool" in low:
            g["Pooling"] += 1
        elif "resize" in low or "upsample" in low:
            g["Resize"] += 1
        elif "softmax" in low:
            g["Softmax"] += 1
        elif "norm" in low:
            g["Normalization"] += 1
        else:
            g["khac"] += 1
    return g


def fusion_depth(names):
    import re
    tot, deepest, n_myl = 0, ("", 0), 0
    for s in names:
        if not (s.startswith("__myl") or s.startswith("__mye")):
            continue
        n_myl += 1
        body = re.sub(r"^__myl_|^__mye", "", s)
        body = re.sub(r"_myl\d+_\d+$", "", body)
        toks = [t for t in re.findall(r"[A-Z][a-z]+", body)]
        tot += len(toks)
        if len(toks) > deepest[1]:
            deepest = (s, len(toks))
    return {"n_myelin": n_myl, "ops_absorbed_est": tot,
            "deepest_name": deepest[0][:80], "deepest_ops": deepest[1]}


def build_all(workspace_gb, force=False):
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    n_onnx, op_hist = count_onnx_nodes(ONNX_PATH)
    print(f"\n{'='*78}\n1. build ENGINE\n{'='*78}")
    print(f"  ONNX  : {ONNX_PATH.name}  ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"          {n_onnx} node — top op: "
          + ", ".join(f"{k}×{v}" for k, v in op_hist.most_common(6)))
    print(f"  workspace gioi han: {workspace_gb} GB  (VRAM device chi 4 GB)\n")

    infos = {}
    for name, (fp16, tf32) in CONFIGS.items():
        path = ENGINE_DIR / f"pmfnet_{name}.engine"
        info = {}
        if path.exists() and not force:
            print(f"  {name:10} da co san -> {path.name} "
                  f"({path.stat().st_size/1e6:.1f} MB), bo qua build")
            info["plan_mb"] = path.stat().st_size / 1e6
            info["build_s"] = None
        else:
            print(f"  {name:10} dang build (fp16={fp16}, tf32={tf32}) ...", flush=True)
            build_engine(ONNX_PATH, path, fp16=fp16, tf32=tf32,
                         workspace_gb=workspace_gb, info=info)
            print(f"  {' '*10} xong sau {info['build_s']:.1f}s, "
                  f"plan {info['plan_mb']:.1f} MB")
        infos[name] = info
    infos["_onnx_nodes"] = n_onnx
    return infos


def report_fusion(runners, infos):
    n_onnx = infos["_onnx_nodes"]
    print(f"\n{'='*78}\n2. fusion — {n_onnx} node ONNX rut lai con bao nhieu layer?\n{'='*78}")
    print(f"  {'cau hinh':12} {'layer sau fusion':>18} {'ty le gop':>11}   nhom layer chinh")
    print("  " + "-" * 74)
    groups, depths = {}, {}
    for name, r in runners.items():
        names = engine_layers(r.engine)
        g = classify_layers(names)
        groups[name] = g
        depths[name] = fusion_depth(names)
        top = ", ".join(f"{k.split(' ')[0]}×{v}" for k, v in g.most_common(4))
        print(f"  {name:12} {r.engine.num_layers:18d} {n_onnx/r.engine.num_layers:10.2f}x   {top}")

    print(f"\n  Chi tiet nhom layer:")
    keys = sorted({k for g in groups.values() for k in g})
    print(f"  {'nhom':30} " + "  ".join(f"{n:>10}" for n in runners))
    print("  " + "-" * 74)
    for k in keys:
        print(f"  {k:30} " + "  ".join(f"{groups[n].get(k,0):10d}" for n in runners))
        if k == "khac":
            frac = max(groups[n].get(k, 0) / runners[n].engine.num_layers for n in runners)
            if frac > 0.10:
                print(f"  {'':30}   ^^ 'khac' chiem {frac*100:.0f}% — phai in ten "
                      f"that ra xem, dung bao cao voi nhom nay")

    print(f"\n  Do sau fusion (uoc luong tu ten kernel Myelin):")
    print(f"  {'cau hinh':12} {'kernel Myelin':>14} {'op bi hap thu':>15} "
          f"{'sau nhat':>10}")
    print("  " + "-" * 74)
    for n, d in depths.items():
        print(f"  {n:12} {d['n_myelin']:14d} {d['ops_absorbed_est']:15d} "
              f"{d['deepest_ops']:9d} op")
    d = depths[next(iter(depths))]
    print(f"\n  Kernel sau nhat: {d['deepest_name']}")
    print(f"    -> {d['deepest_ops']} op ONNX gop thanh mot kernel, mot lan doc/ghi VRAM")
    return {n: dict(g) for n, g in groups.items()}, depths


class TorchGraph:
    def __init__(self, model, device="cuda"):
        self.static_in = torch.empty(1, 3, SIZE, SIZE, device=device)
        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    model(self.static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = model(self.static_in)
            torch.cuda.synchronize()

    def __call__(self, x):
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_out


@torch.inference_mode()
def bench_paired(fns, x, rounds, per_round, warmup=50):
    for f in fns.values():
        for _ in range(warmup):
            f(x)
    torch.cuda.synchronize()

    samples = {k: [] for k in fns}
    print(f"\n  {'vong':5} " + "  ".join(f"{k:>12}" for k in fns) + "     clock/nhiet")
    print("  " + "-" * 74)
    for rd in range(rounds):
        line = []
        for k, f in fns.items():
            t = []
            for _ in range(per_round):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                f(x)
                torch.cuda.synchronize()
                t.append((time.perf_counter() - t0) * 1000)
            samples[k] += t
            line.append(f"{np.mean(t):12.3f}")
        st = query_gpu_state()
        print(f"  {rd+1:<5} " + "  ".join(line)
              + f"     {st.get('clocks_sm_mhz','?')} MHz / {st.get('temp_c','?')}C")
    return samples


@torch.inference_mode()
def check_accuracy(fns, n_images):
    from eval_dice import IMG_DIR, MSK_DIR, dice_iou, load_mask, preprocess

    files = sorted(IMG_DIR.glob("*.jpg"))[:n_images]
    print(f"\n{'='*78}\n4. do chinh xac — {len(files)} anh Kvasir\n{'='*78}")

    dice = {k: [] for k in fns}
    flip = {k: [] for k in fns if k != "torch_tf32"}
    dpair = {k: [] for k in fns if k != "torch_tf32"}
    maxd = {k: 0.0 for k in fns if k != "torch_tf32"}

    for f in files:
        x, (h, w) = preprocess(f)
        gt = load_mask(MSK_DIR / f.name)
        probs = {k: fn(x).float().clone() for k, fn in fns.items()}
        ref = probs["torch_tf32"]
        for k, p in probs.items():
            m = torch.nn.functional.interpolate(p, size=(h, w), mode="bilinear",
                                                align_corners=False)[0, 0] > 0.5
            dice[k].append(dice_iou(m, gt)[0])
            if k != "torch_tf32":
                flip[k].append(((p > 0.5) != (ref > 0.5)).float().mean().item() * 100)
                maxd[k] = max(maxd[k], (p - ref).abs().max().item())
                mr = torch.nn.functional.interpolate(ref, size=(h, w), mode="bilinear",
                                                     align_corners=False)[0, 0] > 0.5
                dpair[k].append(dice_iou(m, mr.cpu().numpy())[0])

    print(f"  {'cau hinh':12} {'Dice vs gt':>11} {'Delta':>9} "
          f"{'Dice vs torch':>14} {'pixel-flip':>11} {'max|d|':>10}")
    print("  " + "-" * 74)
    base = float(np.mean(dice["torch_tf32"]))
    out = {}
    for k in fns:
        d = float(np.mean(dice[k]))
        if k == "torch_tf32":
            print(f"  {k:12} {d:10.4f} {'— moc —':>9} {'—':>14} {'—':>11} {'—':>10}")
            out[k] = {"dice": round(d, 5)}
        else:
            print(f"  {k:12} {d:10.4f} {d-base:+9.5f} {np.mean(dpair[k]):13.4f} "
                  f"{np.mean(flip[k]):10.4f}% {maxd[k]:10.2e}")
            out[k] = {"dice": round(d, 5), "dice_delta": round(d - base, 5),
                      "dice_vs_torch": round(float(np.mean(dpair[k])), 5),
                      "pixel_flip_pct": round(float(np.mean(flip[k])), 5),
                      "max_abs_diff": maxd[k]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=float, default=1.5, help="GB")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--per-round", type=int, default=125, help="iter/cau hinh/vong")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--force-build", action="store_true")
    args = ap.parse_args()

    assert ONNX_PATH.exists(), f"chua co {ONNX_PATH} — chay src/export_onnx.py truoc"
    setup_measurement_env(tf32_matmul=True)

    infos = build_all(args.workspace, force=args.force_build) if not args.skip_build \
        else {**{k: {} for k in CONFIGS}, "_onnx_nodes": count_onnx_nodes(ONNX_PATH)[0]}

    runners = {k: TRTRunner(ENGINE_DIR / f"pmfnet_{k}.engine") for k in CONFIGS}
    for k, r in runners.items():
        vram = getattr(r.engine, "device_memory_size_v2", None) or r.engine.device_memory_size
        infos[k]["vram_runtime_mb"] = vram / 1e6
        infos[k]["layers_after_fusion"] = r.engine.num_layers

    groups, depths = report_fusion(runners, infos)

    model = load_pmfnet(device="cuda")
    fns = {"torch_tf32": TorchGraph(model)}
    fns.update(runners)

    x = torch.randn(1, 3, SIZE, SIZE, device="cuda")
    print(f"\n  Kiem tra shape/dtype dau ra:")
    with torch.inference_mode():
        for k, f in fns.items():
            o = f(x)
            print(f"    {k:12} {tuple(o.shape)} {str(o.dtype):14} "
                  f"range [{o.min().item():.4f}, {o.max().item():.4f}]")

    print(f"\n{'='*78}\n3. LATENCY — paired, {args.rounds} vong × "
          f"{args.per_round} iter/cau hinh\n{'='*78}")
    samples = bench_paired(fns, x, args.rounds, args.per_round)

    print(f"\n  {'cau hinh':12} {'mean':>8} {'p50':>8} {'p99':>8} {'jitter':>7} "
          f"{'miss':>7} {'vs torch':>9} {'VRAM':>9}")
    print("  " + "-" * 74)
    base = summarize(samples["torch_tf32"], BUDGET_MS)
    lat = {}
    for k in fns:
        st = summarize(samples[k], BUDGET_MS)
        vram = infos.get(k, {}).get("vram_runtime_mb")
        print(f"  {k:12} {st.mean_ms:8.3f} {st.p50_ms:8.3f} {st.p99_ms:8.3f} "
              f"{st.p99_ms/st.p50_ms:6.2f}x {st.miss_rate_pct:6.1f}% "
              f"{base.mean_ms/st.mean_ms:8.2f}x "
              + (f"{vram:7.1f}MB" if vram else f"{'—':>9}"))
        lat[k] = {kk: vv for kk, vv in vars(st).items()}

    acc = check_accuracy(fns, args.n_images)

    best = min((k for k in CONFIGS), key=lambda k: summarize(samples[k], BUDGET_MS).mean_ms)
    b = summarize(samples[best], BUDGET_MS)
    print(f"\n{'='*78}\n5. doi chieu ngan sach\n{'='*78}")
    print(f"  Nhanh nhat: {best} = {b.mean_ms:.3f} ms "
          f"({base.mean_ms/b.mean_ms:.2f}x so voi PyTorch+CUDA Graph)")
    print(f"  Muc tieu Ngay 4: infer <= 9.3 ms de latency noi tiep co ghi video < 16.7 ms")
    print(f"  -> {'dat' if b.mean_ms <= 9.3 else 'chua dat'} "
          f"({b.mean_ms:.3f} vs 9.3, con thieu {max(0, b.mean_ms-9.3):.3f} ms)")
    e2e_est = 20.10 - base.mean_ms + b.mean_ms
    print(f"  Uoc tinh latency noi tiep (co ghi): {e2e_est:.2f} ms "
          f"-> {1000/e2e_est:.1f} FPS  ({'dat' if e2e_est <= 16.7 else 'chua dat'})")

    save_result({
        "config": "trt_precision_sweep", "stage": "infer",
        "onnx_nodes": infos["_onnx_nodes"],
        "engines": {k: {kk: vv for kk, vv in infos[k].items() if kk != "flags"}
                    for k in CONFIGS},
        "layer_groups": groups, "fusion_depth": depths,
        "latency": lat, "accuracy": acc,
        "best": best, "e2e_serial_estimate_ms": round(e2e_est, 2),
        "env_after": query_gpu_state(),
    })


if __name__ == "__main__":
    main()
