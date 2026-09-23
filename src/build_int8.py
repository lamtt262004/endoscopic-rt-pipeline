"""
build_int8.py — Luong tu hoa int8 bang PTQ + calibration.

    python src/build_int8.py                    # build ca hai calibrator + do
    python src/build_int8.py --skip-build       # dung engine da co
    python src/build_int8.py --n-calib 300 --n-eval 200
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import (load_pmfnet, query_gpu_state, save_result,
                         setup_measurement_env, summarize)
from build_engine import TorchGraph, bench_paired
from calib_loader import cache_path_for, calib_files, eval_files, make_calibrator
from trt_utils import TRTRunner, build_engine

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "onnx" / "pmfnet_legacy.onnx"
ENGINE_DIR = ROOT / "engines"
BUDGET_MS = 16.7
ALGOS = ["entropy2", "minmax"]


def build_int8(algo, n_calib, workspace_gb, force=False):
    path = ENGINE_DIR / f"pmfnet_trt_int8_{algo}.engine"
    info = {}
    if path.exists() and not force:
        print(f"  int8_{algo:9} da co -> {path.name} ({path.stat().st_size/1e6:.1f} MB)")
        info["plan_mb"] = path.stat().st_size / 1e6
        return path, info

    print(f"  int8_{algo:9} dang build ({n_calib} anh calibration) ...", flush=True)
    cal = make_calibrator(algo, calib_files(n_calib), cache_path_for(algo, n_calib))
    build_engine(ONNX_PATH, path, fp16=True, int8=True, tf32=True,
                 workspace_gb=workspace_gb, calibrator=cal, info=info)
    print(f"  {' '*14} xong sau {info['build_s']:.1f}s, plan {info['plan_mb']:.1f} MB")
    return path, info


@torch.inference_mode()
def accuracy(fns, files):
    from eval_dice import MSK_DIR, dice_iou, load_mask, preprocess

    print(f"\n{'='*80}\nDO chinh xac — {len(files)} anh, khong chong tap calibration\n{'='*80}")
    dice = {k: [] for k in fns}
    flip = {k: [] for k in fns if k != "torch"}
    for f in files:
        x, (h, w) = preprocess(f)
        gt = load_mask(MSK_DIR / f.name)
        probs = {k: fn(x).float().clone() for k, fn in fns.items()}
        ref = probs["torch"]
        for k, p in probs.items():
            m = torch.nn.functional.interpolate(p, size=(h, w), mode="bilinear",
                                                align_corners=False)[0, 0] > 0.5
            dice[k].append(dice_iou(m, gt)[0])
            if k != "torch":
                flip[k].append(((p > 0.5) != (ref > 0.5)).float().mean().item() * 100)

    base = float(np.mean(dice["torch"]))
    print(f"  {'cau hinh':16} {'Dice':>9} {'Delta':>10} {'% mat':>8} {'pixel-flip':>12}")
    print("  " + "-" * 62)
    out = {}
    for k in fns:
        d = float(np.mean(dice[k]))
        if k == "torch":
            print(f"  {k:16} {d:9.4f} {'— moc —':>10} {'—':>8} {'—':>12}")
        else:
            print(f"  {k:16} {d:9.4f} {d-base:+10.5f} {(d/base-1)*100:+7.2f}% "
                  f"{np.mean(flip[k]):11.4f}%")
            out[k] = {"dice": round(d, 5), "delta": round(d - base, 5),
                      "pct_lost": round((d / base - 1) * 100, 3),
                      "pixel_flip_pct": round(float(np.mean(flip[k])), 4)}
    out["torch"] = {"dice": round(base, 5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--workspace", type=float, default=1.5)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--per-round", type=int, default=125)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--force-build", action="store_true")
    args = ap.parse_args()

    setup_measurement_env(tf32_matmul=True)
    print(f"{'='*80}\n1. build ENGINE int8 — hai calibrator tren cung du lieu\n{'='*80}")
    print(f"  calibration: {args.n_calib} anh dau   |   danh gia: anh "
          f"[{args.n_calib}:{args.n_calib+args.n_eval}]  (khong chong nhau)")
    infos = {}
    if not args.skip_build:
        for algo in ALGOS:
            _, infos[f"int8_{algo}"] = build_int8(algo, args.n_calib,
                                                  args.workspace, args.force_build)

    tags = {"trt_tf32": ENGINE_DIR / "pmfnet_trt_tf32.engine",
            "trt_fp16": ENGINE_DIR / "pmfnet_trt_fp16.engine"}
    for algo in ALGOS:
        p = ENGINE_DIR / f"pmfnet_trt_int8_{algo}.engine"
        if p.exists():
            tags[f"int8_{algo}"] = p
    runners = {k: TRTRunner(v) for k, v in tags.items()}
    for k, r in runners.items():
        vram = getattr(r.engine, "device_memory_size_v2", None) or r.engine.device_memory_size
        infos.setdefault(k, {})["vram_runtime_mb"] = vram / 1e6
        infos[k]["layers"] = r.engine.num_layers
        infos[k]["plan_mb"] = tags[k].stat().st_size / 1e6

    print(f"\n  {'cau hinh':16} {'layer':>7} {'plan MB':>9} {'VRAM MB':>9}")
    print("  " + "-" * 46)
    for k in runners:
        print(f"  {k:16} {infos[k]['layers']:7d} {infos[k]['plan_mb']:9.1f} "
              f"{infos[k]['vram_runtime_mb']:9.1f}")

    model = load_pmfnet(device="cuda")
    fns = {"torch": TorchGraph(model)}
    fns.update(runners)

    print(f"\n{'='*80}\n2. LATENCY — paired, {args.rounds} vong × "
          f"{args.per_round} iter\n{'='*80}")
    x = torch.randn(1, 3, 256, 256, device="cuda")
    samples = bench_paired(fns, x, args.rounds, args.per_round)

    print(f"\n  {'cau hinh':16} {'mean':>8} {'p50':>8} {'p99':>8} {'jitter':>7} "
          f"{'vs torch':>9} {'vs fp16':>8}")
    print("  " + "-" * 70)
    base = summarize(samples["torch"], BUDGET_MS)
    f16 = summarize(samples["trt_fp16"], BUDGET_MS)
    lat = {}
    for k in fns:
        st = summarize(samples[k], BUDGET_MS)
        print(f"  {k:16} {st.mean_ms:8.3f} {st.p50_ms:8.3f} {st.p99_ms:8.3f} "
              f"{st.p99_ms/st.p50_ms:6.2f}x {base.p50_ms/st.p50_ms:8.2f}x "
              f"{f16.p50_ms/st.p50_ms:7.2f}x")
        lat[k] = {kk: vv for kk, vv in vars(st).items()}

    acc = accuracy(fns, eval_files(args.n_calib, args.n_eval))

    print(f"\n{'='*80}\n3. ket luan\n{'='*80}")
    print("Du doan tu dai activation: GELU max/p99.9 = 33.6x -> MinMax gian scale.")
    if all(f"int8_{a}" in acc for a in ALGOS):
        e, m = acc["int8_entropy2"], acc["int8_minmax"]
        print(f"\n  Do duoc: entropy2 Dice {e['dice']:.4f} ({e['pct_lost']:+.2f}%), "
              f"minmax {m['dice']:.4f} ({m['pct_lost']:+.2f}%)")
        if e["dice"] > m["dice"] * 1.01:
            print(f"  => du doan dung: Entropy2 thang ro ({e['dice']/m['dice']:.3f}x)")
        elif abs(e["dice"] - m["dice"]) / max(m["dice"], 1e-9) < 0.01:
            print(f"  => du doan sai: hai calibrator gan nhu nhu nhau. "
                  f"TRT co the da tu giu cac layer duoi dai o fp16.")
        else:
            print(f"  => du doan nguoc: MinMax lai thang.")

    save_result({
        "config": "int8_ptq", "stage": "infer",
        "n_calib": args.n_calib, "n_eval": args.n_eval,
        "engines": infos, "latency": lat, "accuracy": acc,
        "env_after": query_gpu_state(),
    })


if __name__ == "__main__":
    main()
