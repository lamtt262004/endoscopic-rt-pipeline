"""
export_onnx.py — Ngay 3. PMFNet -> ONNX, thu ca hai exporter roi so sanh.

    python src/export_onnx.py              # export ca hai + onnxsim + validate
    python src/export_onnx.py --n 20       # so anh dung de validate
    python src/export_onnx.py --which legacy
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import load_pmfnet, save_result

_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = _ROOT / "onnx"
SHAPE = (1, 3, 256, 256)

SCAFFOLD_OPS = {"Shape", "Gather", "Unsqueeze", "Concat", "Squeeze",
                "Slice", "Cast", "ConstantOfShape", "Range"}


def make_export_friendly(model, verbose=True):
    import torch.nn as nn

    changed = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d) or not isinstance(m.padding, str):
            continue
        if m.padding != "same":
            raise RuntimeError(f"{name}: padding='{m.padding}' chua xu ly")
        pads = []
        for k, d in zip(m.kernel_size, m.dilation):
            total = d * (k - 1)
            if total % 2 != 0:
                raise RuntimeError(
                    f"{name}: kernel={k} dilation={d} -> tong padding {total} le, "
                    "phai padding lech hai ben; can chen nn.ZeroPad2d thay vi doi so.")
            pads.append(total // 2)
        m.padding = tuple(pads)
        changed.append((name, tuple(pads), m.kernel_size, m.dilation))

    if verbose and changed:
        print(f"  [0] Doi padding='same' -> so nguyen cho {len(changed)} Conv2d")
        dilated = [c for c in changed if max(c[3]) > 1]
        print(f"      trong do {len(dilated)} conv co dilation (chinh la thu pham):")
        for n, p, k, d in dilated[:4]:
            print(f"        {n:34} k={k} dil={d} -> padding={p}")
        if len(dilated) > 4:
            print(f"        ... va {len(dilated)-4} conv nua")
    return model, changed


def verify_patch_equivalent(model_patched, model_ref, x):
    with torch.inference_mode():
        a, b = model_ref(x), model_patched(x)
    d = (a - b).abs().max().item()
    print(f"      kiem chung tuong duong: max |delta| = {d:.2e} "
          f"({'bit-exact, an toan' if d == 0 else 'co lech — dung lai'})")
    if d != 0:
        raise RuntimeError("Phep doi padding lam doi ket qua model!")
    return d


def graph_stats(onnx_model):
    ops = Counter(n.op_type for n in onnx_model.graph.node)
    total = sum(ops.values())
    scaffold = sum(v for k, v in ops.items() if k in SCAFFOLD_OPS)
    return {"total": total, "scaffold": scaffold,
            "scaffold_pct": 100 * scaffold / total if total else 0, "ops": ops}


def print_stats(tag, st, top=10):
    print(f"    {tag:22} {st['total']:5d} node  |  gian giao {st['scaffold']:4d} "
          f"({st['scaffold_pct']:.0f}%)")
    if top:
        items = ", ".join(f"{k}:{v}" for k, v in st["ops"].most_common(top))
        print(f"      {items}")


def export_one(model, x, path, dynamo):
    t0 = time.perf_counter()
    try:
        torch.onnx.export(
            model, (x,), str(path),
            opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamo=dynamo,
        )
        return True, time.perf_counter() - t0, None
    except Exception as e:
        return False, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def simplify(path):
    import onnx
    from onnxsim import simplify as onnxsim_simplify

    m = onnx.load(str(path))
    before = graph_stats(m)
    m_sim, check = onnxsim_simplify(m)
    if not check:
        print("      [!] onnxsim bao check=False — do thi rut gon khong tuong duong")
        return before, None
    onnx.save(m_sim, str(path))
    return before, graph_stats(m_sim)


@torch.inference_mode()
def validate(onnx_path, model, files, device="cuda"):
    import onnxruntime as ort
    from eval_dice import dice_iou, load_mask, preprocess, prob_to_mask, MSK_DIR

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    max_deltas, flips, d_torch, d_onnx, dice_pair = [], [], [], [], []
    for f in files:
        x, hw = preprocess(f, device=device)
        p_t = model(x)
        p_o = torch.from_numpy(
            sess.run(None, {iname: x.cpu().numpy()})[0]).to(device)

        max_deltas.append((p_t - p_o).abs().max().item())
        flips.append(((p_t > 0.5) != (p_o > 0.5)).float().mean().item())

        m_t, m_o = prob_to_mask(p_t, hw), prob_to_mask(p_o, hw)
        gt = load_mask(MSK_DIR / f.name)
        d_torch.append(dice_iou(m_t, gt)[0])
        d_onnx.append(dice_iou(m_o, gt)[0])
        dice_pair.append(dice_iou(m_o, m_t.cpu().numpy())[0])

    return {
        "max_delta": float(np.max(max_deltas)),
        "pixel_flip_pct": float(np.mean(flips) * 100),
        "dice_torch": float(np.mean(d_torch)),
        "dice_onnx": float(np.mean(d_onnx)),
        "dice_vs_torch": float(np.mean(dice_pair)),
    }


def report(v):
    print(f"      max |delta xac suat|   : {v['max_delta']:.2e}   (tham khao)")
    print(f"      pixel doi nhan @0.5    : {v['pixel_flip_pct']:.4f} %")
    print(f"      Dice  PyTorch          : {v['dice_torch']:.4f}")
    print(f"      Dice  ONNX             : {v['dice_onnx']:.4f}   "
          f"(delta {v['dice_onnx']-v['dice_torch']:+.5f})")
    print(f"      Dice  ONNX vs PyTorch  : {v['dice_vs_torch']:.4f}  <- 1.0000 la trung khop")
    ok = (abs(v["dice_onnx"] - v["dice_torch"]) < 0.001
          and v["pixel_flip_pct"] < 0.05)
    print(f"      => {'dat' if ok else 'can xem lai'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="so anh de validate")
    ap.add_argument("--which", choices=["legacy", "dynamo", "both"], default="both")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False

    model = load_pmfnet(device="cuda")
    x = torch.randn(*SHAPE, device="cuda")

    model_ref = load_pmfnet(device="cuda")
    print()
    make_export_friendly(model)
    verify_patch_equivalent(model, model_ref, x)
    del model_ref
    torch.cuda.empty_cache()

    from eval_dice import IMG_DIR
    files = sorted(IMG_DIR.glob("*.jpg"))[:args.n]

    variants = ([("legacy", False), ("dynamo", True)] if args.which == "both"
                else [(args.which, args.which == "dynamo")])
    results = {}

    for name, dynamo in variants:
        print("\n" + "=" * 74)
        print(f"exporter: {name}   (dynamo={dynamo})")
        print("=" * 74)
        path = OUT_DIR / f"pmfnet_{name}.onnx"

        ok, dt, err = export_one(model, x, path, dynamo)
        if not ok:
            print(f"  [X] export that bai sau {dt:.1f}s")
            print(f"      {err[:400]}")
            results[name] = {"export_ok": False, "error": err[:400]}
            continue
        print(f"  [1] export OK ({dt:.1f}s) -> {path.stat().st_size/1e6:.1f} MB")

        before, after = simplify(path)
        print("  [2] do thi:")
        print_stats("truoc onnxsim", before)
        if after:
            print_stats("sau  onnxsim", after)
            print(f"      => giam {before['total']-after['total']} node "
                  f"({1-after['total']/before['total']:.0%}), "
                  f"gian giao {before['scaffold']} -> {after['scaffold']}")
        print(f"      kich thuoc sau rut gon: {path.stat().st_size/1e6:.1f} MB")

        print(f"  [3] doi chieu voi PyTorch tren {len(files)} anh that "
              f"(onnxruntime CPU, hoi lau):")
        v = validate(path, model, files)
        passed = report(v)
        results[name] = {"export_ok": True, "export_s": round(dt, 1),
                         "nodes_before": before["total"],
                         "nodes_after": after["total"] if after else None,
                         "scaffold_before": before["scaffold"],
                         "scaffold_after": after["scaffold"] if after else None,
                         "passed": passed, **v}

    print("\n" + "=" * 74)
    print("so sanh hai exporter")
    print("=" * 74)
    good = {k: v for k, v in results.items() if v.get("export_ok")}
    if not good:
        print("  Ca hai deu that bai. Doc thong bao loi o tren.")
    else:
        print(f"  {'exporter':10} {'node':>8} {'gian giao':>10} {'flip%':>9} "
              f"{'Dice':>8} {'dat?':>6}")
        for k, v in good.items():
            print(f"  {k:10} {v['nodes_after'] or v['nodes_before']:8d} "
                  f"{v['scaffold_after'] if v['scaffold_after'] is not None else v['scaffold_before']:10d} "
                  f"{v['pixel_flip_pct']:9.4f} {v['dice_onnx']:8.4f} "
                  f"{'dat' if v['passed'] else 'khong':>6}")
        best = min(good.items(),
                   key=lambda kv: (not kv[1]["passed"],
                                   kv[1]["nodes_after"] or kv[1]["nodes_before"]))
        print(f"\n  => Dung ban '{best[0]}' cho Ngay 4: it node nhat trong so cac ban dat.")
        print(f"     File: onnx/pmfnet_{best[0]}.onnx")
        print("     Node cang it + gian giao cang it => TensorRT cang de fuse.")

    save_result({"config": "onnx_export", "stage": "export",
                 "shape": list(SHAPE), "opset": 17, "results": results})


if __name__ == "__main__":
    main()
