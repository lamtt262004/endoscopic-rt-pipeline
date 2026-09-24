"""
make_demo.py 

    python src/make_demo.py                         # ca hai video (co polyp + khong polyp)
    python src/make_demo.py --n 600                 # dai hon
    python src/make_demo.py --only nopolyp          # chi mot cai
    python src/make_demo.py --video 1de3ef0f        # video bat ky theo id
    python src/make_demo.py --video "flat polyp" --out flat   # ... hoac theo nhan
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import load_pmfnet, setup_measurement_env
from video_infer import GpuPreprocess, GraphInfer, TRTInfer, postprocess, overlay

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "hyper-kvasir-videos" / "videos"
OUT_DIR = ROOT / "benchmarks"

TARGETS = {
    "polyp":   ("76866169", "small polyp",           "co polyp (small polyp)"),
    "nopolyp": ("164c76bd", "cecum ileocecal valve", "khong co polyp (van hoi manh trang)"),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hud(img, title, area_pct, idx, n):
    h, w = img.shape[:2]
    pad = 10
    box_h = 92
    sub = img[0:box_h, 0:420]
    img[0:box_h, 0:420] = (sub * 0.35).astype(np.uint8)

    cv2.putText(img, title, (pad, 26), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"mask: {area_pct:5.2f}% khung hinh", (pad, 52),
                FONT, 0.55, (120, 220, 255), 1, cv2.LINE_AA)

    bar_w, bar_x, bar_y = 260, pad, 66
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (90, 90, 90), 1)
    filled = int(min(area_pct / 20.0, 1.0) * (bar_w - 2))
    if filled > 0:
        cv2.rectangle(img, (bar_x + 1, bar_y + 1),
                      (bar_x + 1 + filled, bar_y + 11), (60, 60, 230), -1)
    cv2.putText(img, f"{idx+1}/{n}", (bar_x + bar_w + 12, bar_y + 11),
                FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


@torch.inference_mode()
def make_one(key, infer, n, fourcc_str="mp4v", spec=None, out_name=None):
    from videos import resolve, label_for
    if spec is not None:
        path = resolve(spec)
        finding = label_for(path)
        title = finding.upper()[:44]
        key = out_name or path.stem[:8]
    else:
        vid, finding, title = TARGETS[key]
        path = resolve(vid)
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"khong mo duoc {path}"
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n, total)

    out_path = OUT_DIR / f"demo_{key}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc_str),
                             fps, (w, h))
    assert writer.isOpened(), "VideoWriter khong mo duoc — thieu codec?"

    pre = GpuPreprocess(h, w)
    print(f"\n{key:8} | {path.name[:8]} | {finding}")
    print(f"         {w}x{h} @ {fps:.0f} fps, xuat {n}/{total} frame -> {out_path.name}")

    for _ in range(20):
        ok, f = cap.read()
        if ok:
            infer(pre(pre.upload(f)))
    cap.release()
    cap = cv2.VideoCapture(str(path))

    areas = []
    t0 = time.perf_counter()
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        g = pre.upload(frame)
        mask = postprocess(infer(pre(g)), (h, w))
        area = mask.float().mean().item() * 100
        areas.append(area)
        vis = overlay(g, mask).cpu().numpy()
        writer.write(draw_hud(vis, title, area, i, n))
    secs = time.perf_counter() - t0
    writer.release()
    cap.release()

    a = np.array(areas)
    mb = out_path.stat().st_size / 1e6
    print(f"         mask: mean {a.mean():5.2f}%  p50 {np.percentile(a,50):5.2f}%  "
          f"max {a.max():5.2f}%  |  frame >1%: {(a>1).sum()}/{len(a)} ({(a>1).mean()*100:.0f}%)")
    print(f"         {len(a)} frame / {secs:.1f}s  |  file {mb:.1f} MB")
    return key, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="so frame moi video")
    ap.add_argument("--only", choices=list(TARGETS), default=None)
    ap.add_argument("--video", default=None,
                    help="id / tu khoa nhan / duong dan — xuat demo cho video nay")
    ap.add_argument("--out", default=None, help="ten file ra (khong co .mp4)")
    ap.add_argument("--trt", nargs="?", const="fp16", default="fp16",
                    help="engine dung de render; 'none' de dung PyTorch")
    args = ap.parse_args()

    setup_measurement_env(tf32_matmul=True)
    ep = ROOT / "engines" / f"pmfnet_trt_{args.trt}.engine"
    if args.trt != "none" and ep.exists():
        infer = TRTInfer(ep)
        print(f"backend: TensorRT {args.trt.upper()}  ({ep.name})")
    else:
        infer = GraphInfer(load_pmfnet(device="cuda"))
        print("backend: PyTorch + CUDA Graph")

    if args.video:
        make_one(None, infer, args.n, spec=args.video, out_name=args.out)
        return

    keys = [args.only] if args.only else list(TARGETS)
    res = [make_one(k, infer, args.n) for k in keys]

    if len(res) == 2:
        print(f"\n{'='*70}\nSO sanh\n{'='*70}")
        print(f"{'':10} {'mean':>8} {'p50':>8} {'max':>8} {'>1% khung':>11}")
        for k, a in res:
            print(f"{k:10} {a.mean():7.2f}% {np.percentile(a,50):7.2f}% "
                  f"{a.max():7.2f}% {(a>1).mean()*100:10.1f}%")
        (_, ap_), (_, an) = res
        print(f"\nTi so mean co-polyp / khong-polyp = {ap_.mean()/an.mean():.2f}x")
        print("Neu ti so nay gan 1 thi model khong phan biet duoc hai truong hop.")


if __name__ == "__main__":
    main()
