"""
make_clip.py - cat doan dep nhat cua demo thanh clip ngan de nhung vao README.

    python src/make_clip.py                      # tu chon doan
    python src/make_clip.py --secs 12 --start 219
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmarks"


def load_areas(path):
    j = path.with_name(path.stem + "_area.json")
    if not j.exists():
        raise SystemExit(
            f"thieu {j.name}. Chay lai: python src/make_demo.py --n 500 --trt fp16")
    return np.array(json.loads(j.read_text(encoding="utf-8")))


def pick(areas, win):
    if len(areas) <= win:
        return 0
    sc = [areas[i:i+win].mean() - areas[i:i+win].std()
          for i in range(len(areas) - win)]
    return int(np.argmax(sc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="demo_polyp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--secs", type=float, default=12.0)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--probe", type=int, default=500)
    args = ap.parse_args()

    src = OUT_DIR / f"{args.src}.mp4"
    assert src.exists(), f"chua co {src}"
    out = OUT_DIR / f"{args.out or (args.src + '_clip')}.mp4"

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    n = int(round(args.secs * fps))

    areas = load_areas(src)
    start = args.start if args.start is not None else pick(areas, n)
    seg = areas[start:start + n]
    print(f"  doan frame {start}-{start+n}  "
          f"mask: trung binh {seg.mean():.2f}%  p50 {np.percentile(seg,50):.2f}%  "
          f"max {seg.max():.1f}%   (ca video: p50 {np.percentile(areas,50):.2f}%)")

    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert wr.isOpened(), "VideoWriter khong mo duoc"
    k = 0
    while k < n:
        ok, f = cap.read()
        if not ok:
            break
        wr.write(f)
        k += 1
    cap.release()
    wr.release()

    mb = out.stat().st_size / 1e6
    print(f"  {out.name}: {k} frame @ {fps:.0f} fps = {k/fps:.1f}s | {w}x{h} | {mb:.2f} MB")
    if mb > 10:
        print(f"  vuot 10 MB - giam --secs")
    else:
        print(f"  trong gioi han 10 MB cua GitHub")


if __name__ == "__main__":
    main()
