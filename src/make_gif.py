"""
make_gif.py — Cat mot doan demo_*.mp4 thanh GIF de nhung vao README.

    python src/make_gif.py                          # tu chon doan dep nhat
    python src/make_gif.py --start 240 --len 100
    python src/make_gif.py --src demo_nopolyp --out fp_demo
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmarks"


def pick_segment(path, n_probe, seg_len):
    cap = cv2.VideoCapture(str(path))
    areas = []
    for _ in range(n_probe):
        ok, f = cap.read()
        if not ok:
            break
        b, g, r = f[:, :, 0].astype(np.int16), f[:, :, 1].astype(np.int16), f[:, :, 2].astype(np.int16)
        areas.append(float(((r - np.maximum(b, g)) > 40).mean()))
    cap.release()
    a = np.array(areas)
    if len(a) <= seg_len:
        return 0
    win = np.convolve(a, np.ones(seg_len) / seg_len, mode="valid")
    return int(np.argmax(win))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="demo_polyp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--start", type=int, default=None, help="bo trong = tu chon")
    ap.add_argument("--len", type=int, default=100, help="so frame nguon lay ra")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=int, default=12, help="fps cua GIF")
    ap.add_argument("--colors", type=int, default=128)
    ap.add_argument("--probe", type=int, default=500)
    args = ap.parse_args()

    src = OUT_DIR / f"{args.src}.mp4"
    assert src.exists(), f"chua co {src} — chay src/make_demo.py truoc"
    out = OUT_DIR / f"{args.out or args.src}.gif"

    cap = cv2.VideoCapture(str(src))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    step = max(1, round(src_fps / args.fps))

    start = args.start
    if start is None:
        start = pick_segment(src, args.probe, args.len)
        print(f"  tu chon doan bat dau tu frame {start}")

    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, i = [], 0
    while len(frames) * step < args.len:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            nh = int(round(h * args.width / w))
            small = cv2.resize(f, (args.width, nh), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    assert frames, "khong doc duoc frame nao"

    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=args.colors) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:],
                duration=int(1000 / args.fps), loop=0, optimize=True)

    mb = out.stat().st_size / 1e6
    print(f"  {out.name}: {len(pal)} frame @ {args.fps} fps = "
          f"{len(pal)/args.fps:.1f}s | {args.width}x{pal[0].size[1]} | "
          f"{args.colors} mau | {mb:.2f} MB")
    if mb > 5:
        print(f"  {mb:.1f} MB > ngan sach 5 MB — giam --len, --width hoac --colors")
    else:
        print(f"  trong ngan sach 5 MB")


if __name__ == "__main__":
    main()
