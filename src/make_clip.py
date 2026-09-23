"""
make_clip.py - cat doan dep nhat cua demo thanh clip ngan de nhung vao README.
Xuat H.264 + faststart: OpenCV o day chi mo duoc mp4v, ma trinh duyet khong
phat duoc codec do. Can imageio-ffmpeg.

    python src/make_clip.py                      # tu chon doan
    python src/make_clip.py --secs 12 --crf 20   # net hon, nang hon
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmarks"


def to_h264(path, crf=23):
    import subprocess
    from imageio_ffmpeg import get_ffmpeg_exe

    tmp = path.with_name(path.stem + "_h264.mp4")
    cmd = [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path),
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    subprocess.run(cmd, check=True)
    before = path.stat().st_size
    tmp.replace(path)
    return before, path.stat().st_size


def faststart(path):
    d = path.read_bytes()
    boxes, off = {}, 0
    while off < len(d) - 8:
        n = int.from_bytes(d[off:off + 4], "big")
        if n < 8:
            break
        boxes[d[off + 4:off + 8].decode("latin-1", "replace")] = (off, n)
        off += n
    if "moov" not in boxes or "mdat" not in boxes:
        return False
    (mo, mn), (do, dn) = boxes["moov"], boxes["mdat"]
    if mo < do:
        return False

    moov = bytearray(d[mo:mo + mn])

    def patch(buf, base=0):
        i = 0
        while i < len(buf) - 8:
            n = int.from_bytes(buf[i + base:i + base + 4], "big")
            t = bytes(buf[i + base + 4:i + base + 8]).decode("latin-1", "replace")
            if n < 8:
                break
            if t in ("moov", "trak", "mdia", "minf", "stbl"):
                patch(buf, base + i + 8)
            elif t in ("stco", "co64"):
                w = 4 if t == "stco" else 8
                p = base + i + 16
                cnt = int.from_bytes(buf[base + i + 12:base + i + 16], "big")
                for k in range(cnt):
                    q = p + k * w
                    v = int.from_bytes(buf[q:q + w], "big") + mn
                    buf[q:q + w] = v.to_bytes(w, "big")
            i += n

    patch(moov)
    path.write_bytes(d[:do] + bytes(moov) + d[do:do + dn])
    return True


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
    ap.add_argument("--crf", type=int, default=23,
                    help="chat luong H.264: thap hon = net hon, nang hon (18-28)")
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

    try:
        a, b = to_h264(out, args.crf)
        note = f"H.264 crf {args.crf}, {a/1e6:.2f} -> {b/1e6:.2f} MB"
    except Exception as e:
        faststart(out)
        note = f"ffmpeg khong chay duoc ({e}) - giu mp4v, trinh duyet se khong phat duoc"

    mb = out.stat().st_size / 1e6
    print(f"  {out.name}: {k} frame @ {fps:.0f} fps = {k/fps:.1f}s | {w}x{h} | {mb:.2f} MB")
    print(f"  {note}")
    print(f"  {'vuot 10 MB - giam --secs hoac tang --crf' if mb > 10 else 'trong gioi han 10 MB cua GitHub'}")


if __name__ == "__main__":
    main()
