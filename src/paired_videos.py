"""So 3 video theo kieu paired - xen ke vong."""
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bench_utils import load_pmfnet, setup_measurement_env
from video_infer import GpuPreprocess, GraphInfer, postprocess, overlay

SCRATCH = ROOT / "benchmarks" / "_tmp"
SCRATCH.mkdir(parents=True, exist_ok=True)
VIDEOS = [
    ("0220d11b", "polyp bleeding"),
    ("76866169", "small polyp"),
    ("5af764dc", "flat polyp"),
]
ROUNDS = 4
PER_ROUND = 100
STAGES = ("decode", "h2d", "preprocess", "infer", "postprocess", "overlay", "d2h", "encode")


def clocks():
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        sm, t, p = [x.strip() for x in o.split(",")]
        return f"{sm} MHz / {t}C / {p} W"
    except Exception:
        return "?"


setup_measurement_env(tf32_matmul=True)
model = load_pmfnet(device="cuda")
infer = GraphInfer(model)


class Runner:
    def __init__(self, vid, label):
        self.path = next((ROOT / "hyper-kvasir-videos" / "videos").glob(vid + "*.avi"))
        self.vid, self.label = vid, label
        self.cap = cv2.VideoCapture(str(self.path))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.nframe = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.pre = GpuPreprocess(self.h, self.w)
        self.out_pin = torch.empty((self.h, self.w, 3), dtype=torch.uint8, pin_memory=True)
        self.writer = cv2.VideoWriter(str(SCRATCH / f"_out_{vid}.mp4"),
                                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.w, self.h))
        assert self.writer.isOpened()
        self.bitrate = self.path.stat().st_size * 8 / (self.nframe / fps) / 1e6
        self.st = {k: [] for k in STAGES}
        self.e2e = []

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            self.cap.release()
            self.cap = cv2.VideoCapture(str(self.path))
            ok, f = self.cap.read()
        return f

    def frame(self, measure=False, record=True):
        def tick():
            if measure:
                torch.cuda.synchronize()
                return time.perf_counter()
            return 0.0

        t0 = tick()
        f = self.read()
        t1 = tick()
        g = self.pre.upload(f)
        t2 = tick()
        x = self.pre(g)
        t3 = tick()
        prob = infer(x)
        t4 = tick()
        m = postprocess(prob, (self.h, self.w))
        t5 = tick()
        vis = overlay(g, m)
        t6 = tick()
        self.out_pin.copy_(vis)
        t7 = tick()
        self.writer.write(self.out_pin.numpy())
        t8 = tick()
        if measure and record:
            for k, v in zip(STAGES, (t1-t0, t2-t1, t3-t2, t4-t3,
                                     t5-t4, t6-t5, t7-t6, t8-t7)):
                self.st[k].append(v * 1000)


runners = [Runner(v, lab) for v, lab in VIDEOS]

print(f"\n{'video':10} {'nhan':24} {'frame':>6} {'bitrate':>9}")
print("-" * 54)
for r in runners:
    print(f"{r.vid:10} {r.label:24} {r.nframe:6d} {r.bitrate:7.2f} Mb/s")

print(f"\nwarm-up 50 frame/video ... (clock truoc: {clocks()})")
with torch.inference_mode():
    for r in runners:
        for _ in range(50):
            r.frame()
torch.cuda.synchronize()

print(f"\n{'='*78}")
print(f"do xen ke: {ROUNDS} vong x {PER_ROUND} frame/video")
print(f"{'='*78}")
print(f"{'vong':6} {'clock/nhiet':>26}  " + "  ".join(f"{r.vid[:8]:>10}" for r in runners))
print(f"{'':6} {'':>26}  " + "  ".join(f"{'infer ms':>10}" for _ in runners))
print("-" * 78)

per_round_infer = {r.vid: [] for r in runners}
with torch.inference_mode():
    for rd in range(ROUNDS):
        c = clocks()
        line = []
        for r in runners:
            n0 = len(r.st["infer"])
            for _ in range(PER_ROUND):
                r.frame(measure=True)
            v = float(np.mean(r.st["infer"][n0:]))
            per_round_infer[r.vid].append(v)
            line.append(f"{v:10.3f}")
        print(f"{rd+1:<6} {c:>26}  " + "  ".join(line))

print(f"\n{'='*78}\nlatency lien mach (khong sync giua cac khau), xen ke\n{'='*78}")
with torch.inference_mode():
    for rd in range(ROUNDS):
        for r in runners:
            for _ in range(PER_ROUND):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                r.frame()
                torch.cuda.synchronize()
                r.e2e.append((time.perf_counter() - t0) * 1000)

print(f"\n{'khau':14} " + "  ".join(f"{r.vid[:8]:>10}" for r in runners) + "     nhan xet")
print("-" * 78)
for k in STAGES:
    vals = [float(np.mean(r.st[k])) for r in runners]
    spread = (max(vals) - min(vals)) / min(vals) * 100 if min(vals) > 0 else 0
    tag = ""
    if k == "infer":
        tag = f"<- bien doi chung, lech {spread:.1f}%"
    elif spread > 20:
        tag = f"<- lech {spread:.0f}% theo noi dung"
    print(f"{k:14} " + "  ".join(f"{v:10.3f}" for v in vals) + f"     {tag}")
print("-" * 78)
tots = [sum(float(np.mean(r.st[k])) for k in STAGES) for r in runners]
print(f"{'tong khau':14} " + "  ".join(f"{v:10.3f}" for v in tots))
e2es = [np.array(r.e2e) for r in runners]
print(f"{'E2E mean':14} " + "  ".join(f"{v.mean():10.3f}" for v in e2es))
print(f"{'E2E p99':14} " + "  ".join(f"{np.percentile(v,99):10.3f}" for v in e2es))
print(f"{'FPS noi tiep':14} " + "  ".join(f"{1000/v.mean():10.1f}" for v in e2es))

print(f"\nclock sau cung: {clocks()}")
for r in runners:
    r.writer.release()
    r.cap.release()

print(f"\n{'='*78}\nDO troi cua may (infer theo tung vong)\n{'='*78}")
for r in runners:
    s = per_round_infer[r.vid]
    print(f"  {r.vid:10} " + " -> ".join(f"{v:.2f}" for v in s)
          + f"   (vong cuoi / vong dau = {s[-1]/s[0]:.3f}x)")
