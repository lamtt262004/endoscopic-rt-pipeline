"""Throughput pipelining cho 3 video"""
import subprocess, sys
from pathlib import Path
import cv2, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bench_utils import load_pmfnet, setup_measurement_env
from video_infer import GpuPreprocess, GraphInfer, run_pipelined

SCRATCH = ROOT / "benchmarks" / "_tmp"
SCRATCH.mkdir(parents=True, exist_ok=True)
VIDEOS = [("0220d11b", "polyp bleeding"),
          ("76866169", "small polyp"),
          ("5af764dc", "flat polyp")]
N = 300


def clocks():
    o = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=5).stdout.strip()
    return " / ".join(x.strip() for x in o.split(","))


setup_measurement_env(tf32_matmul=True)
model = load_pmfnet(device="cuda")
infer = GraphInfer(model)

items = []
for vid, lab in VIDEOS:
    p = next((ROOT / "hyper-kvasir-videos" / "videos").glob(vid + "*.avi"))
    cap = cv2.VideoCapture(str(p))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    pre = GpuPreprocess(h, w)
    wr = cv2.VideoWriter(str(SCRATCH / f"_pipe_{vid}.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert wr.isOpened()
    items.append((vid, lab, p, pre, wr))

for vid, lab, p, pre, wr in items:
    run_pipelined(p, pre, infer, 60, True, wr)

print(f"\n{'='*70}\nTHROUGHPUT pipelining ({N} frame/video, 2 vong xen ke)\n{'='*70}")
print(f"{'vong':6} {'clock/C/W':>22}  " + "  ".join(f"{v[:8]:>12}" for v, *_ in items))
print("-" * 70)
res = {v: [] for v, *_ in items}
for rd in range(2):
    c = clocks()
    line = []
    for vid, lab, p, pre, wr in items:
        for has_write in (True,):
            k, secs = run_pipelined(p, pre, infer, N, True, wr)
            f = k / secs
            res[vid].append(f)
            line.append(f"{f:10.1f} FPS")
    print(f"{rd+1:<6} {c:>22}  " + "  ".join(line))

print(f"\n{'='*70}\nKHONG ghi VIDEO (bo encode)\n{'='*70}")
nw = {}
for vid, lab, p, pre, wr in items:
    k, secs = run_pipelined(p, pre, infer, N, True, None)
    nw[vid] = k / secs
    print(f"  {vid:10} {lab:26} {k/secs:8.1f} FPS")

print(f"\n{'='*70}\nTONG hop\n{'='*70}")
print(f"{'video':10} {'nhan':26} {'co ghi':>10} {'khong ghi':>11}")
print("-" * 70)
for vid, lab, *_ in items:
    m = sum(res[vid]) / len(res[vid])
    print(f"{vid:10} {lab:26} {m:8.1f} FPS {nw[vid]:8.1f} FPS")
print(f"\nclock cuoi: {clocks()}")
for *_, wr in items:
    wr.release()
