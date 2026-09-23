"""Chay model tren nhieu video khac nhau, thong ke dien tich mask + xuat anh kiem chung."""
import sys, time
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bench_utils import load_pmfnet, setup_measurement_env
from video_infer import GpuPreprocess, GraphInfer, postprocess, overlay

VIDEOS = [
    ("0220d11b-ab12-4b02-93ce-5d7c205c7043", "polyp bleeding",      "ca de, da do"),
    ("76866169-2d53-4a09-a1f5-2cb2b2a89b23", "small polyp",         "ca kho: polyp nho"),
    ("5af764dc-1c6a-4b01-9e50-a8d3a7d0e4e8", "flat polyp",          "ca kho nhat: polyp det"),
    ("164c76bd-9b62-4b9e-a2a4-2e4a5f7e1a3f", "cecum ileocecal valve", "doi chung am: khong polyp"),
]

N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 400

setup_measurement_env(tf32_matmul=True)
model = load_pmfnet(device="cuda")
infer = GraphInfer(model)

rows = []
for vid, finding, note in VIDEOS:
    matches = list((ROOT / "hyper-kvasir-videos" / "videos").glob(vid[:8] + "*.avi"))
    if not matches:
        print(f"[bo qua] khong tim thay {vid[:8]}")
        continue
    path = matches[0]
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[bo qua] khong mo duoc {path.name}")
        continue
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pre = GpuPreprocess(h, w)

    print(f"\n=== {path.name[:8]}  [{finding}]  {note}")
    print(f"    {w}x{h}, {total} frame trong file, chay {N_FRAMES} frame dau")

    areas, best = [], []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(N_FRAMES):
            ok, f = cap.read()
            if not ok:
                break
            g = pre.upload(f)
            m = postprocess(infer(pre(g)), (h, w))
            a = m.float().mean().item()
            areas.append(a)
            if len(best) < 3 or a > best[-1][0]:
                best.append((a, i, f.copy(), overlay(g, m).cpu().numpy()))
                best.sort(key=lambda t: -t[0])
                best[:] = best[:3]
    cap.release()
    secs = time.perf_counter() - t0
    a = np.array(areas)

    print(f"    dien tich mask: mean {a.mean()*100:6.2f}%   p50 {np.percentile(a,50)*100:6.2f}%"
          f"   max {a.max()*100:6.2f}%")
    print(f"    frame co mask >1%  : {(a>0.01).sum():4d}/{len(a)}  ({(a>0.01).mean()*100:.1f}%)")
    print(f"    frame co mask >0.1%: {(a>0.001).sum():4d}/{len(a)}  ({(a>0.001).mean()*100:.1f}%)")
    print(f"    frame gan nhu trong: {(a<0.0001).sum():4d}/{len(a)}")
    print(f"    ({len(a)} frame / {secs:.1f}s = {len(a)/secs:.1f} FPS, co ca tinh dien tich)")

    out = np.vstack([np.hstack([o, v]) for _, _, o, v in best])
    dst = ROOT / "benchmarks" / f"_check_{path.name[:8]}.png"
    cv2.imwrite(str(dst), out)
    print(f"    -> {dst.name}  (trai=goc, phai=overlay; 3 frame mask lon nhat)")

    rows.append((path.name[:8], finding, w, h, a))

print(f"\n{'='*86}\nTONG hop\n{'='*86}")
print(f"{'video':10} {'nhan':24} {'kich thuoc':11} {'mean':>7} {'p50':>7} {'max':>7} {'>1%':>8}")
print("-" * 86)
for vid, finding, w, h, a in rows:
    print(f"{vid:10} {finding[:24]:24} {w}x{h:<6} {a.mean()*100:6.2f}% "
          f"{np.percentile(a,50)*100:6.2f}% {a.max()*100:6.2f}% {(a>0.01).mean()*100:7.1f}%")
