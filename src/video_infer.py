"""
video_infer.py — Pipeline video hoan chinh + do breakdown tung khau.

    python src/video_infer.py --verify              # kiem tra preprocess GPU == cv2 (lam truoc)
    python src/video_infer.py --list polyp          # xem co nhung video nao
    python src/video_infer.py --video 76866169 --n 400     # chay tren 1 video cu the
    python src/video_infer.py --video "flat polyp" --n 200 # ... hoac go theo nhan
    python src/video_infer.py --n 600 --no-write    # bo encode khoi vong do
    python src/video_infer.py --demo out.mp4        # xuat video demo co overlay
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import (load_pmfnet, query_gpu_state, save_result,
                         setup_measurement_env, summarize)

_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = _ROOT / "hyper-kvasir-videos" / "videos"
SIZE = 256
BUDGET_MS = 16.7


class GpuPreprocess:
    def __init__(self, h, w, device="cuda"):
        self.pinned = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.device = device

    def upload(self, frame_bgr):
        self.pinned.copy_(torch.from_numpy(frame_bgr))
        return self.pinned.to(self.device, non_blocking=True)

    def __call__(self, gpu_hwc_bgr):
        t = gpu_hwc_bgr.permute(2, 0, 1)[None]
        t = t.float().div_(255)
        t = F.interpolate(t, (SIZE, SIZE), mode="bilinear",
                          align_corners=False, antialias=False)
        t = t[:, [2, 1, 0]]
        return (t - self.mean) / self.std


def postprocess(prob, hw):
    p = F.interpolate(prob, size=hw, mode="bilinear", align_corners=False)
    return p[0, 0] > 0.5


def overlay(gpu_hwc_bgr, mask, alpha=0.4):
    m = mask.float().unsqueeze(-1)
    color = torch.tensor([0., 0., 255.], device=gpu_hwc_bgr.device)
    out = gpu_hwc_bgr.float() * (1 - alpha * m) + color * (alpha * m)
    return out.to(torch.uint8)


class GraphInfer:
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


class TRTInfer:
    def __init__(self, engine_path, device="cuda"):
        from trt_utils import TRTRunner
        self.runner = TRTRunner(engine_path, device=device)

    def __call__(self, x):
        return self.runner(x)


@torch.inference_mode()
def verify_preprocess(model, n=100):
    from eval_dice import IMG_DIR, MSK_DIR, dice_iou, load_mask, preprocess as cv2_pre

    files = sorted(IMG_DIR.glob("*.jpg"))[:n]
    print(f"\n{'='*72}\nKIEM chung preprocess: GPU vs cv2 tren {len(files)} anh Kvasir\n{'='*72}")

    d_cv2, d_gpu, d_pair, flips, diffs = [], [], [], [], []
    pre = None
    for f in files:
        bgr = cv2.imread(str(f))
        h, w = bgr.shape[:2]
        if pre is None or pre.pinned.shape[:2] != (h, w):
            pre = GpuPreprocess(h, w)
        p_gpu = model(pre(pre.upload(bgr)))
        p_cv2 = model(cv2_pre(f)[0])

        diffs.append((p_gpu - p_cv2).abs().max().item())
        flips.append(((p_gpu > 0.5) != (p_cv2 > 0.5)).float().mean().item())
        gt = load_mask(MSK_DIR / f.name)
        m_gpu, m_cv2 = postprocess(p_gpu, (h, w)), postprocess(p_cv2, (h, w))
        d_gpu.append(dice_iou(m_gpu, gt)[0])
        d_cv2.append(dice_iou(m_cv2, gt)[0])
        d_pair.append(dice_iou(m_gpu, m_cv2.cpu().numpy())[0])

    d_gpu, d_cv2, d_pair = map(np.array, (d_gpu, d_cv2, d_pair))
    print(f"  Dice cv2 (mốc cũ)        : {d_cv2.mean():.4f}")
    print(f"  Dice GPU (đường mới)     : {d_gpu.mean():.4f}   (Δ {d_gpu.mean()-d_cv2.mean():+.5f})")
    print(f"  Dice GPU vs cv2          : {d_pair.mean():.4f}   <- 1.0000 la trung khit")
    print(f"  pixel doi nhan @0.5      : {np.mean(flips)*100:.4f} %")
    print(f"  max |delta xac suat|     : {max(diffs):.2e}   (chi tham khao)")
    ok = abs(d_gpu.mean() - d_cv2.mean()) < 0.001
    print(f"\n  => {'dat — dung duoc duong GPU' if ok else 'lech qua nhieu — xem lai'}")
    save_result({"config": "preprocess_gpu_vs_cv2", "stage": "accuracy",
                 "n_images": len(files), "dice_cv2": round(float(d_cv2.mean()), 5),
                 "dice_gpu": round(float(d_gpu.mean()), 5),
                 "dice_pair": round(float(d_pair.mean()), 5),
                 "pixel_flip_pct": float(np.mean(flips) * 100)})
    return ok


@torch.inference_mode()
def run_serial(cap, pre, infer, n, writer=None, warmup=50, path=None, hw=True):
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_pin = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)

    def one_frame(measure=False):
        t = {}
        def tick():
            if measure:
                torch.cuda.synchronize()
                return time.perf_counter()
            return 0.0

        t0 = tick()
        ok, frame = cap.read()
        if not ok:
            return None
        t1 = tick()
        g_bgr = pre.upload(frame)
        t2 = tick()
        x = pre(g_bgr)
        t3 = tick()
        prob = infer(x)
        t4 = tick()
        mask = postprocess(prob, (h, w))
        t5 = tick()
        vis = overlay(g_bgr, mask)
        t6 = tick()
        out_pin.copy_(vis, non_blocking=False)
        t7 = tick()
        if writer is not None:
            writer.write(out_pin.numpy())
        t8 = tick()

        if measure:
            t = {"decode": t1-t0, "h2d": t2-t1, "preprocess": t3-t2, "infer": t4-t3,
                 "postprocess": t5-t4, "overlay": t6-t5, "d2h": t7-t6, "encode": t8-t7}
            t = {k: v*1000 for k, v in t.items()}
        return t

    for _ in range(warmup):
        if one_frame() is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    stages = {k: [] for k in ("decode", "h2d", "preprocess", "infer",
                              "postprocess", "overlay", "d2h", "encode")}
    for _ in range(n):
        r = one_frame(measure=True)
        if r is None:
            break
        for k, v in r.items():
            stages[k].append(v)

    if path is not None:
        cap.release()
        cap = _open(path, hw)
        for _ in range(warmup):
            cap.grab()

    e2e = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if one_frame() is None:
            break
        torch.cuda.synchronize()
        e2e.append((time.perf_counter() - t0) * 1000)

    return stages, e2e


def run_pipelined(path, pre, infer, n, hw_accel, writer=None, qsize=8):
    h, w = pre.pinned.shape[:2]
    q_in = queue.Queue(maxsize=qsize)
    q_out = queue.Queue(maxsize=qsize) if writer else None
    stop = threading.Event()

    def decoder():
        cap = _open(path, hw_accel)
        k = 0
        while k < n and not stop.is_set():
            ok, f = cap.read()
            if not ok:
                cap.release(); cap = _open(path, hw_accel); continue
            q_in.put(f); k += 1
        q_in.put(None); cap.release()

    def encoder():
        while True:
            item = q_out.get()
            if item is None:
                break
            writer.write(item)

    th = [threading.Thread(target=decoder, daemon=True)]
    if writer:
        th.append(threading.Thread(target=encoder, daemon=True))
    out_pin = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in th:
        t.start()
    k = 0
    with torch.inference_mode():
        while True:
            frame = q_in.get()
            if frame is None:
                break
            g_bgr = pre.upload(frame)
            prob = infer(pre(g_bgr))
            vis = overlay(g_bgr, postprocess(prob, (h, w)))
            if writer:
                out_pin.copy_(vis)
                q_out.put(out_pin.numpy().copy())
            else:
                torch.cuda.current_stream().synchronize()
            k += 1
    if writer:
        q_out.put(None)
    for t in th:
        t.join(timeout=5)
    torch.cuda.synchronize()
    return k, (time.perf_counter() - t0)


@torch.inference_mode()
def run_live(path, pre, infer, n, hw_accel, writer=None, src_fps=25.0, qsize=8,
             warmup=40):
    h, w = pre.pinned.shape[:2]
    q_in = queue.Queue(maxsize=qsize)
    q_out = queue.Queue(maxsize=qsize) if writer else None
    period = 1.0 / src_fps
    qdepth = []

    def decoder(t0):
        cap = _open(path, hw_accel)
        for k in range(n):
            due = t0 + k * period
            gap = due - time.perf_counter()
            if gap > 0:
                time.sleep(gap)
            ok, f = cap.read()
            if not ok:
                cap.release()
                cap = _open(path, hw_accel)
                ok, f = cap.read()
            qdepth.append(q_in.qsize())
            q_in.put((due, f))
        q_in.put(None)
        cap.release()

    def encoder():
        while True:
            item = q_out.get()
            if item is None:
                break
            writer.write(item)

    out_pin = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)

    cap_w = _open(path, hw_accel)
    for _ in range(warmup):
        ok, f = cap_w.read()
        if not ok:
            break
        g = pre.upload(f)
        overlay(g, postprocess(infer(pre(g)), (h, w)))
    cap_w.release()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    th = [threading.Thread(target=decoder, args=(t0,), daemon=True)]
    if writer:
        th.append(threading.Thread(target=encoder, daemon=True))
    for t in th:
        t.start()

    lat, k = [], 0
    while True:
        item = q_in.get()
        if item is None:
            break
        due, frame = item
        g_bgr = pre.upload(frame)
        vis = overlay(g_bgr, postprocess(infer(pre(g_bgr)), (h, w)))
        if writer:
            out_pin.copy_(vis)
            q_out.put(out_pin.numpy().copy())
        else:
            torch.cuda.current_stream().synchronize()
        lat.append((time.perf_counter() - due) * 1000)
        k += 1
    if writer:
        q_out.put(None)
    for t in th:
        t.join(timeout=5)
    torch.cuda.synchronize()
    return np.array(lat), (time.perf_counter() - t0), k, qdepth


def _open(path, hw_accel):
    if hw_accel:
        return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                                [cv2.CAP_PROP_HW_ACCELERATION,
                                 cv2.VIDEO_ACCELERATION_ANY])
    return cv2.VideoCapture(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--video", default=None,
                    help="duong dan, hoac 8 ky tu dau cua id, hoac tu khoa nhan "
                         "('flat polyp'). Bo trong = video dau tien.")
    ap.add_argument("--list", nargs="?", const="", default=None, metavar="tukhoa",
                    help="liet ke video khop tu khoa roi thoat")
    ap.add_argument("--trt", nargs="?", const="fp16", default=None,
                    metavar="PRECISION",
                    help="dung engine TensorRT thay cho PyTorch+CUDA Graph: "
                         "fp16 (mac dinh) | tf32 | fp32 | duong dan .engine")
    ap.add_argument("--eager", action="store_true",
                    help="PyTorch eager, khong CUDA Graph — de lay moc baseline")
    ap.add_argument("--live", default=None, metavar="FPS_LIST",
                    help="do o nhip nguon that thay vi doc file toi da, "
                         "vd --live 25,30,60 . Chi chay muc C.")
    ap.add_argument("--no-write", action="store_true", help="bo encode khoi vong do")
    ap.add_argument("--no-hw", action="store_true", help="tat hardware decode")
    ap.add_argument("--verify", action="store_true", help="chi kiem chung preprocess")
    ap.add_argument("--demo", default=None, help="xuat video demo ra duong dan nay")
    args = ap.parse_args()

    if args.list is not None:
        from videos import find, _print_table
        rows = find(args.list)
        if not rows:
            print(f"Khong co video nao khop '{args.list}'.")
        else:
            _print_table(rows, f"Tim '{args.list}'" if args.list else "Toan bo video")
        return

    setup_measurement_env(tf32_matmul=True)
    model = load_pmfnet(device="cuda")

    if args.verify:
        verify_preprocess(model)
        return

    from videos import resolve, label_for
    path = resolve(args.video)
    hw = not args.no_hw
    cap = _open(path, hw)
    assert cap.isOpened(), f"khong mo duoc {path}"
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fcc >> 8*i) & 0xFF) for i in range(4))
    print(f"\nVideo : {path.name}")
    print(f"        nhan  : {label_for(path)}")
    print(f"        {w}x{h} @ {fps:.0f} fps | codec {codec} | backend {cap.getBackendName()}")
    print(f"        {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frame trong file")
    print(f"        hardware decode: {'bat' if hw else 'tat'}  "
          f"(accel thuc te = {cap.get(cv2.CAP_PROP_HW_ACCELERATION):.0f})")

    pre = GpuPreprocess(h, w)
    if args.trt:
        ep = Path(args.trt) if args.trt.endswith(".engine") \
            else _ROOT / "engines" / f"pmfnet_trt_{args.trt}.engine"
        assert ep.exists(), f"chua co {ep} — chay src/build_engine.py truoc"
        infer = TRTInfer(ep)
        backend = f"TensorRT {args.trt.upper()}"
        del model
        torch.cuda.empty_cache()
    elif args.eager:
        infer = lambda x: model(x)
        backend = "PyTorch eager (tf32 matmul)"
    else:
        infer = GraphInfer(model)
        backend = "PyTorch + CUDA Graph (tf32 matmul)"
    print(f"        backend: {backend}")

    writer = None
    if not args.no_write or args.demo:
        out_path = args.demo or str(_ROOT / "benchmarks" / "_video_out.mp4")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
        assert writer.isOpened(), "VideoWriter khong mo duoc — thieu codec?"
        print(f"        ghi ra: {out_path} (mp4v)")

    if args.live:
        rates = [float(r) for r in args.live.split(",")]
        print(f"\n{'='*78}\nC. NHIP NGUON THAT — frame toi deu dan, khong doc file toi da\n{'='*78}")
        print(f"  {'nguon':>8} {'ngan sach':>10} | {'latency p50':>12}{'p99':>9}"
              f"{'miss':>8} | {'FPS ra':>8} {'q max':>6}  ket luan")
        print("  " + "-" * 76)
        rows = []
        for r in rates:
            lat, secs, k, qd = run_live(path, pre, infer, args.n, hw, writer, r)
            budget = 1000.0 / r
            p50, p99 = np.percentile(lat, 50), np.percentile(lat, 99)
            miss = (lat > budget).mean() * 100
            drift = np.mean(lat[len(lat)//2:]) - np.mean(lat[:len(lat)//2])
            ok = drift < budget * 0.5
            rows.append(dict(src_fps=r, budget_ms=budget, p50=float(p50), p99=float(p99),
                             miss_pct=float(miss), out_fps=k/secs, qmax=max(qd) if qd else 0,
                             drift_ms=float(drift), keeps_up=bool(ok)))
            print(f"  {r:7.0f}f {budget:9.1f}ms | {p50:11.2f}{p99:9.2f}{miss:7.1f}% |"
                  f" {k/secs:7.1f} {max(qd) if qd else 0:6d}  "
                  + ("theo kip" if ok else f"tut lai (+{drift:.0f} ms troi)"))
        print(f"\n  'q max' = so frame ket o hang doi. Bo khong thi pipeline dang thua suc.")
        print(f"  'troi'  = latency nua sau tru nua dau. Tang dan = khong theo kip.")
        if writer:
            writer.release()
        save_result({"config": "live_feed_rate", "stage": "end_to_end",
                     "video": path.name, "backend": backend, "n_frames": args.n,
                     "write_video": writer is not None, "rates": rows,
                     "env_after": query_gpu_state()})
        return

    print(f"\n{'='*72}\nA. LATENCY — mot frame di het 8 khau, noi tiep\n"
          f"   (breakdown tung khau co sync + do lien mach khong sync)\n{'='*72}")
    stages, e2e = run_serial(cap, pre, infer, args.n, writer, path=path, hw=hw)
    cap.release()

    print(f"  {'khau':14} {'mean':>8} {'p50':>8} {'p99':>8}   {'% tong':>7}")
    print("  " + "-" * 52)
    tot = sum(np.mean(v) for v in stages.values() if v)
    for k, v in stages.items():
        if not v:
            continue
        a = np.array(v)
        print(f"  {k:14} {a.mean():8.3f} {np.percentile(a,50):8.3f} "
              f"{np.percentile(a,99):8.3f}   {a.mean()/tot*100:6.1f}%")
    print("  " + "-" * 52)
    print(f"  {'tong cac khau':14} {tot:8.3f}  <- do rieng, da pha overlap")
    st = summarize(e2e, BUDGET_MS)
    print(f"  {'LATENCY':14} {st.mean_ms:8.3f} {st.p50_ms:8.3f} {st.p99_ms:8.3f}"
          f"  <- lien mach, trung thuc")
    print(f"\n  {st}")
    print(f"  FPS suy ra tu latency = {1000/st.mean_ms:.1f}  "
          f"<- khong phai throughput, xem muc B")
    tot50 = sum(float(np.percentile(v, 50)) for v in stages.values() if v)
    d = (tot50 - st.p50_ms) / st.p50_ms * 100
    print(f"  {'tong khau (p50)':14} {tot50:8.3f}   ({d:+.1f}% so voi latency p50)")

    print(f"\n{'='*72}\nB. THROUGHPUT — decode / infer / encode tren 3 luong\n{'='*72}")
    k, secs = run_pipelined(path, pre, infer, args.n, hw, writer)
    fps_pipe = k / secs
    print(f"  {k} frame trong {secs:.2f} s  ->  throughput = {fps_pipe:.1f} FPS")
    print(f"  so voi FPS suy ra tu latency ({1000/st.mean_ms:.1f})  ->  "
          f"{fps_pipe/(1000/st.mean_ms):.2f}x")
    print("\n  pipelining khong giam latency (mot frame van qua du cac khau),")
    print("        chi tang throughput. Hai con so phai bao cao rieng.")
    if writer:
        writer.release()

    save_result({
        "config": "video_pipeline", "stage": "end_to_end",
        "video": path.name, "video_label": label_for(path),
        "backend": backend,
        "resolution": f"{w}x{h}", "codec": codec,
        "hw_decode": hw, "write_video": writer is not None,
        "stages_mean_ms": {k: round(float(np.mean(v)), 3) for k, v in stages.items() if v},
        "sum_stages_ms": round(float(tot), 3),
        **{k: v for k, v in vars(st).items()},
        "fps_serial": round(1000/st.mean_ms, 1),
        "fps_pipelined": round(fps_pipe, 1),
        "env_after": query_gpu_state(),
    })


if __name__ == "__main__":
    main()
