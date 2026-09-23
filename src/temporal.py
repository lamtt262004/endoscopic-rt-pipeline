"""
temporal.py — Ngay 7. Loc thoi gian cho mask, va do danh doi cua no.

    python src/temporal.py                  # quet tham so, in duong cong danh doi
    python src/temporal.py --n 500
    python src/temporal.py --alpha 0.35     # xem ky mot cau hinh
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import load_pmfnet, save_result, setup_measurement_env
from video_infer import GpuPreprocess, GraphInfer, TRTInfer

ROOT = Path(__file__).resolve().parent.parent
SIZE = 256
FPS = 25.0
AREA_THR = 0.01

VIDEOS = {
    "polyp":   ("76866169", "small polyp"),
    "nopolyp": ("164c76bd", "cecum ileocecal valve — khong co polyp"),
}


class TemporalFilter:
    def __init__(self, alpha_up=0.7, alpha_dn=0.12, device="cuda"):
        self.au, self.ad = alpha_up, alpha_dn
        self.ema = None

    def reset(self):
        self.ema = None

    def __call__(self, prob):
        if self.ema is None:
            self.ema = prob.clone()
            return self.ema
        a = torch.where(prob > self.ema, self.au, self.ad)
        self.ema = self.ema + a * (prob - self.ema)
        return self.ema


def filter_numpy(seq, au, ad):
    out = np.empty_like(seq)
    ema = seq[0].copy()
    out[0] = ema
    for i in range(1, len(seq)):
        p = seq[i]
        a = np.where(p > ema, au, ad).astype(np.float32)
        ema = ema + a * (p - ema)
        out[i] = ema
    return out


def episodes(flags):
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def compare(raw_flags, filt_flags):
    raw_ep = episodes(raw_flags)
    caught, delays, missed = 0, [], 0
    for s, e in raw_ep:
        hit = np.where(filt_flags[s:e + 1])[0]
        if len(hit):
            caught += 1
            delays.append(int(hit[0]))
        else:
            missed += 1
    return dict(n_ep=len(raw_ep), caught=caught, missed=missed,
                miss_pct=100 * missed / max(len(raw_ep), 1),
                delay_mean=float(np.mean(delays)) if delays else 0.0,
                delay_max=float(np.max(delays)) if delays else 0.0)


@torch.inference_mode()
def collect(key, infer, n):
    vid, label = VIDEOS[key]
    path = next((ROOT / "hyper-kvasir-videos" / "videos").glob(vid + "*.avi"))
    cap = cv2.VideoCapture(str(path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pre = GpuPreprocess(h, w)
    print(f"  {key:8} {path.name[:8]}  {label}")
    seq = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        seq.append(infer(pre(pre.upload(f)))[0, 0].half().cpu().numpy())
    cap.release()
    a = np.stack(seq).astype(np.float32)
    print(f"           {len(a)} frame, {a.nbytes/1e6:.0f} MB")
    return a


def area_flags(seq, thr=AREA_THR):
    return (seq > 0.5).reshape(len(seq), -1).mean(1) > thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--trt", default="fp16")
    ap.add_argument("--alpha", type=float, default=None,
                    help="xem ky mot alpha_dn (alpha_up co dinh 0.7)")
    args = ap.parse_args()

    setup_measurement_env(tf32_matmul=True)
    ep = ROOT / "engines" / f"pmfnet_trt_{args.trt}.engine"
    if ep.exists():
        infer = TRTInfer(ep)
        backend = f"TensorRT {args.trt.upper()}"
    else:
        infer = GraphInfer(load_pmfnet(device="cuda"))
        backend = "PyTorch + CUDA Graph"

    print(f"{'='*84}\n1. chay model mot lan, luu chuoi xac suat  ({backend})\n{'='*84}")
    seqs = {k: collect(k, infer, args.n) for k in VIDEOS}
    raw = {k: area_flags(v) for k, v in seqs.items()}

    print(f"\n{'='*84}\n2. truoc khi loc\n{'='*84}")
    for k in VIDEOS:
        e = episodes(raw[k])
        lens = [b - a + 1 for a, b in e]
        print(f"  {k:8} {raw[k].mean()*100:5.1f}% frame co phat hien | "
              f"{len(e):3d} dot | dai dot: p50 {np.percentile(lens,50):.0f} "
              f"max {max(lens)} frame | dot <=4 frame: {sum(1 for l in lens if l<=4)}")
    print("\n  dot ngan la thu bo loc se giet: rui ro o video co polyp, loi ich o video khong.")

    base_pct = raw["nopolyp"].mean() * 100

    def sweep(name, pairs, note):
        print(f"\n{'='*84}\n{name}\n{'='*84}")
        print(f"  {note}")
        print(f"  {'alpha':>17} | {'khong polyp (loi ich)':^22} | {'co polyp (gia phai tra)':^36}")
        print(f"  {'up':>8}{'dn':>9} | {'%frame':>9}{'thay doi':>12} | "
              f"{'dot bo sot':>13}{'tre (frame)':>13}{'tre (ms)':>10}")
        print("  " + "-" * 82)
        out = []
        for au, ad in pairs:
            f_fl = {k: area_flags(filter_numpy(seqs[k], au, ad)) for k in VIDEOS}
            pos = compare(raw["polyp"], f_fl["polyp"])
            pct = f_fl["nopolyp"].mean() * 100
            out.append(dict(alpha_up=au, alpha_dn=ad, fp_frame_pct=pct,
                            fp_change_pct=pct / base_pct * 100 - 100,
                            tp_missed=pos["missed"], tp_total=pos["n_ep"],
                            tp_miss_pct=pos["miss_pct"],
                            delay_frames=pos["delay_mean"],
                            delay_ms=pos["delay_mean"] / FPS * 1000))
            print(f"  {au:8.2f}{ad:9.2f} | {pct:8.1f}%{pct/base_pct*100-100:+11.0f}% | "
                  f"{pos['missed']:6d}/{pos['n_ep']:<6d}{pos['delay_mean']:12.2f}"
                  f"{pos['delay_mean']/FPS*1000:10.0f}")
        return out

    rows_dn = sweep(
        "3a. quet alpha_dn — xuong cham (giu lai sau khi tin hieu mat)",
        [(0.7, ad) for ad in (1.0, 0.6, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08)],
        "Y dinh: chong nhap nhay. Du doan cua toi: se giam duong tinh gia.")

    print("\n  du doan sai: 'xuong cham' keo dai moi phat hien, ke ca cai sai.")
    print(f"    Toi da gop hai muc tieu khac nhau lam mot.")

    rows_up = sweep(
        "3b. quet alpha_up — cham bat (nut that su diet duong tinh gia)",
        [(au, 0.35) for au in (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.10, 0.06)],
        "Cham bat -> dot ngan khong kip vuot nguong. Gia phai tra: do tre.")

    print(f"\n{'='*84}\n4. chon diem van hanh\n{'='*84}")
    rows = rows_dn + rows_up
    ok = [r for r in rows_up if r["tp_miss_pct"] <= 5 and r["fp_change_pct"] < -5]
    if ok:
        best = min(ok, key=lambda r: r["fp_frame_pct"])
        print(f"  Rang buoc: bo sot khong qua 5% dot that.")
        print(f"  -> alpha_up = {best['alpha_up']}, alpha_dn = {best['alpha_dn']}")
        print(f"     duong tinh gia : {base_pct:.1f}% -> {best['fp_frame_pct']:.1f}% frame "
              f"({best['fp_change_pct']:+.0f}%)")
        print(f"     bo sot         : {best['tp_missed']}/{best['tp_total']} dot that")
        print(f"     tre            : {best['delay_frames']:.1f} frame = "
              f"{best['delay_ms']:.0f} ms")
        print(f"\n  Phat bieu dung cach: 'giam {abs(best['fp_change_pct']):.0f}% duong tinh gia, "
              f"tra gia {best['delay_ms']:.0f} ms tre'.")
    else:
        print("khong cau hinh nao vua giam duong tinh gia >5% vua giu bo sot <=5%.")
        print(f"     mot tham so cho co ve da lam gi do.")
    print("\n  Bo loc chi giet duoc nhap nhay; duong tinh gia keo dai phai sua du lieu huan luyen.")

    save_result({"config": "temporal_filter_sweep", "stage": "postprocess",
                 "backend": backend, "n_frames": args.n, "area_thr": AREA_THR,
                 "sweep_alpha_dn": rows_dn, "sweep_alpha_up": rows_up,
                 "baseline_fp_frame_pct": float(base_pct)})


if __name__ == "__main__":
    main()
