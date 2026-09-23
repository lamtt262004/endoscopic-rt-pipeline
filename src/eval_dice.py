"""
eval_dice.py — Dice/IoU tren tap anh co mask, theo dung protocol da chot.

    python src/eval_dice.py                 # 200 anh, cau hinh mac dinh
    python src/eval_dice.py --n 1000        # toan bo
    python src/eval_dice.py --compare-tf32  # do lech numerics khi bat tf32 matmul
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_utils import load_pmfnet, save_result

_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = _ROOT / "Kvasir_images" / "segmented-images" / "images"
MSK_DIR = _ROOT / "Kvasir_images" / "segmented-images" / "masks"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(path, size=256, device="cuda"):
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float().div_(255)
    return ((t - MEAN) / STD).to(device), bgr.shape[:2]


def load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (m > 127).astype(np.uint8)


def prob_to_mask(prob, hw):
    p = F.interpolate(prob, size=hw, mode="bilinear", align_corners=False)
    return (p > 0.5)[0, 0]


def dice_iou(pred, gt):
    p = pred.flatten().float()
    g = torch.as_tensor(gt, device=pred.device).flatten().float()
    inter = (p * g).sum()
    sp, sg = p.sum(), g.sum()
    if sp + sg == 0:
        return 1.0, 1.0
    dice = (2 * inter / (sp + sg)).item()
    union = sp + sg - inter
    return dice, (inter / union).item() if union > 0 else 1.0


@torch.inference_mode()
def evaluate(model, files, device="cuda", collect_probs=False):
    dices, ious, probs = [], [], []
    for f in files:
        x, hw = preprocess(f, device=device)
        prob = model(x)
        if collect_probs:
            probs.append(prob.clone())
        gt = load_mask(MSK_DIR / f.name)
        d, i = dice_iou(prob_to_mask(prob, hw), gt)
        dices.append(d)
        ious.append(i)
    return np.array(dices), np.array(ious), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--compare-tf32", action="store_true",
                    help="do lech numerics giua matmul tf32 off/on")
    args = ap.parse_args()

    files = sorted(IMG_DIR.glob("*.jpg"))[:args.n]
    print(f"Danh gia tren {len(files)} anh  |  protocol: resize xac suat ve kich "
          f"thuoc goc -> threshold 0.5 -> macro Dice")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_pmfnet(device="cuda")

    if not args.compare_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        t0 = time.perf_counter()
        d, i, _ = evaluate(model, files)
        print(f"\n  Dice = {d.mean():.4f} (std {d.std():.4f})   "
              f"IoU = {i.mean():.4f}   [{time.perf_counter()-t0:.1f}s]")
        print(f"  Kiem tra dong nhat thuc: 2*IoU/(1+IoU) = "
              f"{2*i.mean()/(1+i.mean()):.4f} >= Dice (bat dang thuc Jensen) — "
              f"{'khop' if 2*i.mean()/(1+i.mean()) >= d.mean() else 'bat thuong'}")
        print(f"  Anh te nhat: Dice = {d.min():.4f} ({files[d.argmin()].name})")
        save_result({"config": "pytorch_tf32matmul_off", "stage": "accuracy",
                     "n_images": len(files), "dice": round(float(d.mean()), 4),
                     "iou": round(float(i.mean()), 4)})
        return

    out = {}
    for tf32 in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = tf32
        d, i, probs = evaluate(model, files, collect_probs=True)
        out[tf32] = (d, i, probs)
        print(f"\n  matmul tf32 {'on ' if tf32 else 'off'}: "
              f"Dice = {d.mean():.4f}   IoU = {i.mean():.4f}")

    d_off, i_off, p_off = out[False]
    d_on, i_on, p_on = out[True]

    diffs = [(a - b).abs().max().item() for a, b in zip(p_off, p_on)]
    flips = [((a > 0.5) != (b > 0.5)).float().mean().item() for a, b in zip(p_off, p_on)]

    print("\n" + "=" * 68)
    print("lech numerics do tf32 (10-bit mantissa vs 23-bit cua fp32)")
    print("=" * 68)
    print(f"  max |delta xac suat|        : {max(diffs):.2e}  (trung binh {np.mean(diffs):.2e})")
    print(f"  ti le pixel doi nhan @0.5   : {np.mean(flips)*100:.4f} %  "
          f"(toi da {max(flips)*100:.4f} %)")
    print(f"  Dice                        : {d_off.mean():.4f} -> {d_on.mean():.4f}  "
          f"(delta {d_on.mean()-d_off.mean():+.5f})")
    print(f"  IoU                         : {i_off.mean():.4f} -> {i_on.mean():.4f}  "
          f"(delta {i_on.mean()-i_off.mean():+.5f})")
    print(f"  so anh Dice tut > 0.005     : {(d_off - d_on > 0.005).sum()} / {len(files)}")

    print()
    if abs(d_on.mean() - d_off.mean()) < 0.001:
        print("  => Lech duoi 0.001 Dice — khong dang ke. Bat tf32 matmul duoc.")
    elif d_on.mean() < d_off.mean() - 0.003:
        print("=> Dice tut that su. Can can nhac doi toc do lay do chinh xac.")
    else:
        print("=> Lech nho nhung dang theo doi. Bat tf32 thi cot benchmark")
        print("     khong duoc ghi la 'fp32' nua.")

    save_result({"config": "tf32_matmul_accuracy_ablation", "stage": "accuracy",
                 "n_images": len(files),
                 "dice_off": round(float(d_off.mean()), 5),
                 "dice_on": round(float(d_on.mean()), 5),
                 "iou_off": round(float(i_off.mean()), 5),
                 "iou_on": round(float(i_on.mean()), 5),
                 "max_prob_delta": float(max(diffs)),
                 "pixel_flip_pct": float(np.mean(flips) * 100)})


if __name__ == "__main__":
    main()
