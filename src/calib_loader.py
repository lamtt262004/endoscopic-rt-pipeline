"""
calib_loader.py — Ngay 5. Nap du lieu calibration cho int8.

    python src/calib_loader.py            # tu kiem tra: nap thu, in thong ke activation
"""
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / "engines"
N_CALIB_DEFAULT = 300


def calib_files(n=N_CALIB_DEFAULT):
    from eval_dice import IMG_DIR
    files = sorted(IMG_DIR.glob("*.jpg"))
    assert len(files) >= n + 100, f"chi co {len(files)} anh, can it nhat {n+100}"
    return files[:n]


def eval_files(skip=N_CALIB_DEFAULT, n=200):
    from eval_dice import IMG_DIR
    return sorted(IMG_DIR.glob("*.jpg"))[skip:skip + n]


def make_calibrator(algo, files, cache_path, input_name="input"):
    base = {"entropy2": trt.IInt8EntropyCalibrator2,
            "minmax": trt.IInt8MinMaxCalibrator,
            "entropy": trt.IInt8EntropyCalibrator}[algo]

    class _Calib(base):
        def __init__(self):
            base.__init__(self)
            self.files = list(files)
            self.i = 0
            self.cache = Path(cache_path)
            self.input_name = input_name
            self.buf = torch.empty(1, 3, 256, 256, device="cuda", dtype=torch.float32)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(self.files):
                return None
            from eval_dice import preprocess
            x, _ = preprocess(self.files[self.i])
            self.buf.copy_(x)
            self.i += 1
            if self.i % 50 == 0:
                print(f"      calibrate {self.i}/{len(self.files)}", flush=True)
            return [int(self.buf.data_ptr())]

        def read_calibration_cache(self):
            if self.cache.exists():
                print(f"      dung cache {self.cache.name}")
                return self.cache.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            self.cache.write_bytes(cache)
            print(f"      ghi cache {self.cache.name} ({len(cache)} byte)")

    return _Calib()


def cache_path_for(algo, n):
    return CACHE_DIR / f"calib_{algo}_{n}.cache"


def _self_test(n=100):
    from bench_utils import load_pmfnet, setup_measurement_env
    from eval_dice import preprocess

    setup_measurement_env(tf32_matmul=True)
    model = load_pmfnet(device="cuda")
    files = calib_files(n)
    print(f"\n{'='*76}\nDAI activation — do tren {len(files)} anh calibration\n{'='*76}")

    acts = {}
    hooks = []

    def hook(name):
        def f(_m, _i, o):
            if isinstance(o, torch.Tensor) and o.is_floating_point():
                a = o.detach().abs()
                acts.setdefault(name, []).append(
                    (a.max().item(), a.mean().item(),
                     torch.quantile(a.flatten().float()[:100000], 0.999).item()))
        return f

    import torch.nn as nn
    for nm, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.Softmax, nn.Conv2d, nn.Linear, nn.GELU)):
            hooks.append(m.register_forward_hook(hook(f"{type(m).__name__}|{nm}")))

    with torch.inference_mode():
        for f in files:
            model(preprocess(f)[0])
    for h in hooks:
        h.remove()

    rows = []
    for nm, v in acts.items():
        mx = max(t[0] for t in v)
        p999 = max(t[2] for t in v)
        rows.append((nm.split("|")[0], nm.split("|")[1], mx, p999, mx / max(p999, 1e-9)))

    print(f"  {'loai':12} {'max':>10} {'p99.9':>10} {'max/p99.9':>11}   y nghia")
    print("  " + "-" * 72)
    by_type = {}
    for t, _, mx, p, r in rows:
        by_type.setdefault(t, []).append((mx, p, r))
    for t, v in sorted(by_type.items(), key=lambda kv: -max(x[2] for x in kv[1])):
        mx = max(x[0] for x in v); p = max(x[1] for x in v); r = max(x[2] for x in v)
        note = ("duoi rat dai -> Entropy2 se cat, MinMax se gian scale"
                if r > 5 else "duoi ngan -> hai calibrator se giong nhau")
        print(f"  {t:12} {mx:10.2f} {p:10.2f} {r:10.1f}x   {note}")

    worst = sorted(rows, key=lambda x: -x[4])[:8]
    print(f"\n  8 layer co duoi dai nhat (ung vien giu o fp16 khi mixed precision):")
    for t, nm, mx, p, r in worst:
        print(f"    {r:7.1f}x  max {mx:9.2f}  {t:10} {nm[:44]}")


if __name__ == "__main__":
    _self_test(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
