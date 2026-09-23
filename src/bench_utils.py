"""bench_utils.py — bo do nghe do dac dung chung."""
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "PMFNet") not in sys.path:
    sys.path.insert(0, str(_ROOT / "PMFNet"))


def setup_measurement_env(tf32_matmul: bool = False, tf32_cudnn: bool = True):
    torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
    torch.backends.cudnn.allow_tf32 = tf32_cudnn

    torch.backends.cudnn.benchmark = True


def describe_env() -> dict:
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    env.update(query_gpu_state())
    return env


THROTTLE_BITS = {
    0x1: "GpuIdle", 0x2: "AppClocksSetting", 0x4: "SwPowerCap",
    0x8: "HwSlowdown", 0x10: "SyncBoost", 0x20: "SwThermalSlowdown",
    0x40: "HwThermalSlowdown", 0x80: "HwPowerBrake", 0x100: "DisplayClock",
}


def decode_throttle(raw) -> str:
    try:
        v = int(str(raw).strip(), 16)
    except (ValueError, TypeError):
        return str(raw)
    if v == 0:
        return "None"
    return "|".join(name for bit, name in THROTTLE_BITS.items() if v & bit) or hex(v)


def query_gpu_state() -> dict:
    fields = "clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"
    for throttle_field in ("clocks_event_reasons.active",
                           "clocks_throttle_reasons.active"):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields},{throttle_field}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip().split(",")
            return {
                "clocks_sm_mhz": out[0].strip(),
                "clocks_mem_mhz": out[1].strip(),
                "temp_c": out[2].strip(),
                "power_w": out[3].strip(),
                "util_pct": out[4].strip(),
                "throttle": decode_throttle(out[5]),
            }
        except Exception:
            continue
    return {"gpu_state": "nvidia-smi khong doc duoc"}


def load_pmfnet(ckpt_path: str | Path = None, device: str = "cuda"):
    from models import pvt_v2_b2, PCRN

    ckpt_path = Path(ckpt_path or _ROOT / "Weights" / "ckpt0.9346.ckpt")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}

    model = PCRN(backbone=pvt_v2_b2())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing or unexpected:
        print(f"[load] !! missing[:5]={missing[:5]}")
        print(f"[load] !! unexpected[:5]={unexpected[:5]}")
        raise RuntimeError("State dict khong khop — dung lai, dung do tiep.")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[load] params = {n_params/1e6:.2f} M")
    return model.eval().to(device)


@dataclass
class LatencyStats:
    n: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    jitter: float
    miss_rate_pct: float
    budget_ms: float

    def __str__(self):
        return (f"n={self.n}  mean={self.mean_ms:6.2f}  p50={self.p50_ms:6.2f}  "
                f"p90={self.p90_ms:6.2f}  p99={self.p99_ms:6.2f}  "
                f"jitter={self.jitter:4.2f}x  "
                f"miss(>{self.budget_ms:.1f}ms)={self.miss_rate_pct:5.1f}%")


def summarize(samples_ms, budget_ms: float = 16.7) -> LatencyStats:
    a = np.asarray(samples_ms, dtype=np.float64)
    p50, p90, p99 = np.percentile(a, [50, 90, 99])
    return LatencyStats(
        n=len(a), mean_ms=float(a.mean()),
        p50_ms=float(p50), p90_ms=float(p90), p99_ms=float(p99),
        min_ms=float(a.min()), max_ms=float(a.max()),
        jitter=float(p99 / p50),
        miss_rate_pct=float((a > budget_ms).mean() * 100),
        budget_ms=budget_ms,
    )


@torch.inference_mode()
def bench(fn, warmup: int = 50, iters: int = 500, budget_ms: float = 16.7):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    return summarize(samples, budget_ms), samples


def save_result(row: dict, path: str | Path = None):
    path = Path(path or _ROOT / "benchmarks" / "results.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"[save] -> {path}")
