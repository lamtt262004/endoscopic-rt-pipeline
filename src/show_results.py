"""
show_results.py — doc lai benchmarks/results.jsonl cho de nhin.

    python src/show_results.py          # bang tom tat
    python src/show_results.py -v       # kem dieu kien do (tf32, clock, nhiet)
"""
import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "results.jsonl"


def main():
    verbose = "-v" in sys.argv
    if not PATH.exists():
        sys.exit(f"Chua co {PATH} — chay src/diag_01_async.py truoc.")

    rows = [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{'thoi diem':<20} {'config':<28} {'mean':>7} {'p50':>7} {'p99':>7} "
          f"{'jit':>5} {'miss%':>6} {'FPS':>6}")
    print("-" * 92)

    for r in rows:
        variants = ([("", r)] if "mean_ms" in r
                    else [(k, r[k]) for k in ("eager", "graph") if k in r])
        for suffix, s in variants:
            name = r.get("config", "?") + (f" [{suffix}]" if suffix else "")
            print(f"{r.get('ts','?'):<20} {name:<28} {s['mean_ms']:7.2f} "
                  f"{s['p50_ms']:7.2f} {s['p99_ms']:7.2f} {s['jitter']:5.2f} "
                  f"{s['miss_rate_pct']:6.1f} {1000/s['mean_ms']:6.1f}")
        if verbose:
            env = r.get("env", {})
            bits = []
            if env:
                bits.append(f"tf32 cudnn={env.get('tf32_cudnn')}/matmul={env.get('tf32_matmul')}")
                bits.append(f"benchmark={env.get('cudnn_benchmark')}")
                bits.append(f"clock@start={env.get('clocks_sm_mhz')}MHz {env.get('temp_c')}C")
            for k in ("gpu_busy_pct", "enqueue_ratio_n1", "n_kernels", "speedup"):
                if k in r:
                    bits.append(f"{k}={r[k]}")
            for k in ("clock_eager", "clock_graph"):
                if k in r:
                    bits.append(f"{k}: {r[k]}")
            for b in bits:
                print(f"{'':20} └ {b}")
            print()

    print(f"\n{len(rows)} lan do trong {PATH}")
    print("Luu y: moi dong deu kem dieu kien do (xem -v). So do o dieu kien khac "
          "nhau thi khong so sanh truc tiep duoc.")


if __name__ == "__main__":
    main()
