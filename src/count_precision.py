"""
count_precision.py — bat co int8 xong thi thuc su bao nhieu layer chay int8?

    python src/count_precision.py            # build lai voi DETAILED roi dem
    python src/count_precision.py --reuse    # dung engine _detailed da co
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_loader import cache_path_for, calib_files, make_calibrator
from trt_utils import build_engine

ROOT = Path(__file__).resolve().parent.parent
ONNX = ROOT / "onnx" / "pmfnet_legacy.onnx"
ENG = ROOT / "engines"


def build_detailed(tag, n_calib=300, workspace_gb=1.5):
    out = ENG / f"pmfnet_{tag}_detailed.engine"
    if out.exists():
        print(f"  {tag:16} da co {out.name}")
        return out
    print(f"  {tag:16} build lai voi DETAILED ...", flush=True)
    kw = dict(workspace_gb=workspace_gb, detailed=True, tf32=True)
    if tag == "trt_fp16":
        build_engine(ONNX, out, fp16=True, **kw)
    elif tag.startswith("trt_int8_"):
        algo = tag.replace("trt_int8_", "")
        cal = make_calibrator(algo, calib_files(n_calib), cache_path_for(algo, n_calib))
        build_engine(ONNX, out, fp16=True, int8=True, calibrator=cal, **kw)
    else:
        raise ValueError(tag)
    return out


def count(path):
    rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    eng = rt.deserialize_cuda_engine(Path(path).read_bytes())
    insp = eng.create_engine_inspector()

    w_type, io_type, tac_type = Counter(), Counter(), Counter()
    int8_layers, no_json = [], 0

    def dtype_of(io):
        s = str(io.get("Format/Datatype", io.get("Format", "?")))
        for k in ("Int8", "int8", "Half", "fp16", "Float", "fp32", "Int32"):
            if k in s:
                return {"int8": "Int8", "fp16": "Half", "fp32": "Float"}.get(k, k)
        return s[:24]

    for i in range(eng.num_layers):
        try:
            d = json.loads(insp.get_layer_information(i, trt.LayerInformationFormat.JSON))
        except Exception:
            d = None
        if not isinstance(d, dict):
            no_json += 1
            continue

        w = d.get("Weights")
        w_type[w.get("Type", "?") if isinstance(w, dict) else "<khong co weight>"] += 1

        for io in (d.get("Outputs") or []):
            io_type[dtype_of(io)] += 1

        tac = str(d.get("TacticName", ""))
        low = tac.lower()
        if "i8" in low or "imma" in low or "int8" in low:
            t = "int8"
            int8_layers.append((d.get("LayerType", "?"), str(d.get("Name", "?"))[:56]))
        elif "f16" in low or "hmma" in low or "h1688" in low:
            t = "fp16"
        elif "f32" in low or "s1688" in low or "tf32" in low:
            t = "fp32/tf32"
        elif tac == "":
            t = "<khong co tactic>"
        else:
            t = "khac"
        tac_type[t] += 1

    n = eng.num_layers
    del eng
    return n, w_type, io_type, tac_type, int8_layers, no_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="trt_fp16,trt_int8_entropy2")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    print(f"{'='*76}\nBUILD lai voi ProfilingVerbosity.DETAILED\n{'='*76}")
    paths = {}
    for t in args.tags.split(","):
        p = ENG / f"pmfnet_{t}_detailed.engine"
        paths[t] = p if (args.reuse and p.exists()) else build_detailed(t)

    for t, p in paths.items():
        n, w, io, tac, int8, no_json = count(p)
        print(f"\n{'='*76}\n{t}  —  {n} layer\n{'='*76}")

        print(f"  {'kieu trong so':26} {'kernel (tu TacticName)':30}")
        print("  " + "-" * 62)
        keys = sorted(set(w) | set(tac), key=lambda k: -(w.get(k, 0) + tac.get(k, 0)))
        for k in keys:
            a, b = w.get(k, 0), tac.get(k, 0)
            if a or b:
                print(f"  {k:22} {a:5d}   |  {k:16} {b:5d}")
        print(f"\n  kieu tensor dau ra:")
        for k, v in io.most_common(6):
            print(f"    {k:22} {v:5d}")

        n8 = tac.get("int8", 0)
        w8 = w.get("Int8", 0)
        print(f"\n  => kernel int8 : {n8}/{n} layer ({n8/n*100:.1f} %)")
        print(f"  => weight int8 : {w8}/{n} layer ({w8/n*100:.1f} %)")
        if no_json:
            print(f"  ({no_json} layer khong doc duoc JSON)")
        if int8:
            print(f"\n  Loai layer duoc int8:")
            for k, v in Counter(t_ for t_, _ in int8).most_common(8):
                print(f"    {v:4d}  {k}")
            print(f"  Vi du ten:")
            for _, nm in int8[:6]:
                print(f"    {nm}")


if __name__ == "__main__":
    main()
