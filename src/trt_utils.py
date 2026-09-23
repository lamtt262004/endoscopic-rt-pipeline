"""
trt_utils.py — build engine tu ONNX + chay inference bang torch tensor.

    python src/trt_utils.py
"""
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

_ROOT = Path(__file__).resolve().parent.parent

TRT2TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


def build_engine(onnx_path, engine_path=None, fp16=False, int8=False,
                 workspace_gb=1.0, verbose=False, tf32=True, info=None,
                 calibrator=None, precision_overrides=None, detailed=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = Path(onnx_path).read_bytes()
    if not parser.parse(onnx_bytes):
        msgs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Parse ONNX that bai:\n  " + "\n  ".join(msgs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * (1 << 30)))
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("GPU nay khong co fp16 nhanh")
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("GPU nay khong co int8 nhanh")
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is None:
            raise RuntimeError("int8=True nhung khong co calibrator — "
                               "xem src/calib_loader.py")
        config.int8_calibrator = calibrator

    if precision_overrides:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        n_hit = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for pat, dtype in precision_overrides.items():
                if pat in layer.name:
                    layer.precision = dtype
                    for j in range(layer.num_outputs):
                        layer.set_output_type(j, dtype)
                    n_hit += 1
                    break
        if info is not None:
            info["precision_override_layers"] = n_hit
        print(f"      ep precision cho {n_hit} layer")

    if tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)

    if detailed:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    if info is not None:
        info["layers_before_fusion"] = network.num_layers
        info["workspace_gb"] = workspace_gb
        info["flags"] = {"fp16": fp16, "int8": int8, "tf32": tf32}

    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if plan is None:
        raise RuntimeError("build_serialized_network tra ve None — xem log builder")

    plan = bytes(plan)
    if info is not None:
        info["build_s"] = build_s
        info["plan_mb"] = len(plan) / 1e6

    if engine_path:
        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        Path(engine_path).write_bytes(plan)
    return plan


def engine_layers(engine):
    insp = engine.create_engine_inspector()
    names = []
    for i in range(engine.num_layers):
        try:
            s = insp.get_layer_information(i, trt.LayerInformationFormat.ONELINE)
            names.append(s.strip())
        except Exception:
            names.append("?")
    return names


class TRTRunner:
    def __init__(self, plan_or_path, device="cuda", stream=None):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        plan = (Path(plan_or_path).read_bytes()
                if isinstance(plan_or_path, (str, Path)) else plan_or_path)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError("deserialize that bai — engine build tu GPU/TRT khac?")
        self.context = self.engine.create_execution_context()
        self.device = device

        self.stream = stream or torch.cuda.Stream(device=device)

        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

    def __call__(self, *tensors):
        assert len(tensors) == len(self.inputs), \
            f"can {len(self.inputs)} input, nhan {len(tensors)}"

        for name, t in zip(self.inputs, tensors):
            t = t.contiguous()
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        outs = []
        for name in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = TRT2TORCH[self.engine.get_tensor_dtype(name)]
            o = torch.empty(shape, dtype=dtype, device=self.device)
            self.context.set_tensor_address(name, o.data_ptr())
            outs.append(o)

        cur = torch.cuda.current_stream(device=self.device)
        self.stream.wait_stream(cur)
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        cur.wait_stream(self.stream)

        for t in list(tensors) + outs:
            t.record_stream(self.stream)

        return outs[0] if len(outs) == 1 else outs


def _smoke():
    import torch.nn as nn

    print("=" * 66)
    print("smoke test: torch -> ONNX -> onnxsim -> TRT engine -> inference")
    print("=" * 66)
    print(f"  TRT {trt.__version__} | torch {torch.__version__}")

    tmp = _ROOT / "benchmarks" / "_smoke"
    tmp.mkdir(parents=True, exist_ok=True)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 16, 3, padding=1)
            self.n1 = nn.GroupNorm(4, 16)
            self.c2 = nn.Conv2d(16, 1, 1)

        def forward(self, x):
            return torch.sigmoid(self.c2(nn.functional.gelu(self.n1(self.c1(x)))))

    m = Tiny().eval().cuda()
    x = torch.randn(1, 3, 64, 64, device="cuda")
    with torch.inference_mode():
        ref = m(x).clone()

    onnx_path = tmp / "tiny.onnx"
    torch.onnx.export(m, (x,), str(onnx_path), opset_version=17,
                      input_names=["input"], output_names=["output"],
                      dynamo=False)
    print(f"\n  [1] export ONNX opset 17  -> {onnx_path.stat().st_size/1024:.1f} KB")

    import onnx
    from onnxsim import simplify
    model_onnx = onnx.load(str(onnx_path))
    n_before = len(model_onnx.graph.node)
    model_sim, check = simplify(model_onnx)
    onnx.save(model_sim, str(onnx_path))
    print(f"  [2] onnxsim: {n_before} -> {len(model_sim.graph.node)} node, "
          f"check={'OK' if check else 'fail'}")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x.cpu().numpy()})[0]
    d_ort = np.abs(ort_out - ref.cpu().numpy()).max()
    print(f"  [3] onnxruntime vs torch : max |diff| = {d_ort:.2e}")

    for fp16 in (False, True):
        tag = "fp16" if fp16 else "fp32"
        plan = build_engine(onnx_path, tmp / f"tiny_{tag}.engine",
                            fp16=fp16, workspace_gb=0.5)
        runner = TRTRunner(plan)
        out = runner(x)
        torch.cuda.synchronize()
        d = (out.float() - ref).abs().max().item()
        print(f"  [4] TRT {tag}: engine {len(plan)/1024:.0f} KB | "
              f"out {tuple(out.shape)} {out.dtype} | max |diff| vs torch = {d:.2e}")

    print("\n  Ca duong di chay duoc.")
    print(f"  (file tam o {tmp}, xoa duoc)")


if __name__ == "__main__":
    _smoke()
