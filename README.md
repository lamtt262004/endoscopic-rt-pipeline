# endoscopic-rt-pipeline

Real-time polyp segmentation on endoscopic video, deployed with TensorRT.

## Objective

Take **PMFNet** (the polyp segmentation model I published at ATiGB 2025) from a PyTorch
checkpoint to a video pipeline that runs on a 4 GB laptop GPU, and measure it honestly.

## Performance

### Test setup

| item | value |
|---|---|
| Hardware | RTX 3050 Laptop, 4 GB VRAM, PCIe 3.0 ×8 |
| Software | CUDA 12.8 · TensorRT 10.16 · PyTorch 2.11 |
| Input | 720×576 video, VC-1, 25 fps |
| Batch size | 1 |


### Results

| configuration | inference | latency | throughput |
|---|---|---|---|
| PyTorch eager | 36.73 ms | 44.11 ms | 24.0 FPS |
| PyTorch + CUDA Graphs, TF32 matmul | 12.81 ms | 20.43 ms | 60.7 FPS |
| TensorRT FP32, TF32 disabled | 11.60 ms | 19.00 ms | 66.0 FPS |
| TensorRT TF32 | 10.84 ms | 18.56 ms | 69.4 FPS |
| **TensorRT FP16** | **7.09 ms** | **15.26 ms** | **91.1 FPS** |

**Inference** is the forward pass alone. The four accelerated rows were interleaved in one paired
run; the eager baseline was measured in an earlier session, so read 36.73 ms as indicative rather
than paired. **Latency** and **throughput** come from the full video pipeline, with all five
configurations run back to back in a single session.

Every PyTorch row runs cuDNN with TF32 enabled, which is the library default, so convolutions reach
Tensor Cores even where the row says FP32. Only matmul TF32 differs: off in the eager inference
figure, on everywhere else.

Precision does not move Dice. On 100 Kvasir-SEG images, PyTorch scores **0.9340** and all three
engines **0.9339**; on a separate 200 images, held out from the calibration set used in
[Quantization](#quantization) below, every configuration scores **0.9450**. The two sets are
comparable within themselves but not against each other, so both are quoted rather than merged.
FP16 flips **0.0029 %** of pixels at the 0.5 threshold, about 2 in a 256×256 mask.

https://github.com/user-attachments/assets/ccac0cf0-5d2c-415b-bd4e-5dfeff407ba3

The bottom row of the table, running: TensorRT FP16 engine, 720×576 source, overlay composed on the
GPU. Mask area is printed per frame. Rendered by `src/make_demo.py --trt fp16`, so what is shown is
produced by the same engine the numbers describe.

### Notes

TensorRT and PyTorch's cuDNN path both enable TF32 by default, so the usual "TRT FP32 vs PyTorch
FP32" comparison does not measure what it claims. Building an extra engine with TF32 explicitly
disabled separates the compiler contribution (**1.10×**) from the Tensor Core contribution
(**1.07×**); FP16 adds a further **1.51×**.

On fusion: the 1000-node ONNX graph compiles down to 441 layers, 192 of which are Myelin fusion
subgraphs absorbing roughly 1100 operations between them.

## Operating point

### Inference in isolation

Everything above was measured while reading a video file as fast as the pipeline allows, so the GPU
barely idles. A real endoscope sends a frame every **40 ms**; after ~15 ms of work the GPU sits idle
for ~25 ms and clocks itself down.

The table below isolates that one variable: a single engine call on a fixed tensor, with an enforced
sleep between calls and nothing else competing for the GPU.

| gap between frames | FP16 inference | SM clock |
|---|---|---|
| 0 ms — file at full speed | 7.19 ms | 1762 MHz |
| 8 ms — 60 fps source | 8.88 ms | 1419 MHz |
| 18 ms — 40 fps source | 13.46 ms | 730 MHz |
| **33 ms — 25 fps source (this dataset)** | **19.23 ms** | **439 MHz** |

### Notes

FP16's advantage over TF32 shrinks from **1.48×** to **1.07×**, which means most of it is an artefact
of how the benchmark feeds the GPU rather than a property of the deployment. Forcing the GPU to stay
busy with filler work removes the effect completely, but costs more in median latency than it
recovers in the tail.

### The full pipeline

Running everything instead — three threads, a real video, frames released on a fixed cadence, and
latency timed from when each frame arrives rather than from when the pipeline gets to it — gives a
milder picture. Three runs of 300 frames each:

| source | frame budget | latency p99 | over budget | output rate | queue depth |
|---|---|---|---|---|---|
| 25 fps | 40.0 ms | 30.6 ms | **0.7 %** | 25.0 fps | 0–1 |
| 30 fps | 33.3 ms | 30.5 ms | 1.0 % | 30.0 fps | 0–1 |
| 40 fps | 25.0 ms | 36.5 ms | 6–13 % | 39.9 fps | 1–2 |
| 60 fps | 16.7 ms | 64.9 ms | **44–89 %** | 59.7 fps | 3 |

**Throughput keeps up at every rate** — output matches input and the queue never exceeds three
frames. What fails is the latency budget, and only once the source is fast enough to shrink it.

Median latency also stays at 16–21 ms across all four rates, rather than tracking the 7.19 → 19.23 ms
slowdown measured on inference alone. **Threading is its own keep-alive**: decoding frame *k+1* and
encoding frame *k−1* overlap the GPU work for frame *k*, so the card never idles long enough to drop
to 439 MHz. The effect that made inference 2.7× slower in isolation is largely absorbed by the
pipeline that ships. Reproduce with `src/video_infer.py --live 25,30,40,60`.


## Quantization

### Method

Post-training quantization with 300 calibration images, comparing two calibrators:
`IInt8EntropyCalibrator2` and `IInt8MinMaxCalibrator`. Evaluated on 200 images that do **not** overlap
the calibration set.

### Results

The engine builds, runs, and leaves Dice unchanged. It is also **not faster than FP16**, and its
latency tail is worse. Both calibrators give identical results.

### Reason

Counting layers explains it: **36 of 436 layers actually execute INT8 kernels**, and every one of them
is a convolution, no GEMM or MatMul at all. PVT-v2's transformer blocks were already absorbed into
Myelin fusion subgraphs during the FP16 pass, and Myelin emits no INT8 path for those patterns, so
quantization never reached the part that dominates runtime. That is also why the choice of calibrator
makes no difference.

Fusion did not cancel the quantization win so much as put it out of reach: once Myelin owns a
subgraph it picks the kernel for the whole cluster, and it offers no INT8 one. The implicit
calibration path has no way to reopen that decision.

## Limitations

### Still images vs video

Dice 0.9450 is measured on still images and does not transfer to video. I checked this with a
**negative control**: a video of a normal ileocecal valve, containing no polyp.

| video | median mask area | frames with mask > 1 % |
|---|---|---|
| small polyp | 1.08 % | 52 % |
| flat polyp | 0.00 % | 30 % |
| no polyp | 0.00 % | 27 % |

First 400 frames of each video, one run of `src/screen_videos.py`.

The model separates a clearly visible polyp from the control, but is **indistinguishable from it on
flat lesions**. Its largest false positive on the control video covers 32.8 % of the frame, more than
its largest true positive on the polyp video, so no area threshold can separate the two.

The failure mode is specific: it fires on **near, bright, in-focus mucosal wall** — folds, and tissue
pressed against the scope. Kvasir-SEG contains no normal mucosa, no motion blur and no narrow-band
imaging, so the model never had a chance to learn to stay silent.

### Temporal filtering

False-positive and true-positive episodes have nearly the same duration distribution (medians of 1
and 2 frames), so duration is not a feature that separates them: cutting false positives by 18 %
costs 44 % of true detection episodes. Closing this gap needs **training data containing negative
frames** (SUN-SEG, LDPolypVideo), not post-processing.

## Data

| path, relative to the repository root | source |
|---|---|
| `Kvasir_images/segmented-images/{images,masks}/` | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) — 1000 polyp images with masks |
| `hyper-kvasir-videos/videos/*.avi` | [HyperKvasir](https://datasets.simula.no/hyper-kvasir/) — labelled videos, 720×576 @ 25 fps |
| `hyper-kvasir-videos/video-annotations.csv` | HyperKvasir |

## Usage

Python 3.10, CUDA 12.8, an NVIDIA GPU.

```bash
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# export ONNX and validate it by Dice
python src/export_onnx.py --n 20

# build FP32 / TF32 / FP16 engines, count fused layers, measure latency and Dice
python src/build_engine.py

# video pipeline
python src/video_infer.py --verify
python src/video_infer.py --trt fp16 --video 76866169 --n 400

# the two measurements that matter most
python src/diag_05_live_rate.py
python src/count_precision.py

# demo video, and the clip embedded above
python src/make_demo.py --n 500 --trt fp16
python src/make_clip.py --secs 12
```

Videos are selected by clinical label rather than by UUID:

```bash
python src/videos.py "flat polyp"
python src/video_infer.py --video 1de3ef0f --n 400
```

## Code

**Pipeline**

| file | purpose |
|---|---|
| `video_infer.py` | 8-stage pipeline, per-stage breakdown, threaded pipelining |
| `videos.py` | select videos by clinical label |
| `temporal.py` | temporal filter and its trade-off curve |

**Engine build**

| file | purpose |
|---|---|
| `export_onnx.py` | ONNX export, graph patching, Dice-based validation |
| `trt_utils.py` | builder and runner |
| `build_engine.py` | FP32/TF32/FP16 sweep, fusion depth, latency, accuracy |

**Quantization**

| file | purpose |
|---|---|
| `calib_loader.py` | calibrators and activation-range analysis |
| `build_int8.py` | PTQ with two calibrators |
| `count_precision.py` | counts layers that actually execute INT8 |

**Measurement**

| file | purpose |
|---|---|
| `bench_utils.py` | condition capture, percentiles, deadline-miss rate |
| `eval_dice.py` | Dice/IoU under a fixed protocol |
| `show_results.py` | reads back `benchmarks/results.jsonl` |
| `paired_videos.py` | compares three videos, rounds interleaved so drift cancels |
| `paired_pipelined.py` | same comparison for pipelined throughput |
| `screen_videos.py` | mask-area statistics across videos, including the negative control |
| `diag_01`…`diag_06` | async, CUDA Graphs, TF32, FP16 spikes, feed rate, keep-alive |

**Demo**

| file | purpose |
|---|---|
| `make_demo.py` | paired demo videos, with and without a polyp |
| `make_clip.py` | trims the demo to the clip embedded above |

## Model

**PMFNet**. Published as *"Polyp
Segmentation with Transformer-CNN Integration and Multi-Scale Feature Fusion"*, ATiGB 2025.
