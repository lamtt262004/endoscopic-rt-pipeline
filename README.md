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
| PyTorch + CUDA Graphs + TF32 | 12.81 ms | 20.43 ms | 60.7 FPS |
| TensorRT FP32 (TF32 disabled) | 11.60 ms | 19.00 ms | 66.0 FPS |
| TensorRT TF32 | 10.84 ms | 18.56 ms | 69.4 FPS |
| **TensorRT FP16** | **7.09 ms** | **15.26 ms** | **91.1 FPS** |

**Inference** is the forward pass alone, with all five configurations interleaved in a single run.
**Latency** and **throughput** come from the full video pipeline, again all five back to back in one
session.

Dice stays at **0.9450** across all five (200 held-out Kvasir-SEG images). FP16 flips only
**0.0033 %** of pixels at the 0.5 threshold (about 2 pixels in a 256×256 mask).

https://github.com/user-attachments/assets/d9fab0f8-8124-47f0-9b6f-516df9858f45

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

### Feed rate

Everything above was measured while reading a video file as fast as the pipeline allows, so the GPU
barely idles. A real endoscope sends a frame every **40 ms**; after ~15 ms of work the GPU sits idle
for ~25 ms and clocks itself down.

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

The pipeline still keeps up with a 25 fps source - latency p50 is 26.9 ms against a 40 ms budget.


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

The fusion win and the quantization win cancel each other out, and the implicit, calibration path cannot have both. 

## Limitations

### Still images vs video

Dice 0.9450 is measured on still images and does not transfer to video. I checked this with a
**negative control**: a video of a normal ileocecal valve, containing no polyp.

| video | median mask area | frames with mask > 1 % |
|---|---|---|
| small polyp | 1.17 % | 52 % |
| flat polyp | 0.00 % | 30 % |
| no polyp | 0.00 % | 30 % |

The model separates a clearly visible polyp from the control, but is **indistinguishable from it on
flat lesions**. Its largest false positive on the control video covers 32.7 % of the frame, more than
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
| `make_gif.py` | gif export, if a self-playing image is preferred |

## Model

**PMFNet** . Published as *"Polyp
Segmentation with Transformer-CNN Integration and Multi-Scale Feature Fusion"*, ATiGB 2025.



