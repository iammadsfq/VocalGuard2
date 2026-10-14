"""
Quantization and CPU Latency Benchmarking for Vocal-Guard M1 (LCNN).
Target: Social Robot On-Device Streaming Audio Inference (e.g. live call/voice assistant).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import pandas as pd
import torch
import torch.nn as nn

from vocal_guard.models.lcnn import LCNN


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    time_frames: int = 399,
):
    """
    Export PyTorch LCNN model to ONNX format.
    Batch size is dynamic (flexible for single streaming sample or batching).
    time_frames is fixed to the specific ring-buffer duration:
      - 399 frames (4.0s full utterance)
      - 200 frames (2.0s streaming ring buffer)
      - 128 frames (1.28s low-latency streaming chunk)
    """
    model.eval()
    dummy_input = torch.randn(1, 1, 60, time_frames)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"✅ ONNX model successfully exported: {output_path} (time_frames={time_frames})")


def quantize_onnx_model(
    input_onnx_path: str,
    output_quant_path: str,
):
    """Apply dynamic INT8 quantization to the ONNX model."""
    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=output_quant_path,
        weight_type=QuantType.QInt8,
    )
    print(f"✅ Quantized INT8 ONNX model successfully created: {output_quant_path}")


def benchmark_pytorch(
    model: nn.Module,
    input_tensor: torch.Tensor,
    warmup: int = 25,
    iterations: int = 100,
    num_threads: int = 1,
) -> dict:
    """Benchmark PyTorch CPU inference latency."""
    torch.set_num_threads(num_threads)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def benchmark_onnxruntime(
    onnx_path: str,
    dummy_input: np.ndarray,
    warmup: int = 25,
    iterations: int = 100,
    num_threads: int = 1,
) -> dict:
    """Benchmark ONNX Runtime CPU inference latency."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def verify_numeric_parity(
    pytorch_model: nn.Module,
    onnx_path: str,
    onnx_int8_path: str,
    test_input: np.ndarray,
):
    """Verify output parity and score difference between PyTorch FP32, ONNX FP32, and ONNX INT8."""
    pytorch_model.eval()
    with torch.no_grad():
        pt_out = pytorch_model(torch.from_numpy(test_input)).numpy()

    sess_fp32 = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_fp32_out = sess_fp32.run(None, {sess_fp32.get_inputs()[0].name: test_input})[0]

    sess_int8 = ort.InferenceSession(onnx_int8_path, providers=["CPUExecutionProvider"])
    onnx_int8_out = sess_int8.run(None, {sess_int8.get_inputs()[0].name: test_input})[0]

    max_diff_fp32 = float(np.max(np.abs(pt_out - onnx_fp32_out)))
    max_diff_int8 = float(np.max(np.abs(pt_out - onnx_int8_out)))

    pt_score = float(pt_out[0, 0] - pt_out[0, 1])
    onnx_fp32_score = float(onnx_fp32_out[0, 0] - onnx_fp32_out[0, 1])
    onnx_int8_score = float(onnx_int8_out[0, 0] - onnx_int8_out[0, 1])

    print("\n" + "=" * 70)
    print("🔬 NUMERIC INTEGRITY & SCORE FIDELITY CHECK")
    print("=" * 70)
    print(f"PyTorch FP32 Logits:         {pt_out.tolist()} -> Score: {pt_score:.4f}")
    print(f"ONNX FP32 Logits:            {onnx_fp32_out.tolist()} -> Score: {onnx_fp32_score:.4f} (Max Diff: {max_diff_fp32:.2e})")
    print(f"ONNX INT8 Logits:            {onnx_int8_out.tolist()} -> Score: {onnx_int8_score:.4f} (Max Diff: {max_diff_int8:.2e})")
    print(f"Score Delta (|FP32 - INT8|): {abs(pt_score - onnx_int8_score):.4f}")
    print("=" * 70)


def main():
    checkpoint_path = "models/lcnn_best_model.pt"
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    print("=" * 70)
    print("🚀 VOCAL-GUARD M1 QUANTIZATION & CPU STREAMING BENCHMARK")
    print("=" * 70)

    # 1. Load trained PyTorch model
    print("Loading PyTorch LCNN Model...")
    model = LCNN(num_classes=2)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Export & Quantize 4.0-second model (Standard ASVspoof window: 399 frames)
    onnx_fp32_4s = "models/lcnn_m1_4s.onnx"
    onnx_int8_4s = "models/lcnn_m1_4s_int8.onnx"
    export_to_onnx(model, onnx_fp32_4s, time_frames=399)
    quantize_onnx_model(onnx_fp32_4s, onnx_int8_4s)

    # 3. Export & Quantize 1.28-second streaming model (128 frames)
    onnx_fp32_1s = "models/lcnn_m1_128frames.onnx"
    onnx_int8_1s = "models/lcnn_m1_128frames_int8.onnx"
    export_to_onnx(model, onnx_fp32_1s, time_frames=128)
    quantize_onnx_model(onnx_fp32_1s, onnx_int8_1s)

    # 4. Measure model sizes
    size_pt = os.path.getsize(checkpoint_path) / (1024 * 1024)
    size_fp32_4s = os.path.getsize(onnx_fp32_4s) / (1024 * 1024)
    size_int8_4s = os.path.getsize(onnx_int8_4s) / (1024 * 1024)
    reduction = ((size_fp32_4s - size_int8_4s) / size_fp32_4s) * 100.0

    print("\n" + "=" * 70)
    print("💾 MODEL COMPRESSION RESULTS:")
    print(f"  • PyTorch FP32 Checkpoint: {size_pt:.2f} MB")
    print(f"  • ONNX FP32 Model:         {size_fp32_4s:.2f} MB")
    print(f"  • ONNX Dynamic INT8:       {size_int8_4s:.2f} MB (Compression: {reduction:.1f}%)")
    print("=" * 70)

    # 5. Verify numeric parity on 4.0s sample
    np.random.seed(42)
    sample_4s = np.random.randn(1, 1, 60, 399).astype(np.float32)
    verify_numeric_parity(model, onnx_fp32_4s, onnx_int8_4s, sample_4s)

    # 6. Benchmark Latency for Streaming Scenarios
    scenarios = [
        ("4.0s Full Utterance (399 frames)", onnx_fp32_4s, onnx_int8_4s, sample_4s, 4.0),
        ("1.28s Streaming Chunk (128 frames)", onnx_fp32_1s, onnx_int8_1s, np.random.randn(1, 1, 60, 128).astype(np.float32), 1.28),
    ]

    benchmark_rows = []

    print("\n" + "=" * 70)
    print("⚡ CPU INFERENCE LATENCY BENCHMARK (Intel i7-8665U)")
    print("=" * 70)

    for label, fp32_path, int8_path, x_input, duration_sec in scenarios:
        x_pt = torch.from_numpy(x_input)
        print(f"\n▶ Scenario: {label}")

        # PyTorch FP32 (Single Thread)
        pt_res = benchmark_pytorch(model, x_pt, num_threads=1)
        rtf_pt = (pt_res["mean_ms"] / 1000.0) / duration_sec
        print(f"  • PyTorch FP32 (1T): {pt_res['mean_ms']:.2f} ± {pt_res['std_ms']:.2f} ms | p95: {pt_res['p95_ms']:.2f} ms | RTF: {rtf_pt:.4f}")

        # ONNX FP32 (Single Thread)
        onnx_fp32_res = benchmark_onnxruntime(fp32_path, x_input, num_threads=1)
        rtf_fp32 = (onnx_fp32_res["mean_ms"] / 1000.0) / duration_sec
        print(f"  • ONNX FP32 (1T):    {onnx_fp32_res['mean_ms']:.2f} ± {onnx_fp32_res['std_ms']:.2f} ms | p95: {onnx_fp32_res['p95_ms']:.2f} ms | RTF: {rtf_fp32:.4f}")

        # ONNX INT8 (Single Thread)
        onnx_int8_1t = benchmark_onnxruntime(int8_path, x_input, num_threads=1)
        rtf_int8_1t = (onnx_int8_1t["mean_ms"] / 1000.0) / duration_sec
        print(f"  • ONNX INT8 (1T):    {onnx_int8_1t['mean_ms']:.2f} ± {onnx_int8_1t['std_ms']:.2f} ms | p95: {onnx_int8_1t['p95_ms']:.2f} ms | RTF: {rtf_int8_1t:.4f}")

        # ONNX INT8 (4 Threads)
        onnx_int8_4t = benchmark_onnxruntime(int8_path, x_input, num_threads=4)
        rtf_int8_4t = (onnx_int8_4t["mean_ms"] / 1000.0) / duration_sec
        print(f"  • ONNX INT8 (4T):    {onnx_int8_4t['mean_ms']:.2f} ± {onnx_int8_4t['std_ms']:.2f} ms | p95: {onnx_int8_4t['p95_ms']:.2f} ms | RTF: {rtf_int8_4t:.4f}")

        benchmark_rows.append({
            "scenario": label,
            "duration_sec": duration_sec,
            "pytorch_fp32_mean_ms": pt_res["mean_ms"],
            "pytorch_fp32_p95_ms": pt_res["p95_ms"],
            "pytorch_fp32_rtf": rtf_pt,
            "onnx_fp32_mean_ms": onnx_fp32_res["mean_ms"],
            "onnx_fp32_p95_ms": onnx_fp32_res["p95_ms"],
            "onnx_fp32_rtf": rtf_fp32,
            "onnx_int8_1t_mean_ms": onnx_int8_1t["mean_ms"],
            "onnx_int8_1t_p95_ms": onnx_int8_1t["p95_ms"],
            "onnx_int8_1t_rtf": rtf_int8_1t,
            "onnx_int8_4t_mean_ms": onnx_int8_4t["mean_ms"],
            "onnx_int8_4t_p95_ms": onnx_int8_4t["p95_ms"],
            "onnx_int8_4t_rtf": rtf_int8_4t,
        })

    # Save to CSV
    df = pd.DataFrame(benchmark_rows)
    df.to_csv("results/cpu_benchmark_results.csv", index=False)
    print("\n" + "=" * 70)
    print("✅ Benchmark results saved to results/cpu_benchmark_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
