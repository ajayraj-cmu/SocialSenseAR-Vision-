"""Standalone SAM FPS benchmark — no server, no client, just raw inference speed.

Usage:
    python -m server.sam_benchmark                    # baseline PyTorch
    python -m server.sam_benchmark --export trt       # export to TensorRT then bench
    python -m server.sam_benchmark --export onnx      # export to ONNX then bench
    python -m server.sam_benchmark --engine PATH      # bench a pre-exported engine
    python -m server.sam_benchmark --compile           # torch.compile() bench
    python -m server.sam_benchmark --fp16              # FP16 PyTorch bench
"""

import argparse
import time
import sys
import os
import numpy as np

def get_test_frame(width=640, height=480):
    """Generate a realistic test frame (or capture from webcam)."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"Using webcam frame: {frame.shape}")
            return frame
    except:
        pass
    # Fallback: synthetic frame with some structure
    frame = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Add some rectangles to give SAM something to segment
    import cv2
    cv2.rectangle(frame, (100, 100), (300, 350), (0, 0, 200), -1)
    cv2.rectangle(frame, (350, 50), (550, 250), (0, 180, 0), -1)
    cv2.circle(frame, (400, 350), 80, (200, 200, 0), -1)
    print(f"Using synthetic frame: {frame.shape}")
    return frame


def bench_ultralytics(model_path, frame, imgsz=512, conf=0.25, device="cuda",
                      warmup=5, iterations=50):
    """Benchmark ultralytics FastSAM model (PyTorch, TRT engine, or ONNX)."""
    from ultralytics import FastSAM

    print(f"\nLoading model: {model_path}")
    model = FastSAM(model_path)

    # Warmup
    print(f"Warming up ({warmup} iters)...")
    for _ in range(warmup):
        model(frame, device=device, retina_masks=True, imgsz=imgsz, conf=conf, verbose=False)

    # Benchmark
    print(f"Benchmarking ({iterations} iters)...")
    times = []
    for i in range(iterations):
        t0 = time.perf_counter()
        results = model(frame, device=device, retina_masks=True, imgsz=imgsz, conf=conf, verbose=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    fps = 1000.0 / np.mean(times)
    n_masks = 0
    if results and results[0].masks is not None:
        n_masks = len(results[0].masks.data)

    print(f"\n{'='*50}")
    print(f"Model: {model_path}")
    print(f"Input: {imgsz}px, conf={conf}")
    print(f"Masks found: {n_masks}")
    print(f"Mean: {np.mean(times):.1f}ms | Median: {np.median(times):.1f}ms")
    print(f"Min:  {np.min(times):.1f}ms | Max: {np.max(times):.1f}ms")
    print(f"P95:  {np.percentile(times, 95):.1f}ms | P99: {np.percentile(times, 99):.1f}ms")
    print(f"FPS:  {fps:.1f}")
    print(f"{'='*50}")
    return fps, times


def export_tensorrt(model_path="FastSAM-s.pt", imgsz=512, fp16=True):
    """Export FastSAM to TensorRT engine."""
    from ultralytics import FastSAM
    print(f"\nExporting {model_path} to TensorRT (imgsz={imgsz}, fp16={fp16})...")
    model = FastSAM(model_path)
    engine_path = model.export(format="engine", imgsz=imgsz, half=fp16, simplify=True)
    print(f"Exported to: {engine_path}")
    return engine_path


def export_onnx(model_path="FastSAM-s.pt", imgsz=512, fp16=True):
    """Export FastSAM to ONNX."""
    from ultralytics import FastSAM
    print(f"\nExporting {model_path} to ONNX (imgsz={imgsz}, half={fp16})...")
    model = FastSAM(model_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, half=fp16, simplify=True)
    print(f"Exported to: {onnx_path}")
    return onnx_path


def bench_torch_compile(model_path="FastSAM-s.pt", frame=None, imgsz=512, conf=0.25,
                        warmup=5, iterations=50):
    """Benchmark with torch.compile() applied to the model."""
    import torch
    from ultralytics import FastSAM

    print(f"\nLoading model with torch.compile(): {model_path}")
    model = FastSAM(model_path)

    # Apply torch.compile to the underlying model
    try:
        model.model = torch.compile(model.model, mode="reduce-overhead", backend="inductor")
        print("torch.compile() applied successfully (reduce-overhead mode)")
    except Exception as e:
        print(f"torch.compile() failed: {e}")
        print("Trying default mode...")
        try:
            model.model = torch.compile(model.model)
            print("torch.compile() applied (default mode)")
        except Exception as e2:
            print(f"torch.compile() failed completely: {e2}")
            return 0, []

    # Warmup (compile happens on first few runs)
    print(f"Warming up ({warmup} iters, compilation happens here)...")
    for i in range(warmup):
        t0 = time.perf_counter()
        model(frame, device="cuda", retina_masks=True, imgsz=imgsz, conf=conf, verbose=False)
        print(f"  warmup {i+1}: {(time.perf_counter()-t0)*1000:.0f}ms")

    # Benchmark
    print(f"Benchmarking ({iterations} iters)...")
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        results = model(frame, device="cuda", retina_masks=True, imgsz=imgsz, conf=conf, verbose=False)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    fps = 1000.0 / np.mean(times)
    print(f"\n{'='*50}")
    print(f"torch.compile() results:")
    print(f"Mean: {np.mean(times):.1f}ms | FPS: {fps:.1f}")
    print(f"{'='*50}")
    return fps, times


def main():
    parser = argparse.ArgumentParser(description="SAM FPS Benchmark")
    parser.add_argument("--model", default="FastSAM-s.pt", help="Model path")
    parser.add_argument("--export", choices=["trt", "onnx"], help="Export format")
    parser.add_argument("--engine", type=str, help="Pre-exported engine path")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument("--fp16", action="store_true", help="FP16 mode")
    parser.add_argument("--imgsz", type=int, default=512, help="Input size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    args = parser.parse_args()

    frame = get_test_frame()

    # Baseline first
    print("\n" + "="*60)
    print("BASELINE: PyTorch FastSAM-s.pt")
    print("="*60)
    baseline_fps, _ = bench_ultralytics(
        args.model, frame, imgsz=args.imgsz, conf=args.conf,
        warmup=args.warmup, iterations=args.iters
    )

    if args.export == "trt":
        engine_path = export_tensorrt(args.model, imgsz=args.imgsz, fp16=True)
        print("\n" + "="*60)
        print("TensorRT ENGINE")
        print("="*60)
        trt_fps, _ = bench_ultralytics(
            engine_path, frame, imgsz=args.imgsz, conf=args.conf,
            warmup=args.warmup, iterations=args.iters
        )
        print(f"\nSpeedup: {trt_fps/baseline_fps:.2f}x")

    elif args.export == "onnx":
        onnx_path = export_onnx(args.model, imgsz=args.imgsz, fp16=False)
        print("\n" + "="*60)
        print("ONNX Runtime")
        print("="*60)
        onnx_fps, _ = bench_ultralytics(
            onnx_path, frame, imgsz=args.imgsz, conf=args.conf,
            warmup=args.warmup, iterations=args.iters
        )
        print(f"\nSpeedup: {onnx_fps/baseline_fps:.2f}x")

    elif args.engine:
        print("\n" + "="*60)
        print(f"PRE-EXPORTED ENGINE: {args.engine}")
        print("="*60)
        eng_fps, _ = bench_ultralytics(
            args.engine, frame, imgsz=args.imgsz, conf=args.conf,
            warmup=args.warmup, iterations=args.iters
        )
        print(f"\nSpeedup: {eng_fps/baseline_fps:.2f}x")

    elif args.compile:
        print("\n" + "="*60)
        print("torch.compile()")
        print("="*60)
        compile_fps, _ = bench_torch_compile(
            args.model, frame, imgsz=args.imgsz, conf=args.conf,
            warmup=10, iterations=args.iters
        )
        print(f"\nSpeedup: {compile_fps/baseline_fps:.2f}x")

    elif args.fp16:
        import torch
        print("\nNote: FP16 is handled via --export trt --fp16 for best results")


if __name__ == "__main__":
    main()
