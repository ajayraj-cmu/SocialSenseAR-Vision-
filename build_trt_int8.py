"""Build INT8 TensorRT engine for SAM3 vision encoder.

Uses IInt8EntropyCalibrator2 with real camera frames captured to calibration_frames/.
Preprocessing matches sam3_segmenter._preprocess_fast() exactly.

Usage:
    python build_trt_int8.py                          # Build INT8 vision engine
    python build_trt_int8.py --calib-frames 500       # Use more calibration frames
    python build_trt_int8.py --onnx sam3_vision.onnx  # Specify ONNX path
"""

import os
import sys
import glob
import time
import numpy as np
import cv2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def setup_trt_path():
    """Add tensorrt_libs to PATH for nvinfer DLLs."""
    import importlib.util
    spec = importlib.util.find_spec('tensorrt')
    if spec and spec.origin:
        sp_dir = os.path.dirname(os.path.dirname(spec.origin))
        trt_libs = os.path.join(sp_dir, 'tensorrt_libs')
        if os.path.isdir(trt_libs):
            os.environ['PATH'] = trt_libs + ';' + os.environ.get('PATH', '')
            print(f"Added to PATH: {trt_libs}")


def preprocess_frame(frame_bgr, target_h=1008, target_w=1008):
    """Match sam3_segmenter._preprocess_fast() — returns FP32 numpy [1,3,H,W].

    INT8 calibration requires FP32 inputs (TRT quantizes internally).
    """
    resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # (H, W, 3) uint8 -> (1, 3, H, W) float32
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]

    # Rescale 0-255 -> 0-1
    tensor /= 255.0

    # Normalize: mean=0.5, std=0.5 -> range [-1, 1]
    tensor = (tensor - 0.5) / 0.5

    return np.ascontiguousarray(tensor)


def _get_calibrator_base():
    """Import TRT and return IInt8EntropyCalibrator2 base class."""
    setup_trt_path()
    import tensorrt as trt
    return trt.IInt8EntropyCalibrator2


class Sam3VisionCalibrator(_get_calibrator_base()):
    """INT8 entropy calibrator for SAM3 vision encoder.

    Uses torch.cuda for GPU memory (avoids pycuda dependency).
    """

    def __init__(self, calib_dir, max_frames=300, cache_file="sam3_vision_int8.cache"):
        super().__init__()
        import torch

        self.cache_file = os.path.join(PROJECT_ROOT, cache_file)

        # Load calibration frame paths
        patterns = [os.path.join(calib_dir, f"*.{ext}") for ext in ("png", "jpg", "jpeg")]
        all_frames = sorted(sum((glob.glob(p) for p in patterns), []))
        self.frame_paths = all_frames[:max_frames]
        print(f"Calibrator: {len(self.frame_paths)} frames from {calib_dir}")

        if len(self.frame_paths) == 0:
            raise RuntimeError(f"No calibration frames found in {calib_dir}")

        # Preprocess first frame to get shape
        sample = preprocess_frame(cv2.imread(self.frame_paths[0]))
        self.shape = sample.shape  # (1, 3, 1008, 1008)
        self.dtype = np.float32
        self.batch_size = 1
        self.current_index = 0

        # Allocate GPU memory using torch (returns stable data_ptr)
        self.device_buffer = torch.zeros(*self.shape, dtype=torch.float32, device="cuda")
        print(f"Calibrator: input shape {self.shape}, "
              f"{self.device_buffer.nelement() * 4 / 1024 / 1024:.1f} MB per batch")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Feed one preprocessed frame to TRT for calibration."""
        import torch

        if self.current_index >= len(self.frame_paths):
            return None

        path = self.frame_paths[self.current_index]
        frame = cv2.imread(path)
        if frame is None:
            print(f"  WARNING: Could not read {path}, skipping")
            self.current_index += 1
            return self.get_batch(names)

        data = preprocess_frame(frame)
        # Copy numpy -> torch GPU tensor
        self.device_buffer.copy_(torch.from_numpy(data).cuda())

        self.current_index += 1
        if self.current_index % 50 == 0:
            print(f"  Calibrating... {self.current_index}/{len(self.frame_paths)}")

        return [self.device_buffer.data_ptr()]

    def read_calibration_cache(self):
        """Read cached calibration data if available (speeds up repeated builds)."""
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Save calibration data for reuse."""
        print(f"Writing calibration cache: {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def build_int8_engine(onnx_path, engine_path, calib_dir, max_calib_frames=300):
    """Build INT8+FP16 TRT engine for SAM3 vision encoder."""
    setup_trt_path()
    import tensorrt as trt

    print(f"TensorRT version: {trt.__version__}")
    print(f"Input ONNX: {onnx_path}")
    print(f"Output engine: {engine_path}")
    print(f"Precision: INT8 + FP16 mixed")
    print()

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print("Parsing ONNX model...")
    t0 = time.perf_counter()
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(i)}")
            sys.exit(1)
    print(f"Parsed in {(time.perf_counter() - t0)*1000:.0f}ms")

    print(f"\nNetwork inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {inp.name}: {inp.shape} {inp.dtype}")
    print(f"Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {out.name}: {out.shape} {out.dtype}")

    # Builder config: INT8 + FP16 mixed precision
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)  # 4GB workspace

    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)

    # Create calibrator
    calibrator = Sam3VisionCalibrator(
        calib_dir=calib_dir,
        max_frames=max_calib_frames,
        cache_file="sam3_vision_int8.cache"
    )
    config.int8_calibrator = calibrator

    # Static batch profile (vision encoder is always batch=1)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        # Replace dynamic dims (-1) with 1
        shape = [1 if d == -1 else d for d in shape]
        profile.set_shape(inp.name, shape, shape, shape)
        print(f"  Profile {inp.name}: {shape}")
    config.add_optimization_profile(profile)

    print()
    print("Building INT8 TensorRT engine (this may take 10-20 minutes)...")
    print("  Calibrating with real camera frames for accurate quantization...")
    t0 = time.perf_counter()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.perf_counter() - t0

    if engine_bytes is None:
        print("ERROR: Engine build failed!")
        sys.exit(1)

    engine_size = engine_bytes.nbytes
    print(f"\nEngine built in {build_time:.1f}s ({engine_size / 1024 / 1024:.1f} MB)")

    with open(engine_path, 'wb') as f:
        f.write(memoryview(engine_bytes))
    print(f"Saved to: {engine_path}")

    # Validate
    print("\nValidating engine...")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = list(engine.get_tensor_shape(name))
            context.set_input_shape(name, shape)

    print(f"  Context memory: {engine.device_memory_size / 1024 / 1024:.0f}MB")
    print("  Validation OK")
    print(f"\nDone! INT8 engine: {engine_path}")

    return engine_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build INT8 TRT engine for SAM3 vision")
    parser.add_argument("--onnx", default=os.path.join(PROJECT_ROOT, "sam3_vision.onnx"),
                        help="Path to vision ONNX model")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "sam3_vision_int8.engine"),
                        help="Output engine path")
    parser.add_argument("--calib-dir", default=os.path.join(PROJECT_ROOT, "calibration_frames"),
                        help="Directory with calibration PNG/JPEG frames")
    parser.add_argument("--calib-frames", type=int, default=300,
                        help="Max number of calibration frames to use")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ERROR: {args.onnx} not found. Run sam3_export.py first.")
        sys.exit(1)

    if not os.path.isdir(args.calib_dir):
        print(f"ERROR: Calibration directory {args.calib_dir} not found.")
        print("Capture frames first: python capture_calib_frames.py")
        sys.exit(1)

    build_int8_engine(args.onnx, args.output, args.calib_dir, args.calib_frames)


if __name__ == "__main__":
    main()
