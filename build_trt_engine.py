"""Build TensorRT engine from ONNX decoder model.

Run this standalone (no PyTorch in VRAM) to build the engine with full GPU memory.
The engine is saved and loaded by sam3_segmenter.py at runtime.

Usage:
    python build_trt_engine.py                    # Build from sam3_topk_decoder.onnx (preferred)
    python build_trt_engine.py sam3_decoder.onnx  # Build from specific ONNX file
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def build_engine(onnx_path, engine_path):
    """Build TRT FP16 engine from ONNX model with dynamic batch 1-4."""
    # Add tensorrt_libs to PATH for nvinfer DLLs
    import importlib.util
    spec = importlib.util.find_spec('tensorrt')
    if spec and spec.origin:
        sp_dir = os.path.dirname(os.path.dirname(spec.origin))
        trt_libs = os.path.join(sp_dir, 'tensorrt_libs')
        if os.path.isdir(trt_libs):
            os.environ['PATH'] = trt_libs + ';' + os.environ.get('PATH', '')
            print(f"Added to PATH: {trt_libs}")

    import tensorrt as trt

    print(f"TensorRT version: {trt.__version__}")
    print(f"Input ONNX: {onnx_path}")
    print(f"Output engine: {engine_path}")
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

    print(f"Network inputs: {network.num_inputs}")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {inp.name}: {inp.shape} {inp.dtype}")
    print(f"Network outputs: {network.num_outputs}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {out.name}: {out.shape} {out.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Dynamic batch profiles
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        shape = list(inp.shape)
        min_shape = shape.copy()
        opt_shape = shape.copy()
        max_shape = shape.copy()
        if shape[0] == -1:
            min_shape[0] = 1
            opt_shape[0] = 1  # optimize for batch=1 (sequential per-prompt)
            max_shape[0] = 4
        profile.set_shape(name, min_shape, opt_shape, max_shape)
        print(f"  Profile {name}: min={min_shape} opt={opt_shape} max={max_shape}")

    config.add_optimization_profile(profile)

    print()
    print("Building TensorRT engine (this may take several minutes)...")
    t0 = time.perf_counter()
    engine_bytes = builder.build_serialized_network(network, config)
    build_time = time.perf_counter() - t0

    if engine_bytes is None:
        print("ERROR: Engine build failed!")
        sys.exit(1)

    engine_size = engine_bytes.nbytes
    print(f"Engine built in {build_time:.1f}s ({engine_size / 1024 / 1024:.1f} MB)")

    with open(engine_path, 'wb') as f:
        f.write(memoryview(engine_bytes))
    print(f"Saved to: {engine_path}")

    # Validate
    print()
    print("Validating engine...")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape = list(engine.get_tensor_shape(name))
            if shape[0] == -1:
                shape[0] = 1
            context.set_input_shape(name, shape)

    print(f"  Context memory: {engine.device_memory_size / 1024 / 1024:.0f}MB")
    print("  Validation OK")
    print()
    print("Done!")


def main():
    # Default: build from top-k decoder (preferred), fall back to full decoder
    if len(sys.argv) > 1:
        onnx_name = sys.argv[1]
    else:
        # Prefer top-k decoder
        topk = os.path.join(PROJECT_ROOT, "sam3_topk_decoder.onnx")
        full = os.path.join(PROJECT_ROOT, "sam3_decoder.onnx")
        if os.path.exists(topk):
            onnx_name = "sam3_topk_decoder.onnx"
        elif os.path.exists(full):
            onnx_name = "sam3_decoder.onnx"
        else:
            print("ERROR: No decoder ONNX found. Run sam3_export.py first.")
            sys.exit(1)

    onnx_path = os.path.join(PROJECT_ROOT, onnx_name)
    engine_name = onnx_name.replace(".onnx", ".engine")
    engine_path = os.path.join(PROJECT_ROOT, engine_name)

    if not os.path.exists(onnx_path):
        print(f"ERROR: {onnx_path} not found.")
        sys.exit(1)

    build_engine(onnx_path, engine_path)


if __name__ == "__main__":
    main()
