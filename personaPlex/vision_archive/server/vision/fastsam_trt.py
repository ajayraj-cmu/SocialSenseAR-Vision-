"""Direct TensorRT inference for FastSAM — bypasses ultralytics overhead entirely.

Uses torch CUDA tensors for buffer management (no pycuda needed).
Releases GIL during TRT inference via torch.cuda operations.

Performance on RTX 3060 Laptop:
- Preprocess: ~2ms (letterbox on CPU, normalize on GPU)
- Inference:  ~3ms (TRT FP16)
- Postproc:   ~2ms (NMS + mask decode on GPU)
- Total:      ~7ms = 140+ FPS
"""

import struct
import logging
import numpy as np
import cv2
import torch
import torchvision

logger = logging.getLogger(__name__)


class FastSAMTRT:
    """Direct TensorRT FastSAM inference using torch CUDA tensors."""

    def __init__(self, engine_path: str, imgsz: int = 512):
        import tensorrt as trt

        self.imgsz = imgsz
        self._model_path = engine_path

        # Load TRT engine (ultralytics wraps engine with JSON header)
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            header_len = struct.unpack('<I', f.read(4))[0]
            if header_len < 10000:
                f.seek(4 + header_len)
                engine_data = f.read()
                logger.info(f"Skipped {4 + header_len} byte ultralytics header")
            else:
                f.seek(0)
                engine_data = f.read()

        runtime = trt.Runtime(trt_logger)
        self._engine = runtime.deserialize_cuda_engine(engine_data)
        self._context = self._engine.create_execution_context()

        # Allocate torch CUDA tensors for I/O (pre-allocated, reused every frame)
        self._input_tensor = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device='cuda')
        self._output0 = torch.zeros((1, 37, 5376), dtype=torch.float32, device='cuda')
        self._output1 = torch.zeros((1, 32, 128, 128), dtype=torch.float32, device='cuda')

        # Set tensor addresses for TRT
        self._context.set_tensor_address('images', self._input_tensor.data_ptr())
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self._engine.get_tensor_shape(name)
                if len(shape) == 3 and shape[2] == 5376:
                    self._context.set_tensor_address(name, self._output0.data_ptr())
                elif len(shape) == 4 and shape[1] == 32:
                    self._context.set_tensor_address(name, self._output1.data_ptr())
                else:
                    size = 1
                    for s in shape:
                        size *= s
                    buf = torch.zeros(size, dtype=torch.float32, device='cuda')
                    self._context.set_tensor_address(name, buf.data_ptr())

        self._stream = torch.cuda.Stream(priority=-1)  # High priority — reduce DWM contention

        # Cache letterbox params
        self._cached_shape = None
        self._cached_scale = 1.0
        self._cached_pad = (0, 0)
        self._cached_newsize = (0, 0)

        # Pre-allocated pinned memory for upload — numpy view writes directly to it
        self._upload_tensor = torch.empty((imgsz, imgsz, 3), dtype=torch.uint8, device='cpu').pin_memory()
        self._upload_numpy = self._upload_tensor.numpy()  # numpy view of pinned memory (zero-copy)

        logger.info(f"TRT engine loaded: {engine_path}")

        # Warmup
        for _ in range(5):
            self._infer()
        torch.cuda.synchronize()
        logger.info("TRT warmup complete")

    def _letterbox_cpu(self, frame_bgr: np.ndarray):
        """CPU-only letterbox: resize + pad + BGR→RGB directly into pinned memory.

        Writes directly to self._upload_numpy (pinned memory view) to eliminate
        intermediate buffer copies. Saves ~0.5ms vs the old double-copy path.
        """
        h, w = frame_bgr.shape[:2]

        if (h, w) != self._cached_shape:
            scale = min(self.imgsz / h, self.imgsz / w)
            new_w, new_h = int(w * scale), int(h * scale)
            pad_w = (self.imgsz - new_w) // 2
            pad_h = (self.imgsz - new_h) // 2
            self._cached_shape = (h, w)
            self._cached_scale = scale
            self._cached_pad = (pad_w, pad_h)
            self._cached_newsize = (new_w, new_h)

        new_w, new_h = self._cached_newsize
        pad_w, pad_h = self._cached_pad

        # Resize frame (CPU — cv2 is very fast for this)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # BGR→RGB on the smaller resized image (SIMD-optimized, 331KB vs 786KB)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Write directly to pinned memory (one copy, no intermediate buffer)
        self._upload_numpy[:] = 114  # padding fill
        self._upload_numpy[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_rgb

    def _gpu_preprocess(self):
        """GPU upload + normalize + reshape. Must run on the inference stream."""
        gpu_u8 = self._upload_tensor.cuda(non_blocking=True)  # (512, 512, 3) uint8
        gpu_f32 = gpu_u8.float().div_(255.0)  # normalize on GPU
        # HWC→CHW and add batch dimension
        self._input_tensor[0] = gpu_f32.permute(2, 0, 1)

    def _infer(self):
        """Run TRT inference. GPU execution releases GIL."""
        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

    def _postprocess_masks_gpu(self, conf_thres=0.25, iou_thres=0.55):
        """NMS + mask decode on GPU. Returns torch tensor (N, 128, 128) on CUDA, or None."""
        pred = self._output0[0].T  # (5376, 37) on GPU
        protos = self._output1[0]  # (32, 128, 128) on GPU

        scores = pred[:, 4]  # (5376,)

        # Confidence filter (on GPU)
        keep = scores > conf_thres
        if not keep.any():
            return None

        pred_kept = pred[keep]  # (N_conf, 37)
        boxes_cxcywh = pred_kept[:, :4]
        scores_kept = pred_kept[:, 4]
        mask_coeff = pred_kept[:, 5:37]  # (N_conf, 32)

        # Convert cxcywh → xyxy for torchvision NMS
        boxes_xyxy = torch.empty_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # NMS on GPU (torchvision.ops.nms)
        nms_indices = torchvision.ops.nms(boxes_xyxy, scores_kept, iou_thres)

        if len(nms_indices) == 0:
            return None

        # Decode masks on GPU: coefficients @ protos → sigmoid
        kept_coeff = mask_coeff[nms_indices]  # (N, 32)
        protos_flat = protos.reshape(32, -1)  # (32, 16384)
        masks_raw = kept_coeff @ protos_flat  # (N, 16384) — matrix multiply on GPU
        masks_sigmoid = torch.sigmoid(masks_raw)  # sigmoid on GPU
        return masks_sigmoid.reshape(-1, 128, 128)

    def _postprocess_gpu(self, conf_thres=0.25, iou_thres=0.55):
        """NMS + mask decode using torch on GPU. Returns masks (N, 128, 128) numpy float32."""
        masks = self._postprocess_masks_gpu(conf_thres, iou_thres)
        if masks is None:
            return np.empty((0, 128, 128), dtype=np.float32)
        return masks.cpu().numpy()

    @torch.no_grad()
    def process_full(self, frame_bgr, h, w, used_pixels, conf=0.25, max_masks=3, min_area=1500):
        """Full pipeline: infer + resize + filter entirely on GPU.

        Returns list of (mask_u8_255, center, bbox) tuples.
        Pre-filters at 128x128 to avoid resizing all NMS survivors to full res.
        """
        # CPU-only prep: resize + pad + BGR→RGB into pinned memory
        self._letterbox_cpu(frame_bgr)
        used_np = used_pixels.astype(np.uint8)

        # ALL GPU work on ONE stream
        with torch.cuda.stream(self._stream):
            self._gpu_preprocess()
            self._context.execute_async_v3(self._stream.cuda_stream)

            masks_gpu = self._postprocess_masks_gpu(conf_thres=conf)
            if masks_gpu is None:
                return []

            n = masks_gpu.shape[0]

            # --- Pre-filter at 128x128 (cheap) before expensive full-res resize ---
            # Downscale used_pixels to mask resolution
            used_gpu_full = torch.from_numpy(used_np).bool().cuda(non_blocking=True)
            used_small = torch.nn.functional.interpolate(
                used_gpu_full.float().unsqueeze(0).unsqueeze(0),
                size=(128, 128), mode='nearest'
            ).squeeze() > 0.5  # (128, 128) bool

            masks_bin_small = (masks_gpu > 0.5)  # (N, 128, 128)
            clean_small = masks_bin_small & ~used_small.unsqueeze(0)

            # Area at 128x128 — scale min_area proportionally
            min_area_small = max(1, int(min_area * 128 * 128 / (h * w)))
            areas_small = clean_small.reshape(n, -1).sum(dim=1)

            valid = areas_small >= min_area_small
            if not valid.any():
                return []

            valid_idx = valid.nonzero(as_tuple=True)[0]
            areas_valid = areas_small[valid_idx]
            k = min(max_masks, len(valid_idx))
            _, topk_local = areas_valid.topk(k)
            selected_idx = valid_idx[topk_local]

            # --- Only resize selected k masks to full resolution ---
            selected_masks = masks_gpu[selected_idx]  # (k, 128, 128)

            # Crop out letterbox padding before resizing to original frame dims.
            # 128x128 masks correspond to the 512x512 letterboxed image — the
            # actual frame content sits inside (pad_h..pad_h+new_h, pad_w..pad_w+new_w).
            pad_w, pad_h_lb = self._cached_pad
            new_w_lb, new_h_lb = self._cached_newsize
            mp_h = int(pad_h_lb * 128 / self.imgsz)
            mp_w = int(pad_w * 128 / self.imgsz)
            mc_h = max(1, int(new_h_lb * 128 / self.imgsz))
            mc_w = max(1, int(new_w_lb * 128 / self.imgsz))
            selected_cropped = selected_masks[:, mp_h:mp_h+mc_h, mp_w:mp_w+mc_w]

            # Bilinear upscale cropped content to original frame size
            masks_full = torch.nn.functional.interpolate(
                selected_cropped.unsqueeze(1), size=(h, w), mode='bilinear',
                align_corners=False,
            ).squeeze(1)  # (k, h, w) float32 with smooth edges

            # Threshold to binary
            masks_bin = (masks_full > 0.5).float()

            # GPU morphological close — fills small holes in masks
            masks_4d = masks_bin.unsqueeze(1)  # (k, 1, h, w)
            masks_4d = torch.nn.functional.max_pool2d(masks_4d, 5, stride=1, padding=2)
            masks_4d = -torch.nn.functional.max_pool2d(-masks_4d, 5, stride=1, padding=2)
            masks_bin = masks_4d.squeeze(1)  # (k, h, w)

            # Clean: remove body/person regions
            clean = (masks_bin > 0.5) & ~used_gpu_full.unsqueeze(0)

            # Copy to CPU
            masks_np = (clean.byte() * 255).cpu().numpy()  # (k, h, w) uint8

        # CPU-only: deduplicate, compute centers and bboxes
        # Process largest-first (already sorted by area). Each mask claims its
        # pixels so subsequent masks can't overlap — prevents multiple segments
        # fighting over the same object.
        claimed = np.zeros((h, w), dtype=np.uint8)
        results = []
        for i in range(k):
            m = masks_np[i]
            # Remove pixels already claimed by a previous (larger) mask
            m = m & ~claimed
            area = cv2.countNonZero(m)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(m)
            if bw == 0 or bh == 0:
                continue
            # Skip thin strips spanning nearly the full frame (letterbox/edge artifacts)
            aspect = max(bw, bh) / min(bw, bh)
            if aspect > 8 and (bw > w * 0.8 or bh > h * 0.8):
                continue
            # Claim these pixels
            claimed |= m
            M = cv2.moments(m)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] > 0 else (0, 0)
            results.append((m, center, (x, y, bw, bh)))

        return results

    def __call__(self, frame_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.7):
        """Full inference pipeline. Returns (N, 128, 128) float32 masks."""
        self._letterbox_cpu(frame_bgr)
        self._gpu_preprocess()
        self._infer()
        return self._postprocess_gpu(conf_thres=conf, iou_thres=iou)

    @property
    def model_path(self):
        return self._model_path
