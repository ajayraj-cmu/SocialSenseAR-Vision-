"""
GPU kernels for accelerated effect processing.

Uses CuPy for CUDA acceleration when available.
Falls back to CPU implementations when CuPy is not installed.
"""
import numpy as np

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# GPU kernels (compiled once at module load)
_steady_blend_kernel = None
_transition_blend_kernel = None

if CUPY_AVAILABLE:
    # Kernel for steady-state effect: blends color/grayscale based on left/right masks
    _steady_blend_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void grayscale_mask_blend_split(
        const unsigned char* __restrict__ frame_rgb,
        const unsigned char* __restrict__ left_mask,
        const unsigned char* __restrict__ right_mask,
        unsigned char* __restrict__ output,
        int height, int half_width
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        int total = height * half_width * 2;
        if (idx >= total) return;

        int row = idx / (half_width * 2);
        int col = idx % (half_width * 2);

        // Determine which mask to use based on column
        unsigned char mask_val;
        if (col < half_width) {
            mask_val = left_mask[row * half_width + col];
        } else {
            mask_val = right_mask[row * half_width + (col - half_width)];
        }

        int rgb_idx = idx * 3;
        unsigned char r = frame_rgb[rgb_idx];
        unsigned char g = frame_rgb[rgb_idx + 1];
        unsigned char b = frame_rgb[rgb_idx + 2];

        // Fast grayscale approximation
        unsigned char gray = (unsigned char)((r * 77 + g * 150 + b * 29) >> 8);

        float m = mask_val * 0.00392156862f;  // 1/255
        float inv_m = 1.0f - m;

        // Output as BGR
        output[rgb_idx] = (unsigned char)(b * m + gray * inv_m);
        output[rgb_idx + 1] = (unsigned char)(g * m + gray * inv_m);
        output[rgb_idx + 2] = (unsigned char)(r * m + gray * inv_m);
    }
    ''', 'grayscale_mask_blend_split')

    # Kernel for transition: RGB in, computes gray internally, BGR out
    _transition_blend_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void transition_blend(
        const unsigned char* __restrict__ rgb_in,
        const unsigned char* __restrict__ weight,
        unsigned char* __restrict__ bgr_out,
        int total_pixels
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= total_pixels) return;

        int rgb_idx = idx * 3;
        unsigned char r = rgb_in[rgb_idx];
        unsigned char g = rgb_in[rgb_idx + 1];
        unsigned char b = rgb_in[rgb_idx + 2];

        // Compute grayscale in kernel
        unsigned char gray = (unsigned char)((r * 77 + g * 150 + b * 29) >> 8);

        unsigned int wt = weight[idx];
        unsigned int inv_wt = 255 - wt;

        // Output as BGR
        bgr_out[rgb_idx] = (unsigned char)((b * wt + gray * inv_wt) >> 8);
        bgr_out[rgb_idx + 1] = (unsigned char)((g * wt + gray * inv_wt) >> 8);
        bgr_out[rgb_idx + 2] = (unsigned char)((r * wt + gray * inv_wt) >> 8);
    }
    ''', 'transition_blend')

    print("CuPy CUDA kernels loaded!")


class GPUBuffers:
    """Preallocated GPU buffers for effect processing.

    Reuses buffers across frames to avoid allocation overhead.
    """

    def __init__(self):
        self.frame = None
        self.left_mask = None
        self.right_mask = None
        self.output = None
        self.output_cpu = None
        self.dims = (0, 0)

        # Transition-specific buffers
        self.trans_frame = None
        self.trans_weight = None
        self.trans_output = None
        self.trans_output_cpu = None
        self.trans_dims = (0, 0)

    def ensure_steady_buffers(self, h: int, w: int) -> None:
        """Ensure steady-state buffers are allocated."""
        if not CUPY_AVAILABLE:
            return

        if self.dims != (h, w):
            half_w = w // 2
            self.frame = cp.empty((h, w, 3), dtype=cp.uint8)
            self.left_mask = cp.empty((h, half_w), dtype=cp.uint8)
            self.right_mask = cp.empty((h, half_w), dtype=cp.uint8)
            self.output = cp.empty((h, w, 3), dtype=cp.uint8)
            self.output_cpu = np.empty((h, w, 3), dtype=np.uint8)
            self.dims = (h, w)

    def ensure_transition_buffers(self, h: int, w: int) -> None:
        """Ensure transition buffers are allocated."""
        if not CUPY_AVAILABLE:
            return

        if self.trans_dims != (h, w):
            self.trans_frame = cp.empty((h, w, 3), dtype=cp.uint8)
            self.trans_weight = cp.empty((h, w), dtype=cp.uint8)
            self.trans_output = cp.empty((h, w, 3), dtype=cp.uint8)
            self.trans_output_cpu = np.empty((h, w, 3), dtype=np.uint8)
            self.trans_dims = (h, w)


# Global buffer instance
_buffers = GPUBuffers()


def apply_steady_gpu(frame_rgb: np.ndarray, left_mask: np.ndarray,
                     right_mask: np.ndarray) -> np.ndarray:
    """Apply steady-state effect using GPU.

    Args:
        frame_rgb: RGB input frame
        left_mask: Left eye mask (H, half_W)
        right_mask: Right eye mask (H, half_W)

    Returns:
        BGR output frame
    """
    import cv2

    h, w = frame_rgb.shape[:2]
    half_w = w // 2

    # Resize masks if needed
    if left_mask is None:
        left_mask = np.zeros((h, half_w), dtype=np.uint8)
    elif left_mask.shape[:2] != (h, half_w):
        left_mask = cv2.resize(left_mask, (half_w, h), interpolation=cv2.INTER_LINEAR)

    if right_mask is None:
        right_mask = np.zeros((h, half_w), dtype=np.uint8)
    elif right_mask.shape[:2] != (h, half_w):
        right_mask = cv2.resize(right_mask, (half_w, h), interpolation=cv2.INTER_LINEAR)

    if CUPY_AVAILABLE:
        _buffers.ensure_steady_buffers(h, w)

        # Transfer to GPU
        if frame_rgb.flags['C_CONTIGUOUS']:
            _buffers.frame.set(frame_rgb)
        else:
            _buffers.frame.set(np.ascontiguousarray(frame_rgb))

        if left_mask.flags['C_CONTIGUOUS']:
            _buffers.left_mask.set(left_mask)
        else:
            _buffers.left_mask.set(np.ascontiguousarray(left_mask))

        if right_mask.flags['C_CONTIGUOUS']:
            _buffers.right_mask.set(right_mask)
        else:
            _buffers.right_mask.set(np.ascontiguousarray(right_mask))

        # Run kernel
        threads = 256
        blocks = (h * w + threads - 1) // threads
        _steady_blend_kernel(
            (blocks,), (threads,),
            (_buffers.frame, _buffers.left_mask, _buffers.right_mask, _buffers.output, h, half_w)
        )

        # Get result
        _buffers.output.get(out=_buffers.output_cpu)
        return _buffers.output_cpu

    else:
        # CPU fallback using PyTorch
        import torch
        with torch.no_grad():
            frame_t = torch.from_numpy(np.ascontiguousarray(frame_rgb)).cuda().float() / 255.0
            frame_t = frame_t.permute(2, 0, 1)
            gray_t = 0.299 * frame_t[0] + 0.587 * frame_t[1] + 0.114 * frame_t[2]
            full_mask = np.hstack([left_mask, right_mask])
            mask_t = torch.from_numpy(full_mask).cuda().float() / 255.0
            gray_3ch = gray_t.unsqueeze(0).expand(3, -1, -1)
            mask_3ch = mask_t.unsqueeze(0).expand(3, -1, -1)
            result_t = torch.lerp(gray_3ch, frame_t, mask_3ch)
            result_t = torch.stack([result_t[2], result_t[1], result_t[0]], dim=0)
            return (result_t * 255).byte().permute(1, 2, 0).cpu().numpy()


def apply_transition_gpu(frame_rgb: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Apply transition effect using GPU.

    Args:
        frame_rgb: RGB input frame
        weight: Blend weight mask (0=gray, 255=color)

    Returns:
        BGR output frame
    """
    import cv2

    h, w = frame_rgb.shape[:2]

    if CUPY_AVAILABLE:
        _buffers.ensure_transition_buffers(h, w)

        # Transfer to GPU
        if frame_rgb.flags['C_CONTIGUOUS']:
            _buffers.trans_frame.set(frame_rgb)
        else:
            _buffers.trans_frame.set(np.ascontiguousarray(frame_rgb))

        if weight.flags['C_CONTIGUOUS']:
            _buffers.trans_weight.set(weight)
        else:
            _buffers.trans_weight.set(np.ascontiguousarray(weight))

        # Run kernel
        total_pixels = h * w
        threads = 256
        blocks = (total_pixels + threads - 1) // threads
        _transition_blend_kernel(
            (blocks,), (threads,),
            (_buffers.trans_frame, _buffers.trans_weight, _buffers.trans_output, total_pixels)
        )

        # Get result
        _buffers.trans_output.get(out=_buffers.trans_output_cpu)
        return _buffers.trans_output_cpu

    else:
        # CPU fallback
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        wt = weight[:, :, np.newaxis].astype(np.uint16)
        inv_wt = 255 - wt
        return ((frame_bgr.astype(np.uint16) * wt + gray_3ch.astype(np.uint16) * inv_wt) >> 8).astype(np.uint8)
