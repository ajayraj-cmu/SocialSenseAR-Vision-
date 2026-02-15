"""Run-length encoding for binary segmentation masks.

Encodes binary masks into compact byte sequences for network transmission.
Format: sequence of uint16 little-endian run lengths, starting with a 0-run.

Typical compression: 300KB raw mask -> 200-2000 bytes RLE.
"""
import numpy as np


def encode_rle(mask: np.ndarray) -> bytes:
    """Encode a binary mask to RLE bytes (vectorized — no Python pixel loops).

    Args:
        mask: 2D numpy array (any dtype). Values > 0 are foreground.

    Returns:
        RLE-encoded bytes. Format: [uint16 run_length, ...] starting with 0-count.
    """
    # Fast path: view as contiguous uint8 to avoid float comparisons
    flat = mask.ravel()
    if flat.dtype != np.uint8:
        flat = (flat > 0.5).view(np.uint8) if flat.dtype == np.bool_ else (flat > 0).astype(np.uint8)
    n = flat.size

    if n == 0:
        return b""

    # Vectorized: find transition points
    change_idx = np.flatnonzero(np.diff(flat))

    # Build run lengths from boundary positions
    n_runs = len(change_idx) + 1
    bounds = np.empty(n_runs + 1, dtype=np.int64)
    bounds[0] = 0
    bounds[1:n_runs] = change_idx + 1
    bounds[n_runs] = n
    run_lengths = np.diff(bounds)

    # Ensure starts with 0-run (background first)
    if flat[0] != 0:
        run_lengths = np.concatenate(([0], run_lengths))

    # For typical masks (640×480 = 307200), max run fits in uint16 only if < 65536
    # Most runs are small, so check once and fast-path
    if np.all(run_lengths <= 65535):
        arr = run_lengths.astype(np.uint16)
        return arr.tobytes()  # already little-endian on x86

    # Slow path: split runs > 65535
    counts = []
    for c in run_lengths:
        while c > 65535:
            counts.append(65535)
            counts.append(0)
            c -= 65535
        counts.append(int(c))
    return np.array(counts, dtype=np.uint16).tobytes()


def decode_rle(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode RLE bytes back to a binary mask (vectorized).

    Args:
        data: RLE-encoded bytes from encode_rle().
        width: Mask width.
        height: Mask height.

    Returns:
        2D numpy uint8 array (0 or 255).
    """
    if not data:
        return np.zeros((height, width), dtype=np.uint8)

    total_pixels = width * height

    # Parse run lengths from uint16 little-endian
    counts = np.frombuffer(data, dtype=np.uint16)

    # Values alternate: 0 (background), 255 (foreground), 0, 255, ...
    values = np.zeros(len(counts), dtype=np.uint8)
    values[1::2] = 255  # odd indices are foreground

    # Vectorized expansion
    flat = np.repeat(values, counts)

    # Handle size mismatch (truncate or pad)
    if len(flat) > total_pixels:
        flat = flat[:total_pixels]
    elif len(flat) < total_pixels:
        flat = np.pad(flat, (0, total_pixels - len(flat)))

    return flat.reshape(height, width)


def mask_to_bbox(mask: np.ndarray) -> tuple:
    """Extract bounding box from a binary mask.

    Args:
        mask: 2D array where nonzero = foreground.

    Returns:
        (x_min, y_min, x_max, y_max) in pixel coordinates,
        or None if mask is empty.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (int(x_min), int(y_min), int(x_max), int(y_max))


def mask_centroid(mask: np.ndarray) -> tuple:
    """Compute centroid of a binary mask.

    Args:
        mask: 2D array where nonzero = foreground.

    Returns:
        (cx, cy) in pixel coordinates, or None if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)))
