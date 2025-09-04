import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
from tifffile import TiffFile  # type: ignore
from skimage.exposure import rescale_intensity  # type: ignore
from skimage.feature import peak_local_max, blob_log  # type: ignore
from skimage.color import gray2rgb  # type: ignore
from skimage.draw import circle_perimeter  # type: ignore
from skimage.util import img_as_float  # type: ignore
from skimage.filters import threshold_otsu, gaussian, threshold_sauvola  # type: ignore
from skimage.morphology import (  # type: ignore
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)
from skimage.measure import regionprops  # type: ignore
from skimage.segmentation import watershed  # type: ignore
from skimage.segmentation import find_boundaries  # type: ignore
import scipy.ndimage as ndi  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import imageio.v3 as iio  # type: ignore


def _select_series_and_level(
    series_list: Sequence,
    preferred_series_index: int,
    preferred_level_index: Optional[int],
    max_dim: int = 2048,
):
    """
    Choose a series and pyramid level that will fit comfortably in memory.

    - If preferred options are provided and valid, use them.
    - Otherwise choose the first series with a level whose max(Y,X) <= max_dim,
      falling back to the coarsest available level.
    """

    if not series_list:
        raise ValueError("No series found in the TIFF file.")

    # Try to honor the user's preferred series and level first
    if 0 <= preferred_series_index < len(series_list):
        series = series_list[preferred_series_index]
        levels = getattr(series, "levels", None) or [series]
        if preferred_level_index is not None:
            if 0 <= preferred_level_index < len(levels):
                return preferred_series_index, preferred_level_index
            else:
                raise ValueError(
                    f"Requested level {preferred_level_index} is out of range for series {preferred_series_index}"
                )
        # Auto-pick a level for this series
        best_level = _choose_level_index(levels, max_dim=max_dim)
        return preferred_series_index, best_level

    # Otherwise, search across series to find a good level
    for s_idx, s in enumerate(series_list):
        levels = getattr(s, "levels", None) or [s]
        level_idx = _choose_level_index(levels, max_dim=max_dim)
        if level_idx is not None:
            return s_idx, level_idx

    # Fallback: use the first series, coarsest level
    levels = getattr(series_list[0], "levels", None) or [series_list[0]]
    return 0, len(levels) - 1


def _choose_level_index(levels: Sequence, max_dim: int = 2048) -> Optional[int]:
    """Pick the smallest level whose largest spatial dimension <= max_dim."""
    chosen = None
    for idx, level in enumerate(levels):
        shape = level.shape
        axes = getattr(level, "axes", None) or ""
        # Determine Y, X dims
        y_idx = axes.find("Y") if "Y" in axes else None
        x_idx = axes.find("X") if "X" in axes else None
        if y_idx is None or x_idx is None:
            continue
        y, x = shape[y_idx], shape[x_idx]
        if max(y, x) <= max_dim:
            chosen = idx
            break
    return chosen if chosen is not None else (len(levels) - 1 if levels else None)


def _axis_index(axes: str, axis_label: str) -> Optional[int]:
    return axes.find(axis_label) if axis_label in axes else None


def _ensure_channel_first_2d(
    data: np.ndarray,
    axes: str,
    keep_time_index: int = 0,
    projection_mode: str = "max",
) -> Tuple[np.ndarray, List[str]]:
    """
    Return data shaped as (C, Y, X) for preview generation.
    - Select a single timepoint (if T present)
    - Project Z using max or take middle slice
    """
    arr = data
    axes_str = axes

    # Handle time
    t_idx = _axis_index(axes_str, "T")
    if t_idx is not None and arr.shape[t_idx] > 1:
        indexer = [slice(None)] * arr.ndim
        indexer[t_idx] = min(keep_time_index, arr.shape[t_idx] - 1)
        arr = arr[tuple(indexer)]
        axes_str = axes_str.replace("T", "")

    # Handle Z projection or middle slice
    z_idx = _axis_index(axes_str, "Z")
    if z_idx is not None and arr.shape[z_idx] > 1:
        if projection_mode == "max":
            arr = arr.max(axis=z_idx)
        else:
            mid = arr.shape[z_idx] // 2
            arr = np.take(arr, indices=mid, axis=z_idx)
        axes_str = axes_str.replace("Z", "")

    # Ensure axes has Y and X
    if "Y" not in axes_str or "X" not in axes_str:
        raise ValueError(f"Cannot identify spatial axes in order: {axes_str}")

    # Move channel axis to front if present; otherwise create a singleton channel
    c_idx = _axis_index(axes_str, "C")
    if c_idx is None:
        # Insert a channel dimension at front
        # Current order likely YX or others; move Y,X to last two positions
        y_idx = _axis_index(axes_str, "Y")
        x_idx = _axis_index(axes_str, "X")
        perm = [i for i in range(arr.ndim) if i not in (y_idx, x_idx)] + [y_idx, x_idx]
        arr = np.transpose(arr, perm)
        r = arr[np.newaxis, ...]  # (1, Y, X)
        channel_names = ["channel0"]
        return r, channel_names

    # Reorder to C, Y, X
    # Determine positions of C,Y,X in current array
    current_axes = list(axes_str)
    order = [c_idx, current_axes.index("Y"), current_axes.index("X")]
    arr = np.transpose(arr, order)

    # Try to name channels 0..C-1; OME metadata parsing could improve this later
    num_c = arr.shape[0]
    channel_names = [f"channel{idx}" for idx in range(num_c)]
    return arr, channel_names


def _contrast_stretch(
    img: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.9,
) -> np.ndarray:
    """Apply percentile-based contrast stretching per-channel to uint8 range."""
    if img.ndim == 2:
        lo, hi = np.percentile(img, [low_percentile, high_percentile])
        if hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)
        return rescale_intensity(img, in_range=(lo, hi), out_range=(0, 255)).astype(
            np.uint8
        )

    if img.ndim == 3:
        # Assume (C, Y, X)
        out = np.empty((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
        for c in range(img.shape[0]):
            lo, hi = np.percentile(img[c], [low_percentile, high_percentile])
            if hi <= lo:
                out[c] = 0
            else:
                out[c] = rescale_intensity(
                    img[c], in_range=(lo, hi), out_range=(0, 255)
                ).astype(np.uint8)
        return out

    raise ValueError("Expected 2D or 3D array for contrast stretching")


def _save_previews(
    arr_cyx: np.ndarray,
    channel_names: List[str],
    output_dir: str,
    base_name: str,
) -> List[str]:
    """Save one PNG per channel and an RGB composite if possible. Returns file paths."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    # Save per-channel grayscale previews
    for c_idx, ch_name in enumerate(channel_names):
        img8 = _contrast_stretch(arr_cyx[c_idx])
        # Save without original image name prefix
        out_path = os.path.join(output_dir, f"{ch_name}.png")
        iio.imwrite(out_path, img8)
        saved_paths.append(out_path)

    # If at least 3 channels, make an RGB composite using first three channels
    if arr_cyx.shape[0] >= 3:
        r = _contrast_stretch(arr_cyx[0])
        g = _contrast_stretch(arr_cyx[1])
        b = _contrast_stretch(arr_cyx[2])
        rgb = np.stack([r, g, b], axis=-1)  # (Y, X, 3)
        out_path = os.path.join(output_dir, "composite_RGB.png")
        iio.imwrite(out_path, rgb)
        saved_paths.append(out_path)

    return saved_paths


def inspect_and_preview(
    filepath: str,
    series_index: int = 0,
    level_index: Optional[int] = None,
    keep_time_index: int = 0,
    projection_mode: str = "max",
    preview_max_dim: int = 2048,
    output_dir: Optional[str] = None,
) -> List[str]:
    """
    Inspect a TIFF/OME-TIFF and save quicklook previews.
    Returns list of saved image paths.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with TiffFile(filepath) as tf:
        print(f"Path: {filepath}")
        print(f"Is OME-TIFF: {getattr(tf, 'is_ome', False)}")
        print(f"Pages: {len(tf.pages)}  Series: {len(tf.series)}")
        for idx, s in enumerate(tf.series):
            axes = getattr(s, "axes", "")
            shape = getattr(s, "shape", None)
            levels = getattr(s, "levels", None)
            lvl_str = f" levels={len(levels)}" if levels else ""
            print(f"  Series {idx}: shape={shape} axes='{axes}'{lvl_str}")

        # Choose series and level
        s_idx, l_idx = _select_series_and_level(
            tf.series, series_index, level_index, max_dim=preview_max_dim
        )
        series = tf.series[s_idx]
        levels = getattr(series, "levels", None) or [series]
        level = levels[l_idx]
        print(
            f"Using series {s_idx}, level {l_idx}: shape={level.shape} axes='{level.axes}'"
        )

        # Read the selected level into memory
        arr = level.asarray()
        print(f"Loaded array dtype={arr.dtype} shape={arr.shape}")

        # Reorder and project to (C, Y, X)
        arr_cyx, channel_names = _ensure_channel_first_2d(
            arr,
            level.axes,
            keep_time_index=keep_time_index,
            projection_mode=projection_mode,
        )
        print(f"Preview array shape (C,Y,X) = {arr_cyx.shape}")

        # Define output directory
        if output_dir is None:
            parent = os.path.dirname(filepath)
            stem = os.path.splitext(os.path.basename(filepath))[0]
            output_dir = os.path.join(parent, f"{stem}__previews")

        base_name = os.path.splitext(os.path.basename(filepath))[0]
        saved = _save_previews(arr_cyx, channel_names, output_dir, base_name)
        print("Saved previews:")
        for p in saved:
            print(f"  {p}")
        return saved


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect TIFF/OME-TIFF and export quicklook previews, or count dots on a preview image."
    )
    # Inspection args
    parser.add_argument("--input", required=False, help="Path to .tif/.tiff file")
    parser.add_argument(
        "--series", type=int, default=0, help="Series index (default 0)"
    )
    parser.add_argument(
        "--level", type=int, default=None, help="Pyramid level index; default auto"
    )
    parser.add_argument(
        "--time", type=int, default=0, help="Time index to preview if T present"
    )
    parser.add_argument(
        "--zproject",
        choices=["max", "mid"],
        default="max",
        help="Z handling: maximum projection or middle slice",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2048,
        help="Target max spatial dimension for preview level selection",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for previews"
    )

    # Dot counting on a PNG preview
    parser.add_argument(
        "--count-image",
        type=str,
        default=None,
        help="Path to a grayscale preview PNG to count dots on",
    )
    parser.add_argument(
        "--min-sigma",
        type=float,
        default=1.5,
        help="Minimum sigma for LoG blob detection",
    )
    parser.add_argument(
        "--max-sigma",
        type=float,
        default=6.0,
        help="Maximum sigma for LoG blob detection",
    )
    parser.add_argument(
        "--num-sigma",
        type=int,
        default=10,
        help="Number of sigma levels between min and max",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="Absolute threshold for LoG detection (0-1 after normalization)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Blob overlap merging parameter (0-1)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Integer downsample factor before detection (speedup)",
    )
    # Thresholding controls
    parser.add_argument(
        "--threshold-mode",
        choices=["otsu", "percentile", "sauvola"],
        default="otsu",
        help="How to compute the foreground threshold",
    )
    parser.add_argument(
        "--thresh-percent",
        type=float,
        default=85.0,
        help="If percentile mode, use this intensity percentile (0-100)",
    )
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=1.0,
        help="Scale the computed threshold (e.g., 0.9 to include dimmer objects)",
    )
    parser.add_argument(
        "--ws-footprint",
        type=int,
        default=5,
        help="Footprint (square side) for peak-local-max in watershed splitting",
    )
    parser.add_argument(
        "--closing-radius",
        type=int,
        default=0,
        help="Radius for morphological closing (0 disables)",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["distance", "log", "both"],
        default="both",
        help="How to generate watershed seeds: distance map peaks, LoG blobs, or both",
    )
    parser.add_argument(
        "--min-sep-px",
        type=int,
        default=3,
        help="Minimum separation (in detection pixels) between seeds",
    )
    parser.add_argument(
        "--log-threshold",
        type=float,
        default=0.02,
        help="LoG detection threshold (relative to image scale)",
    )
    parser.add_argument(
        "--circularity-min",
        type=float,
        default=0.25,
        help="Minimum circularity (4*pi*area/perimeter^2) to accept a region",
    )
    parser.add_argument(
        "--max-diam-um",
        type=float,
        default=None,
        help="Maximum acceptable circle diameter in microns (optional)",
    )
    parser.add_argument(
        "--min-contrast",
        type=float,
        default=0.0,
        help="Minimum center-minus-ring contrast (0-1 normalized) to keep a detection",
    )
    parser.add_argument(
        "--hmax",
        type=float,
        default=0.0,
        help="h value for h-maxima on distance map to generate more watershed markers (0 disables)",
    )
    parser.add_argument(
        "--min-area-px",
        type=int,
        default=9,
        help="Minimum region area in pixels (detection scale) before measurements",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate images (mask, distance) to the output folder",
    )
    # Physical units
    parser.add_argument(
        "--width-um",
        type=float,
        default=None,
        help="Image width in microns (for physical-size filtering)",
    )
    parser.add_argument(
        "--height-um",
        type=float,
        default=None,
        help="Image height in microns (for physical-size filtering)",
    )
    parser.add_argument(
        "--min-diam-um",
        type=float,
        default=None,
        help="Minimum acceptable circle diameter in microns",
    )
    return parser.parse_args(argv)


def _count_dots_on_preview(
    preview_png_path: str,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
    overlap: float,
    downsample: int,
    width_um: Optional[float] = None,
    height_um: Optional[float] = None,
    min_diam_um: Optional[float] = None,
    threshold_mode: str = "otsu",
    thresh_percent: float = 85.0,
    threshold_scale: float = 1.0,
    ws_footprint: int = 5,
    circularity_min: float = 0.25,
    min_area_px: int = 9,
    max_diam_um: Optional[float] = None,
    debug: bool = False,
    closing_radius: int = 0,
    min_contrast: float = 0.0,
    hmax: float = 0.0,
    seed_mode: str = "both",
    min_sep_px: int = 3,
    log_threshold: float = 0.02,
    save_csv: bool = True,
) -> Tuple[int, str]:
    if not os.path.exists(preview_png_path):
        raise FileNotFoundError(f"Preview image not found: {preview_png_path}")

    img_uint8 = iio.imread(preview_png_path)
    if img_uint8.ndim == 3:
        # if RGB, convert to grayscale by taking luminance-like mean
        img_uint8 = img_uint8.mean(axis=2).astype(np.uint8)

    # Keep full-resolution image for overlay drawing
    img_full = img_as_float(img_uint8)

    # Build detection image (optionally downsampled for speed)
    if downsample > 1:
        det_img = img_full[::downsample, ::downsample]
        scale_factor = float(downsample)
    else:
        det_img = img_full
        scale_factor = 1.0
    # 1) Smooth and threshold to remove dark background
    sm = gaussian(det_img, sigma=1.0, truncate=2.0)
    # Compute threshold
    out_dir_dbg = os.path.dirname(preview_png_path)
    if debug:
        iio.imwrite(
            os.path.join(out_dir_dbg, "smooth_debug.png"),
            (np.clip(sm, 0, 1) * 255).astype(np.uint8),
        )
    if threshold_mode == "percentile":
        t = np.percentile(sm, np.clip(thresh_percent, 0.0, 100.0))
        t = t * float(threshold_scale)
        mask = sm > max(t, 0.0)
    elif threshold_mode == "sauvola":
        # Adaptive local threshold; large window to capture soft edges
        window_size = max(15, int(min(sm.shape) * 0.03))
        if window_size % 2 == 0:
            window_size += 1
        sau_t = threshold_sauvola(sm, window_size=window_size)
        mask = sm > sau_t
    else:
        try:
            t = threshold_otsu(sm)
        except Exception:
            t = np.percentile(sm, 90)
        t = t * float(threshold_scale)
        mask = sm > max(t, 0.0)
    if debug:
        # Save thresholded map and mask
        if threshold_mode != "sauvola":
            thr_img = (sm > max(t, 0.0)).astype(np.uint8) * 255
            iio.imwrite(os.path.join(out_dir_dbg, "threshold_map_debug.png"), thr_img)
        iio.imwrite(
            os.path.join(out_dir_dbg, "mask_debug.png"), (mask.astype(np.uint8) * 255)
        )
    mask = remove_small_objects(mask, min_size=max(1, int(min_area_px)))
    mask = remove_small_holes(mask, area_threshold=16)
    if closing_radius and closing_radius > 0:
        mask = binary_closing(mask, footprint=disk(int(closing_radius)))

    # 2) Distance transform and watershed to split touching objects
    distance = ndi.distance_transform_edt(mask)
    if debug:
        dm_vis = (255 * (distance / (distance.max() + 1e-6))).astype(np.uint8)
        iio.imwrite(os.path.join(out_dir_dbg, "distance_debug.png"), dm_vis)

    # Build seeds per seed_mode
    seeds_mask = np.zeros_like(mask, dtype=bool)
    if seed_mode in ("distance", "both"):
        coords = peak_local_max(
            distance,
            footprint=np.ones((max(1, int(ws_footprint)), max(1, int(ws_footprint)))),
            labels=mask,
        )
        if coords.size > 0:
            seeds_mask[tuple(coords.T)] = True

    if seed_mode in ("log", "both"):
        # Estimate sigma range from physical diameter if available; otherwise fallback to generic
        sigma_min = 1.5
        sigma_max = 6.0
        if min_diam_um is not None and width_um is not None and height_um is not None:
            H_full, W_full = img_full.shape
            px_x = width_um / float(W_full)
            px_y = height_um / float(H_full)
            px_um = np.sqrt(px_x * px_y)
            min_rad_px_full = (min_diam_um / px_um) / 2.0
            max_rad_px_full = min_rad_px_full * 2.5
            # account for downsample
            min_rad_px = min_rad_px_full / scale_factor
            max_rad_px = max_rad_px_full / scale_factor
            sigma_min = float(max(1.0, float(min_rad_px) / np.sqrt(2.0)))
            sigma_max = float(max(sigma_min + 0.5, float(max_rad_px) / np.sqrt(2.0)))
        blobs = blob_log(
            sm,
            min_sigma=sigma_min,
            max_sigma=sigma_max,
            num_sigma=10,
            threshold=log_threshold,
        )
        # Enforce min separation by writing to seeds_mask with strides around each seed
        for yx in blobs[:, :2]:
            y, x = int(yx[0]), int(yx[1])
            y0 = max(0, y - min_sep_px)
            y1 = min(seeds_mask.shape[0], y + min_sep_px + 1)
            x0 = max(0, x - min_sep_px)
            x1 = min(seeds_mask.shape[1], x + min_sep_px + 1)
            seeds_mask[y0:y1, x0:x1] = False
            if mask[y, x]:
                seeds_mask[y, x] = True

    markers = ndi.label(seeds_mask & mask)[0]
    if debug:
        iio.imwrite(
            os.path.join(out_dir_dbg, "markers_debug.png"),
            (seeds_mask.astype(np.uint8) * 255),
        )
    # Watershed on negative smoothed intensity to better split touching bright blobs
    labels_ws = watershed(-sm, markers, mask=mask)
    if debug:
        mark_vis = (markers > 0).astype(np.uint8) * 255
        iio.imwrite(os.path.join(out_dir_dbg, "markers_debug.png"), mark_vis)
        bounds = find_boundaries(labels_ws, mode="outer")
        bvis = bounds.astype(np.uint8) * 255
        iio.imwrite(os.path.join(out_dir_dbg, "boundaries_debug.png"), bvis)

    # 3) Measure regions and filter by circularity and size
    detections = []
    regions = regionprops(labels_ws)
    # Compute pixel size if physical dimensions provided
    px_size_y_um = None
    px_size_x_um = None
    if width_um is not None and height_um is not None:
        H_full, W_full = img_full.shape
        px_size_x_um = width_um / float(W_full)
        px_size_y_um = height_um / float(H_full)
    min_radius_px = None
    if (
        min_diam_um is not None
        and px_size_x_um is not None
        and px_size_y_um is not None
    ):
        # Use geometric mean pixel size to convert diameter to pixels (full-res)
        px_size_um = np.sqrt(px_size_x_um * px_size_y_um)
        min_radius_px = (min_diam_um / px_size_um) / 2.0
        # Convert threshold into detection-scale pixels if we downsampled
        if downsample > 1:
            min_radius_px = min_radius_px / float(downsample)
    for r in regions:
        if r.area < max(1, int(min_area_px)):
            continue
        perim = r.perimeter if r.perimeter > 0 else 1.0
        circ = 4.0 * np.pi * (r.area / (perim * perim))
        if circ < circularity_min:
            continue
        cy, cx = r.centroid
        rad = np.sqrt(r.area / np.pi)
        # Physical min size filter
        if min_radius_px is not None and rad < min_radius_px:
            continue
        # Physical max size filter (optional)
        if (
            max_diam_um is not None
            and px_size_x_um is not None
            and px_size_y_um is not None
        ):
            px_size_um = np.sqrt(px_size_x_um * px_size_y_um)
            max_radius_px = (max_diam_um / px_size_um) / 2.0
            if downsample > 1:
                max_radius_px = max_radius_px / float(downsample)
            if rad > max_radius_px:
                continue
        # Intensity contrast test: mean(center) - mean(ring)
        if min_contrast and min_contrast > 0:
            r_in = int(max(1, rad * 0.8))
            r_out = int(max(r_in + 1, rad * 1.3))
            cyi, cxi = int(cy), int(cx)
            # Extract a local patch to avoid scanning the full image
            pad = int(max(r_out + 1, 8))
            y0 = max(0, cyi - pad)
            y1 = min(det_img.shape[0], cyi + pad + 1)
            x0 = max(0, cxi - pad)
            x1 = min(det_img.shape[1], cxi + pad + 1)
            patch = det_img[y0:y1, x0:x1]
            py, px = np.ogrid[y0:y1, x0:x1]
            dist = np.sqrt((py - cyi) ** 2 + (px - cxi) ** 2)
            center_mask = dist <= r_in
            ring_mask = (dist > r_in) & (dist <= r_out)
            if center_mask.any() and ring_mask.any():
                contrast = float(patch[center_mask].mean() - patch[ring_mask].mean())
                gmin, gmax = float(det_img.min()), float(det_img.max())
                denom = max(1e-6, gmax - gmin)
                contrast /= denom
                if contrast < min_contrast:
                    continue
        detections.append((cy, cx, rad))

    count = len(detections)

    # 4) Create overlay visualization and draw green circle borders
    base = gray2rgb((img_full * 255).astype(np.uint8))
    overlay = base.copy()
    dets_full_res = []
    for y, x, r in detections:
        yf, xf, rf = float(y), float(x), float(r)
        if downsample > 1:
            yf = yf * float(scale_factor)
            xf = xf * float(scale_factor)
            rf = rf * float(scale_factor)
        rr, cc = circle_perimeter(
            int(yf), int(xf), max(int(rf), 1), shape=overlay.shape[:2]
        )
        overlay[rr, cc] = [0, 255, 0]
        dets_full_res.append((yf, xf, rf))

    # 5) Draw total count at top-right
    pil_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(pil_img)
    text = str(count)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        # Fallback dimensions
        text_w, text_h = (len(text) * 8, 12)
    pad = 10
    _, W = overlay.shape[0], overlay.shape[1]
    x0 = W - text_w - pad
    y0 = pad
    draw.rectangle([x0 - 4, y0 - 2, x0 + text_w + 4, y0 + text_h + 2], fill=(0, 0, 0))
    draw.text((x0, y0), text, fill=(0, 255, 0), font=font)
    overlay = np.array(pil_img)

    out_dir = os.path.dirname(preview_png_path)
    # Write CSV of detections (full-res coordinates) if requested
    if save_csv:
        try:
            csv_path = os.path.join(out_dir, "detections.csv")
            with open(csv_path, "w") as f:
                f.write("y,x,r\n")
                for yf, xf, rf in dets_full_res:
                    f.write(f"{yf:.3f},{xf:.3f},{rf:.3f}\n")
        except Exception:
            pass
    out_path = os.path.join(out_dir, "circles_overlay.png")
    iio.imwrite(out_path, overlay)
    print(f"Circle count: {count}")
    print(f"Overlay saved: {out_path}")
    return count, out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        # Dot counting mode if --count-image is provided
        if args.count_image:
            _count_dots_on_preview(
                preview_png_path=args.count_image,
                min_sigma=args.min_sigma,
                max_sigma=args.max_sigma,
                num_sigma=args.num_sigma,
                threshold=args.threshold,
                overlap=args.overlap,
                downsample=args.downsample,
                width_um=args.width_um,
                height_um=args.height_um,
                min_diam_um=args.min_diam_um,
                threshold_mode=args.threshold_mode,
                thresh_percent=args.thresh_percent,
                threshold_scale=args.threshold_scale,
                ws_footprint=args.ws_footprint,
                circularity_min=args.circularity_min,
                min_area_px=args.min_area_px,
                debug=args.debug,
                closing_radius=args.closing_radius,
                min_contrast=args.min_contrast,
                hmax=args.hmax,
                max_diam_um=args.max_diam_um,
            )
            return 0

        # Otherwise, require --input for inspection
        if not args.input:
            raise ValueError(
                "Either --input (TIFF) or --count-image (PNG) must be provided."
            )

        inspect_and_preview(
            filepath=args.input,
            series_index=args.series,
            level_index=args.level,
            keep_time_index=args.time,
            projection_mode=args.zproject,
            preview_max_dim=args.max_dim,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
