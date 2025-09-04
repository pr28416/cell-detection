import os
import time
import tempfile

# Set environment variables for large uploads before importing streamlit
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1024"
os.environ["STREAMLIT_SERVER_MAX_MESSAGE_SIZE"] = "1024"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["maxUploadSize"] = "1024"

# typing imports removed

import numpy as np  # type: ignore  # noqa: F401
import streamlit as st  # type: ignore
import imageio.v3 as iio  # type: ignore
import plotly.express as px  # type: ignore
from skimage.transform import resize  # type: ignore
from streamlit_cropper import st_cropper  # type: ignore
from PIL import Image  # type: ignore

from main import inspect_and_preview, _count_dots_on_preview


# Helper functions for robust file upload


# Force configure upload limits using Streamlit's internal config
try:
    from streamlit import config  # type: ignore

    config._set_option("server.maxUploadSize", 1024, "command_line")
    config._set_option("server.maxMessageSize", 1024, "command_line")
except:
    pass

# Also try setting via runtime config
try:
    import streamlit.runtime.config as runtime_config  # type: ignore

    runtime_config.get_config_options()["server.maxUploadSize"] = 1024
    runtime_config.get_config_options()["server.maxMessageSize"] = 1024
except:
    pass

# Configure Streamlit for large file uploads
st.set_page_config(page_title="Cell Detector", layout="wide")
st.title("Cell Detection")

# Display current upload limit for debugging
try:
    from streamlit import config  # type: ignore

    current_limit = config.get_option("server.maxUploadSize")
    st.caption(f"🔧 Current upload limit: {current_limit}MB")
except:
    st.caption("🔧 Upload limit: Using default configuration")

# Upload first with better error handling
st.markdown("### 📁 File Upload")
uploaded = st.file_uploader(
    "Upload .tif/.tiff image",
    type=["tif", "tiff"],
    help="🔬 Upload your microscopy TIFF file. Large files (>500MB) may take several minutes to upload.",
)


# Show upload progress
if uploaded is not None:
    try:
        file_size_mb = len(uploaded.getvalue()) / (1024 * 1024)
        st.success(
            f"✅ Upload successful! File: {uploaded.name} ({file_size_mb:.1f}MB)"
        )

    except Exception as e:
        st.error(f"❌ Upload error: {str(e)}")
        st.error("Please try refreshing the page and uploading again.")
        uploaded = None


# Helper to render settings panel next to slice preview
def render_settings_panel():
    st.subheader("Settings")

    # Basic image parameters
    with st.expander("📏 Image dimensions", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            width_um = st.number_input(
                "Width (µm)", value=1705.6, help="Physical width of the scan."
            )
        with col2:
            height_um = st.number_input(
                "Height (µm)", value=1706.81, help="Physical height of the scan."
            )

        col3, col4 = st.columns(2)
        with col3:
            min_diam_um = st.number_input(
                "Min diameter (µm)",
                value=10.0,
                help="Ignore circles smaller than this size.",
            )
        with col4:
            downsample = st.slider(
                "Speed",
                1,
                4,
                2,
                help="Higher = faster preview, slightly less detail.",
            )

    # Detection parameters
    with st.expander("🎯 Detection", expanded=True):
        threshold_mode = st.selectbox(
            "Threshold method",
            ["percentile", "otsu", "sauvola"],
            help="How we separate bright cells from background.",
        )

        col1, col2 = st.columns(2)
        with col1:
            thresh_percent = st.slider(
                "Percentile (%)",
                50,
                99,
                72,
                help="Lower to include dimmer cells (percentile mode).",
            )
        with col2:
            threshold_scale = st.slider(
                "Threshold scale",
                0.5,
                1.5,
                0.8,
                help="Fine‑tune sensitivity around the threshold.",
            )

    # Cell separation parameters
    with st.expander("✂️ Cell separation", expanded=False):
        seed_mode = st.selectbox(
            "Split method",
            ["both", "distance", "log"],
            help="How centers are found to split touching cells.",
        )

        col1, col2 = st.columns(2)
        with col1:
            ws_footprint = st.slider(
                "Split tightness",
                1,
                9,
                4,
                help="Larger splits clustered cells more aggressively.",
            )
            min_sep_px = st.slider(
                "Seed spacing", 0, 6, 2, help="Minimum spacing between seeds."
            )
        with col2:
            log_threshold = st.slider(
                "Seed strength", 0.0, 0.1, 0.02, help="Raise to reduce spurious seeds."
            )
            closing_radius = st.slider(
                "Fill gaps", 0, 5, 2, help="Fills tiny holes along cell edges."
            )

    # Filtering parameters
    with st.expander("🔍 Filtering", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            circularity_min = st.slider(
                "Roundness filter",
                0.0,
                1.0,
                0.18,
                help="Lower accepts more irregular shapes.",
            )
        with col2:
            min_contrast = st.slider(
                "Contrast filter",
                0.0,
                0.2,
                0.03,
                help="Raise to keep only high‑contrast cells.",
            )

    debug = st.checkbox(
        "💾 Save debug images", value=True, help="Save step-by-step processing images"
    )

    # Reset button
    st.divider()
    if st.button(
        "🔄 Reset to recommended settings",
        help="Restore all parameters to recommended defaults",
    ):
        # Clear session state to trigger reset on next render
        if "_reset_settings" in st.session_state:
            del st.session_state["_reset_settings"]
        st.session_state["_reset_settings"] = True
        st.rerun()

    # Apply reset if requested
    if st.session_state.get("_reset_settings", False):
        st.session_state["_reset_settings"] = False
        # Return default values
        return (
            1705.6,  # width_um
            1706.81,  # height_um
            10.0,  # min_diam_um
            2,  # downsample
            "percentile",  # threshold_mode
            72,  # thresh_percent
            0.8,  # threshold_scale
            2,  # closing_radius
            "both",  # seed_mode
            4,  # ws_footprint
            2,  # min_sep_px
            0.02,  # log_threshold
            0.18,  # circularity_min
            0.03,  # min_contrast
            True,  # debug
        )

    return (
        width_um,
        height_um,
        min_diam_um,
        downsample,
        threshold_mode,
        thresh_percent,
        threshold_scale,
        closing_radius,
        seed_mode,
        ws_footprint,
        min_sep_px,
        log_threshold,
        circularity_min,
        min_contrast,
        debug,
    )


if uploaded is not None:
    # Persist upload to a stable session temp folder to avoid regenerating on each rerun
    if "_work_dir" not in st.session_state:
        st.session_state["_work_dir"] = tempfile.mkdtemp()
    upload_sig = (uploaded.name, getattr(uploaded, "size", None))
    if st.session_state.get("_upload_sig") != upload_sig:
        st.session_state["_upload_sig"] = upload_sig
        in_path = os.path.join(st.session_state["_work_dir"], uploaded.name)
        with open(in_path, "wb") as f:
            f.write(uploaded.read())
        st.session_state["_input_path"] = in_path
        # Reset previews ready flag
        st.session_state["_previews_ready"] = False
    in_path = st.session_state.get("_input_path")

    # Preview generation
    st.subheader("Channel previews")

    @st.cache_data(show_spinner=False)
    def generate_previews(input_path: str):
        return inspect_and_preview(input_path)

    if not st.session_state.get("_previews_ready"):
        with st.status("Generating channel previews...", expanded=True) as status:
            t0 = time.time()
            saved = generate_previews(in_path)
            t1 = time.time()
            st.session_state["_previews_ready"] = True
            status.update(
                label=f"Generated {len(saved)} preview images in {t1 - t0:.2f}s",
                state="complete",
                expanded=False,
            )
    else:
        # Ensure previews exist without recomputation (cache hit)
        _ = generate_previews(in_path)

    # Find previews and show a single zoomable viewer with channel selector
    prev_dir = os.path.splitext(in_path)[0] + "__previews"
    options = []
    paths = {}
    for i in range(4):
        p = os.path.join(prev_dir, f"channel{i}.png")
        if os.path.exists(p):
            key = f"channel{i}"
            options.append(key)
            paths[key] = p
    comp = os.path.join(prev_dir, "composite_RGB.png")
    if os.path.exists(comp):
        options.append("composite_RGB")
        paths["composite_RGB"] = comp

    @st.cache_data(show_spinner=False)
    def load_preview(path: str, max_dim: int = 2048):
        img = iio.imread(path)
        h, w = img.shape[:2]
        scale = max(h, w) / max_dim if max(h, w) > max_dim else 1.0
        if scale > 1.0:
            nh, nw = int(h / scale), int(w / scale)
            img = resize(img, (nh, nw), preserve_range=True, anti_aliasing=True).astype(
                img.dtype
            )
        return img

    if options:
        st.subheader("Image viewer")
        sel = st.selectbox("Channel", options, index=min(1, len(options) - 1))
        img = load_preview(paths[sel])
        fig = px.imshow(img, color_continuous_scale="gray", origin="upper")
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Slice + Settings side-by-side
        left, right = st.columns([2, 1], gap="large")

        # Slice selection (left)
        with left:
            st.subheader("Slice preview")
            st.caption(
                "Drag to select a slice (100–1024 px) of the current channel to preview with your settings."
            )
            current_path = paths.get(sel or "", None)
            if current_path:
                pil_img = Image.open(current_path).convert("L")
                # Get original image dimensions for physical scaling calculation
                orig_h, orig_w = (
                    pil_img.size[1],
                    pil_img.size[0],
                )  # PIL uses (W, H) format

                slice_img = st_cropper(pil_img, aspect_ratio=None, box_color="#00FF00")
                snp = np.array(slice_img)
                h, w = snp.shape[:2]
                if h < 100 or w < 100:
                    st.warning(
                        "Selected slice is too small. Increase selection to at least 100×100."
                    )
                else:
                    # Get original physical dimensions to maintain pixel-to-micron ratio
                    s = st.session_state.get("_settings", {})
                    orig_width_um = s.get("width_um", 1705.6)
                    orig_height_um = s.get("height_um", 1706.81)

                    # Calculate pixel size from ORIGINAL image
                    px_size_x_um = orig_width_um / orig_w
                    px_size_y_um = orig_height_um / orig_h

                    # Handle downscaling if slice is too large
                    actual_h, actual_w = h, w
                    if max(h, w) > 1024:
                        scale = max(h, w) / 1024.0
                        actual_h, actual_w = int(h / scale), int(w / scale)
                        snp = resize(
                            snp, (actual_h, actual_w), preserve_range=True
                        ).astype(np.uint8)

                    # Calculate effective physical dimensions that maintain original pixel size
                    # This tricks the function into using the correct pixel-to-micron ratio
                    slice_width_um = actual_w * px_size_x_um
                    slice_height_um = actual_h * px_size_y_um

                    roi_path = os.path.join(prev_dir, "slice.png")
                    iio.imwrite(roi_path, snp)

                    # Calculate what the minimum radius should be in pixels for debugging  
                    min_diam_um = s.get("min_diam_um", 10.0)
                    avg_px_size_um = np.sqrt(px_size_x_um * px_size_y_um)
                    expected_min_radius_px = (min_diam_um / avg_px_size_um) / 2.0
                    
                    # Show slice info
                    st.caption(
                        f"📏 Slice: {actual_w}×{actual_h} px | Pixel size: {px_size_x_um:.3f}×{px_size_y_um:.3f} µm/px"
                    )
                    st.caption(
                        f"🔍 Debug: {min_diam_um}µm min diameter → expected ~{expected_min_radius_px:.1f}px min radius"
                    )

                    if st.button("Preview on slice"):
                        # Get settings from session state if available, fallback to defaults
                        s = st.session_state.get("_settings", {})
                        width_um = s.get("width_um", 1705.6)
                        height_um = s.get("height_um", 1706.81)
                        min_diam_um = s.get("min_diam_um", 10.0)
                        downsample = s.get("downsample", 2)
                        threshold_mode = s.get("threshold_mode", "percentile")
                        thresh_percent = s.get("thresh_percent", 72.0)
                        threshold_scale = s.get("threshold_scale", 0.8)
                        closing_radius = s.get("closing_radius", 2)
                        seed_mode = s.get("seed_mode", "both")
                        ws_footprint = s.get("ws_footprint", 4)
                        min_sep_px = s.get("min_sep_px", 2)
                        log_threshold = s.get("log_threshold", 0.02)
                        circularity_min = s.get("circularity_min", 0.18)
                        min_contrast = s.get("min_contrast", 0.03)
                        debug = s.get("debug", True)

                        with st.spinner("Detecting on slice..."):
                            t0 = time.time()
                            
                            # Debug: show what we're passing to the function
                            st.write(f"🔧 Debug: Passing width_um={slice_width_um:.2f}, height_um={slice_height_um:.2f}")
                            st.write(f"🔧 Debug: Slice is {actual_w}×{actual_h} px, downsample={downsample}")
                            calc_px_size_x = slice_width_um / actual_w
                            calc_px_size_y = slice_height_um / actual_h
                            st.write(f"🔧 Debug: Function will calculate px_size: {calc_px_size_x:.4f}×{calc_px_size_y:.4f} µm/px")
                            
                            slice_count, _ = _count_dots_on_preview(
                                preview_png_path=roi_path,
                                min_sigma=1.5,
                                max_sigma=6.0,
                                num_sigma=10,
                                threshold=0.03,
                                overlap=0.5,
                                downsample=downsample,
                                width_um=slice_width_um,  # Use calculated slice dimensions
                                height_um=slice_height_um,  # Use calculated slice dimensions
                                min_diam_um=min_diam_um,
                                threshold_mode=threshold_mode,
                                thresh_percent=float(thresh_percent),
                                threshold_scale=float(threshold_scale),
                                ws_footprint=int(ws_footprint),
                                circularity_min=float(circularity_min),
                                min_area_px=9,
                                max_diam_um=None,
                                debug=debug,
                                closing_radius=int(closing_radius),
                                min_contrast=float(min_contrast),
                                hmax=0.0,
                                seed_mode=seed_mode,
                                min_sep_px=int(min_sep_px),
                                log_threshold=float(log_threshold),
                                save_csv=False,
                            )
                            t1 = time.time()
                        st.success(
                            f"🎯 Found **{slice_count} cells** in slice ({t1 - t0:.2f}s)"
                        )
                        st.image(
                            os.path.join(prev_dir, "circles_overlay.png"),
                            caption="Slice overlay",
                            width="stretch",
                        )

        # Settings panel (right)
        with right:
            (
                width_um,
                height_um,
                min_diam_um,
                downsample,
                threshold_mode,
                thresh_percent,
                threshold_scale,
                closing_radius,
                seed_mode,
                ws_footprint,
                min_sep_px,
                log_threshold,
                circularity_min,
                min_contrast,
                debug,
            ) = render_settings_panel()
            # Persist settings for later use (e.g., full run)
            st.session_state["_settings"] = dict(
                width_um=width_um,
                height_um=height_um,
                min_diam_um=min_diam_um,
                downsample=downsample,
                threshold_mode=threshold_mode,
                thresh_percent=float(thresh_percent),
                threshold_scale=float(threshold_scale),
                closing_radius=int(closing_radius),
                seed_mode=seed_mode,
                ws_footprint=int(ws_footprint),
                min_sep_px=int(min_sep_px),
                log_threshold=float(log_threshold),
                circularity_min=float(circularity_min),
                min_contrast=float(min_contrast),
                debug=bool(debug),
            )

    # Full run (only when options/settings are active)
    if options:
        st.subheader("Full run")
        if st.button("Run full detection with selected settings"):
            # Load latest settings from session (ensures variables are defined)
            s = st.session_state.get("_settings", {})
            width_um = s.get("width_um", 1705.6)
            height_um = s.get("height_um", 1706.81)
            min_diam_um = s.get("min_diam_um", 10.0)
            downsample = s.get("downsample", 2)
            threshold_mode = s.get("threshold_mode", "percentile")
            thresh_percent = s.get("thresh_percent", 72.0)
            threshold_scale = s.get("threshold_scale", 0.8)
            closing_radius = s.get("closing_radius", 2)
            seed_mode = s.get("seed_mode", "both")
            ws_footprint = s.get("ws_footprint", 4)
            min_sep_px = s.get("min_sep_px", 2)
            log_threshold = s.get("log_threshold", 0.02)
            circularity_min = s.get("circularity_min", 0.18)
            min_contrast = s.get("min_contrast", 0.03)
            debug = s.get("debug", True)
            c1_path = os.path.join(prev_dir, "channel1.png")
            if not os.path.exists(c1_path):
                st.error("channel1.png not found in previews")
            else:
                prog = st.progress(0)
                prog.progress(5)
                with st.spinner("Running detection..."):
                    t0 = time.time()
                    full_count, _ = _count_dots_on_preview(
                        preview_png_path=c1_path,
                        min_sigma=1.5,
                        max_sigma=6.0,
                        num_sigma=10,
                        threshold=0.03,
                        overlap=0.5,
                        downsample=downsample,
                        width_um=width_um,
                        height_um=height_um,
                        min_diam_um=min_diam_um,
                        threshold_mode=threshold_mode,
                        thresh_percent=float(thresh_percent),
                        threshold_scale=float(threshold_scale),
                        ws_footprint=int(ws_footprint),
                        circularity_min=float(circularity_min),
                        min_area_px=9,
                        max_diam_um=None,
                        debug=debug,
                        closing_radius=int(closing_radius),
                        min_contrast=float(min_contrast),
                        hmax=0.0,
                        seed_mode=seed_mode,
                        min_sep_px=int(min_sep_px),
                        log_threshold=float(log_threshold),
                        save_csv=True,
                    )
                    prog.progress(95)
                    t1 = time.time()
                # Mark detection as completed and store results
                overlay_path = os.path.join(prev_dir, "circles_overlay.png")
                csv_path = os.path.join(prev_dir, "detections.csv")

                # Read and store file data in session state to persist across reruns
                st.session_state["_detection_completed"] = True
                st.session_state["_detection_time"] = t1 - t0
                st.session_state["_cell_count"] = full_count
                st.session_state["_overlay_path"] = overlay_path

                if os.path.exists(overlay_path):
                    with open(overlay_path, "rb") as f:
                        st.session_state["_overlay_data"] = f.read()

                if os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.session_state["_csv_data"] = f.read()

        # Show results if detection has been completed (persistent across reruns)
        if st.session_state.get("_detection_completed", False):
            overlay_path = st.session_state.get("_overlay_path")
            csv_path = st.session_state.get("_csv_path")
            detection_time = st.session_state.get("_detection_time", 0)
            cell_count = st.session_state.get("_cell_count", 0)

            if overlay_path and os.path.exists(overlay_path):
                st.success(
                    f"✅ Detection completed: **{cell_count} cells found** ({detection_time:.2f}s)"
                )

                # Results section with better styling
                st.subheader("Results")
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.image(
                        overlay_path,
                        caption="Detection overlay - circles show detected cells",
                        width="stretch",
                    )

                with col2:
                    st.markdown("### 📥 Downloads")
                    st.markdown("Click to download your results:")

                    # Download overlay image
                    overlay_data = st.session_state.get("_overlay_data")
                    if overlay_data:
                        st.download_button(
                            "🖼️ Overlay image",
                            data=overlay_data,
                            file_name="cell_detection_overlay.png",
                            mime="image/png",
                            help="Download the annotated image with detected circles",
                        )

                    # Download CSV data
                    csv_data = st.session_state.get("_csv_data")
                    if csv_data:
                        st.download_button(
                            "📊 Detection data",
                            data=csv_data,
                            file_name="cell_detection_data.csv",
                            mime="text/csv",
                            help="Download CSV with cell coordinates and properties",
                        )

                    # Clear results button
                    st.markdown("---")
                    if st.button(
                        "🗑️ Clear results", help="Clear detection results to run again"
                    ):
                        st.session_state["_detection_completed"] = False
                        # Clear all detection-related session state
                        for key in [
                            "_overlay_path",
                            "_csv_path",
                            "_detection_time",
                            "_overlay_data",
                            "_csv_data",
                            "_cell_count",
                        ]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()

else:
    st.info("Upload a .tif to begin.")

