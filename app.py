import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import os
import base64
from gee_pipeline import fetch_regional_data
from dip_engine import prepare_tensors_for_inference
from inference_hacker import run_pipeline

# ------- NEW IMPORTS for interactive map -------
import folium
from folium.plugins import Draw

# fallback if streamlit-folium is not installed
try:
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

st.set_page_config(page_title="U-RNN Flood Sim", layout="wide")
st.title("🌊 U-RNN Urban Flood Nowcasting")


# ---------------------------------------------------------------------------
# ANIMATED GIF
# ---------------------------------------------------------------------------
def generate_smooth_flood_gif(rgb_base, predicted_depths):
    # Guard: ensure uint8 RGB
    if rgb_base.dtype != np.uint8:
        rgb_base = np.clip(rgb_base, 0, 255).astype(np.uint8)

    # Gamma-brighten city layer (lifts shadows without clipping highlights)
    gamma = 0.75
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    rgb_bright = lut[rgb_base]

    # Colormap: fully transparent below vmin
    cmap = plt.cm.Blues.copy()
    cmap.set_under(color='none')

    H_img, W_img = rgb_bright.shape[:2]
    extent = [0, W_img, H_img, 0]

    # --- Dynamic threshold ---
    nonzero = predicted_depths[predicted_depths > 0]
    if nonzero.size > 0:
        DRY_THRESHOLD = max(float(np.percentile(nonzero, 10)), 0.001)
    else:
        DRY_THRESHOLD = 0.001

    # Stable colour scale: 95th percentile across full simulation
    all_wet = predicted_depths[predicted_depths > DRY_THRESHOLD]
    if all_wet.size > 0:
        global_vmax = max(float(np.percentile(all_wet, 95)), DRY_THRESHOLD * 2)
    else:
        global_vmax = 1.0

    print(f"[gif] DRY_THRESHOLD={DRY_THRESHOLD:.4f} m  vmax={global_vmax:.4f} m")

    gif_path = "flood_animation.gif"
    frames = []

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    fig.patch.set_facecolor('black')
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    for t in range(predicted_depths.shape[0]):
        ax.clear()
        ax.set_facecolor('black')
        ax.axis('off')

        # 1. City base map
        ax.imshow(rgb_bright, extent=extent, aspect='auto', zorder=1)

        # 2. Upscale 128×128 → 512×512
        depth_128 = predicted_depths[t]
        depth_hires = cv2.resize(
            depth_128, (W_img, H_img), interpolation=cv2.INTER_CUBIC
        )
        depth_smooth = cv2.GaussianBlur(depth_hires, (9, 9), 0)

        # 3. Mask dry pixels
        masked = np.ma.masked_where(depth_smooth <= DRY_THRESHOLD, depth_smooth)

        # 4. Flood overlay
        ax.imshow(
            masked,
            cmap=cmap,
            vmin=DRY_THRESHOLD + 1e-6,
            vmax=global_vmax,
            extent=extent,
            aspect='auto',
            interpolation='bilinear',
            alpha=0.65,
            zorder=2
        )

        # 5. Timestamp
        ax.text(
            8, 24,
            f"+{t * 10} min",
            color='white', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='#00000099', edgecolor='none', pad=3),
            zorder=3
        )

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])

    plt.close(fig)
    imageio.mimsave(gif_path, frames, fps=3, loop=0)
    return gif_path


def display_animated_gif(gif_path: str, caption: str = ""):
    """Embed GIF via base64 HTML — st.image() only shows the first frame."""
    with open(gif_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    html = f"""
    <div style="text-align:center; margin-top:8px;">
        <img src="data:image/gif;base64,{data}"
             style="width:100%; max-width:700px; border-radius:8px;"
             alt="{caption}" />
        <p style="color:#aaa; font-size:13px; margin-top:6px;">{caption}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def show_diagnostic_heatmap(predicted_depths):
    """
    Show a static average-depth heatmap so we can visually confirm the model
    produced non-zero flood signal before even looking at the animation.
    """
    avg_depth = predicted_depths.mean(axis=0)   # (H, W)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(avg_depth, cmap='Blues', interpolation='nearest')
    ax.set_title("Avg flood depth (model output, metres)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.sidebar.header("Simulation Settings")

# --- Interactive Map Option (new) ---
use_map = st.sidebar.checkbox("Use Interactive Map", value=False)

if use_map and FOLIUM_AVAILABLE:
    st.sidebar.info("Draw a rectangle on the map to define your area of interest.")
    # Default centre: roughly the middle of the default bbox
    default_centre = [51.49, -0.15]
    m = folium.Map(location=default_centre, zoom_start=13)

    Draw(
        export=True,
        draw_options={
            'polyline': False,
            'polygon': True,
            'rectangle': True,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        },
        edit_options={'edit': False}
    ).add_to(m)

    # Render map and capture drawn features
    map_data = st_folium(m, width=700, height=400)

    # Extract bounding box from the last drawn rectangle/polygon
    if map_data and map_data.get("last_active_drawing"):
        geom = map_data["last_active_drawing"]["geometry"]
        coords = geom["coordinates"][0]  # outer ring of polygon/rectangle

        # coords is a list of [lng, lat] pairs
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        bbox_from_map = [min(lons), min(lats), max(lons), max(lats)]

        st.sidebar.success(f"Selected area: {bbox_from_map}")
        st.session_state["bbox"] = bbox_from_map
    else:
        st.sidebar.warning("No area drawn yet – default coordinates will be used.")
        st.session_state["bbox"] = [-0.20, 51.46, -0.10, 51.52]

else:
    if use_map and not FOLIUM_AVAILABLE:
        st.sidebar.warning(
            "Interactive map is not available because `streamlit-folium` "
            "is not installed. Falling back to manual input."
        )

    # Fallback to original manual input
    bbox_input = st.sidebar.text_input(
        "BBox (Lon Min, Lat Min, Lon Max, Lat Max)",
        "-0.20, 51.46, -0.10, 51.52"
    )
    # Parse the text and store in session state
    st.session_state["bbox"] = [float(x.strip()) for x in bbox_input.split(',')]

gee_project = st.sidebar.text_input("GEE Project ID")
intensity   = st.sidebar.slider("Rainfall (mm/hr)",   10.0, 150.0, 50.0)
time_steps  = st.sidebar.slider("Duration (steps)",   6,    24,    12)

if st.sidebar.button("Run AI Simulation"):
    bbox = st.session_state.get("bbox", [-0.20, 51.46, -0.10, 51.52])

    with st.spinner("🌍 Fetching 10 m Sentinel-2 data from GEE…"):
        dem, lc, rgb = fetch_regional_data(bbox, project=gee_project.strip())

    with st.spinner("⚙️ Aligning tensors…"):
        dem_p, lc_p, rain_p, rgb_p = prepare_tensors_for_inference(
            dem, lc, rgb, time_steps, intensity
        )

    # Show city base layer immediately
    st.subheader("City Base Layer (Sentinel-2)")
    st.image(rgb_p, caption="10 m Sentinel-2 RGB", use_container_width=True)

    with st.spinner("🧠 Running U-RNN inference…"):
        predicted_depths = run_pipeline(dem_p, lc_p, rain_p)

    # --- Diagnostics ---
    st.subheader("Model Output Diagnostics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Min depth", f"{predicted_depths.min():.4f} m")
    col2.metric("Max depth", f"{predicted_depths.max():.4f} m")
    col3.metric("Mean depth", f"{predicted_depths.mean():.4f} m")

    if predicted_depths.max() < 1e-4:
        st.error(
            "⚠️ Model output is essentially zero across all timesteps. "
            "The weights are likely not loading correctly (check the terminal "
            "for '[inference] WARNING' messages). The animation will be blank."
        )
    else:
        st.success(f"✅ Flood signal detected — max depth {predicted_depths.max():.3f} m")

    show_diagnostic_heatmap(predicted_depths)

    with st.spinner("🎨 Rendering flood animation…"):
        gif_path = generate_smooth_flood_gif(rgb_p, predicted_depths)

    display_animated_gif(
        gif_path,
        caption=f"Urban Flood Simulation ({time_steps * 10} mins)"
    )

    with open(gif_path, "rb") as f:
        gif_bytes = f.read()

    st.download_button(
        "⬇️ Download Animation",
        data=gif_bytes,
        file_name="urnn_flood_sim.gif",
        mime="image/gif"
    )