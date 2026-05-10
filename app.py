# === 🌊 U-RNN Urban Flood Nowcasting – Interactive Streamlit Dashboard ===
# This application:
#   1. Lets the user select a region via an interactive map or manual coordinates
#   2. Fetches DEM, land cover, and Sentinel-2 RGB imagery from Google Earth Engine
#   3. Preprocesses the data into tensors for the U-RNN flood model
#   4. Runs the deep‑learning model to predict flood depths over time
#   5. Renders an animated GIF of the flood progression overlaid on the satellite image

import streamlit as st                # Web app framework
import numpy as np                    # Array operations
import cv2                            # OpenCV for image resizing and smoothing
import matplotlib.pyplot as plt       # Base map rendering and colormaps
import imageio                        # GIF creation from frames
import os                             # File path operations (not used here directly but available)
import base64                         # Encode GIF for HTML embedding

# Custom project modules (must be in the same directory)
from gee_pipeline import fetch_regional_data           # Earth Engine data retrieval
from dip_engine import prepare_tensors_for_inference   # Image processing & tensor prep
from inference_hacker import run_pipeline              # Model loading & inference

# ------- Imports for interactive map -------
import folium                         # Leaflet map rendered in Streamlit
from folium.plugins import Draw       # Drawing tools (rectangle/polygon)

# Try to import streamlit-folium component; fallback gracefully if missing
try:
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# --- Page configuration (must be the first Streamlit command) ---
st.set_page_config(page_title="U-RNN Flood Sim", layout="wide")
st.title("🌊 U-RNN Urban Flood Nowcasting")


# ============================================================================
#   ANIMATED FLOOD GIF GENERATION
# ============================================================================
def generate_smooth_flood_gif(rgb_base, predicted_depths):
    """
    Creates an animated GIF of flood progression over a high‑resolution satellite image.

    Parameters:
        rgb_base:          High‑res RGB image (512x512, uint8) from Sentinel‑2
        predicted_depths:  Float32 array of shape (T, 128, 128) – flood depths in metres
                           for each time step (10‑minute intervals)

    Returns:
        Path to the saved 'flood_animation.gif'
    """
    # -------------------- Guard: ensure uint8 RGB --------------------
    # Some sources may accidentally return float; clip and convert to avoid crashes
    if rgb_base.dtype != np.uint8:
        rgb_base = np.clip(rgb_base, 0, 255).astype(np.uint8)

    # -------------------- Gamma brightening --------------------
    # Use a power‑law (gamma) transformation to lighten the dark city
    # background without blowing out bright areas. This improves visibility of
    # the overlaid flood map.
    gamma = 0.75
    # Build a lookup table (LUT) for speed
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    rgb_bright = lut[rgb_base]          # Apply gamma correction

    # -------------------- Colormap setup --------------------
    # Use matplotlib's 'Blues' colormap, but make values *below* the defined minimum
    # completely transparent (so dry land shows the satellite image).
    cmap = plt.cm.Blues.copy()
    cmap.set_under(color='none')        # Transparent for dry areas

    # Dimensions of the high‑res background
    H_img, W_img = rgb_bright.shape[:2]
    # Extent for imshow: [left, right, bottom, top]
    extent = [0, W_img, H_img, 0]

    # ------------- Dynamic dry/wet threshold -------------
    # Not all model outputs are large; we need an adaptive threshold to
    # separate "dry" from "flooded". Use 10th percentile of *non‑zero* values.
    nonzero = predicted_depths[predicted_depths > 0]
    if nonzero.size > 0:
        DRY_THRESHOLD = max(float(np.percentile(nonzero, 10)), 0.001)
    else:
        DRY_THRESHOLD = 0.001

    # ----------- Global colour scale (vmax) -----------
    # To avoid colour scale flickering between frames, compute a single vmax
    # based on the 95th percentile of all wet values across all timesteps.
    all_wet = predicted_depths[predicted_depths > DRY_THRESHOLD]
    if all_wet.size > 0:
        global_vmax = max(float(np.percentile(all_wet, 95)), DRY_THRESHOLD * 2)
    else:
        global_vmax = 1.0

    print(f"[gif] DRY_THRESHOLD={DRY_THRESHOLD:.4f} m  vmax={global_vmax:.4f} m")

    gif_path = "flood_animation.gif"
    frames = []

    # Set up a matplotlib figure with a black background and minimal padding
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    fig.patch.set_facecolor('black')
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    # Loop over each predicted time step (10 min apart)
    for t in range(predicted_depths.shape[0]):
        ax.clear()
        ax.set_facecolor('black')
        ax.axis('off')                      # Hide axes

        # 1. Render the base RGB map (zorder=1 -> bottom layer)
        ax.imshow(rgb_bright, extent=extent, aspect='auto', zorder=1)

        # 2. Upsample the 128x128 flood depth to match the background (512x512)
        depth_128 = predicted_depths[t]
        # Use bicubic interpolation for smooth upscaling
        depth_hires = cv2.resize(
            depth_128, (W_img, H_img), interpolation=cv2.INTER_CUBIC
        )
        # Apply Gaussian blur to soften blocky artefacts from upscaling
        depth_smooth = cv2.GaussianBlur(depth_hires, (9, 9), 0)

        # 3. Create a masked array: hide all pixels below the dry threshold
        masked = np.ma.masked_where(depth_smooth <= DRY_THRESHOLD, depth_smooth)

        # 4. Overlay the flood layer (zorder=2) with transparency (alpha)
        ax.imshow(
            masked,
            cmap=cmap,
            vmin=DRY_THRESHOLD + 1e-6,   # force values > threshold to be coloured
            vmax=global_vmax,
            extent=extent,
            aspect='auto',
            interpolation='bilinear',
            alpha=0.65,                  # 65% opacity so background is still visible
            zorder=2
        )

        # 5. Add a timestamp label (top‑left corner)
        ax.text(
            8, 24,
            f"+{t * 10} min",
            color='white', fontsize=11, fontweight='bold',
            bbox=dict(facecolor='#00000099', edgecolor='none', pad=3),
            zorder=3                    # Ensure text is on top
        )

        # Render the figure canvas to a NumPy array and capture only RGB channels
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])

    plt.close(fig)                       # Free memory
    # Save the list of frames as an animated GIF (3 frames per second, looping)
    imageio.mimsave(gif_path, frames, fps=3, loop=0)
    return gif_path


def display_animated_gif(gif_path: str, caption: str = ""):
    """
    Embed the GIF in the Streamlit app via a base64‑encoded HTML <img> tag.
    This is necessary because st.image() only shows the first frame of a GIF.
    """
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
    Quick sanity check: display the *average* flood depth over time.
    If the model produced only zeros, this heatmap will be blank,
    alerting the user before they waste time on an empty animation.
    """
    avg_depth = predicted_depths.mean(axis=0)   # Average across time -> (128, 128)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(avg_depth, cmap='Blues', interpolation='nearest')
    ax.set_title("Avg flood depth (model output, metres)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    st.pyplot(fig)                              # Streamlit renders the matplotlib figure
    plt.close(fig)


# ============================================================================
#   SIDEBAR – User Controls
# ============================================================================
st.sidebar.header("Simulation Settings")

# -- Interactive map toggle --
use_map = st.sidebar.checkbox("Use Interactive Map", value=False)

if use_map and FOLIUM_AVAILABLE:
    st.sidebar.info("Draw a rectangle on the map to define your area of interest.")
    # Centre the map on the approximate default area (London)
    default_centre = [51.49, -0.15]
    m = folium.Map(location=default_centre, zoom_start=13)

    # Add drawing tools (rectangle & polygon only)
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
        edit_options={'edit': False}   # prevent accidental edits after drawing
    ).add_to(m)

    # Render the interactive map and capture any drawn features
    map_data = st_folium(m, width=700, height=400)

    # Extract bounding box from the last drawn shape
    if map_data and map_data.get("last_active_drawing"):
        geom = map_data["last_active_drawing"]["geometry"]
        coords = geom["coordinates"][0]   # outer ring of polygon (always exists for rectangle too)

        # coords is a list of [lng, lat] pairs; extract min/max to form bbox
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        bbox_from_map = [min(lons), min(lats), max(lons), max(lats)]

        st.sidebar.success(f"Selected area: {bbox_from_map}")
        st.session_state["bbox"] = bbox_from_map
    else:
        st.sidebar.warning("No area drawn yet – default coordinates will be used.")
        st.session_state["bbox"] = [-0.20, 51.46, -0.10, 51.52]

else:
    # Fallback: if map not requested or library missing, show manual text input
    if use_map and not FOLIUM_AVAILABLE:
        st.sidebar.warning(
            "Interactive map is not available because `streamlit-folium` "
            "is not installed. Falling back to manual input."
        )

    bbox_input = st.sidebar.text_input(
        "BBox (Lon Min, Lat Min, Lon Max, Lat Max)",
        "-0.20, 51.46, -0.10, 51.52"
    )
    # Parse the comma‑separated string into a list of floats
    st.session_state["bbox"] = [float(x.strip()) for x in bbox_input.split(',')]

# -- GEE project ID (optional but often needed) --
gee_project = st.sidebar.text_input("GEE Project ID")

# -- Simulation parameters --
intensity   = st.sidebar.slider("Rainfall (mm/hr)",   10.0, 150.0, 50.0)
time_steps  = st.sidebar.slider("Duration (steps)",   6,    24,    12)

# ============================================================================
#   MAIN SIMULATION BUTTON
# ============================================================================
if st.sidebar.button("Run AI Simulation"):
    # Retrieve bounding box from session state (set by map or manual input)
    bbox = st.session_state.get("bbox", [-0.20, 51.46, -0.10, 51.52])

    # --------- Step 1: Fetch data from Google Earth Engine ---------
    with st.spinner("🌍 Fetching 10 m Sentinel-2 data from GEE…"):
        dem, lc, rgb = fetch_regional_data(bbox, project=gee_project.strip())

    # --------- Step 2: Preprocess images into tensors ---------
    with st.spinner("⚙️ Aligning tensors…"):
        # Returns: dem_p (128x128), lc_p (128x128), rain_p (T x 128x128), rgb_p (512x512 uint8)
        dem_p, lc_p, rain_p, rgb_p = prepare_tensors_for_inference(
            dem, lc, rgb, time_steps, intensity
        )

    # Display the high‑resolution satellite background
    st.subheader("City Base Layer (Sentinel-2)")
    st.image(rgb_p, caption="10 m Sentinel-2 RGB", use_container_width=True)

    # --------- Step 3: Run the U-RNN model ---------
    with st.spinner("🧠 Running U-RNN inference…"):
        predicted_depths = run_pipeline(dem_p, lc_p, rain_p)

    # --------- Step 4: Show output statistics ---------
    st.subheader("Model Output Diagnostics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Min depth", f"{predicted_depths.min():.4f} m")
    col2.metric("Max depth", f"{predicted_depths.max():.4f} m")
    col3.metric("Mean depth", f"{predicted_depths.mean():.4f} m")

    # Warn if the model output is essentially zero (weights possibly not loaded)
    if predicted_depths.max() < 1e-4:
        st.error(
            "⚠️ Model output is essentially zero across all timesteps. "
            "The weights are likely not loading correctly (check the terminal "
            "for '[inference] WARNING' messages). The animation will be blank."
        )
    else:
        st.success(f"✅ Flood signal detected — max depth {predicted_depths.max():.3f} m")

    # Show the average depth heatmap for quick validation
    show_diagnostic_heatmap(predicted_depths)

    # --------- Step 5: Generate and display the animation ---------
    with st.spinner("🎨 Rendering flood animation…"):
        gif_path = generate_smooth_flood_gif(rgb_p, predicted_depths)

    display_animated_gif(
        gif_path,
        caption=f"Urban Flood Simulation ({time_steps * 10} mins)"
    )

    # --------- Step 6: Provide download button ---------
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()

    st.download_button(
        "⬇️ Download Animation",
        data=gif_bytes,
        file_name="urnn_flood_sim.gif",
        mime="image/gif"
    )