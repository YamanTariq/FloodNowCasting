import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
import os
import base64
from gee_pipeline import fetch_regional_data
from dip_engine import prepare_tensors_for_inference
from inference_hacker import run_pipeline

st.set_page_config(page_title="U-RNN Flood Sim", layout="wide")
st.title("🌊 U-RNN Urban Flood Nowcasting")


# ---------------------------------------------------------------------------
# VISUALIZATION ENGINE
# ---------------------------------------------------------------------------

def compute_gsd(bbox):
    """
    Ground Sampling Distance (metres/pixel) of the 128×128 model output
    for a given bounding box. Tells you what spatial detail is achievable.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    # 1 degree latitude ≈ 111,320 m; longitude corrected for lat
    import math
    lat_mid  = (min_lat + max_lat) / 2
    span_m_x = abs(max_lon - min_lon) * 111_320 * math.cos(math.radians(lat_mid))
    span_m_y = abs(max_lat - min_lat) * 111_320
    gsd_x = span_m_x / 128
    gsd_y = span_m_y / 128
    return gsd_x, gsd_y, span_m_x, span_m_y


def generate_smooth_flood_gif(rgb_base, predicted_depths, gsd_m=55.0):
    """
    Renders an animated GIF of the flood simulation overlaid on the city map.

    gsd_m : ground sampling distance of the model output in metres/pixel.
            Used to scale the Gaussian blur kernel — at coarse GSD a large
            kernel makes blobs even worse; at fine GSD a small kernel preserves
            genuine street-level spatial detail.
    """

    # ------------------------------------------------------------------
    # 1. Sanitise the RGB base image
    # ------------------------------------------------------------------
    rgb = np.squeeze(rgb_base)  # Drop any singleton dimensions

    # Handle channel-first layout (C, H, W) → (H, W, C)
    if rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4) and rgb.shape[0] < rgb.shape[-1]:
        rgb = np.transpose(rgb, (1, 2, 0))

    # Ensure 3-channel RGB
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.shape[-1] == 1:
        rgb = np.concatenate([rgb, rgb, rgb], axis=-1)
    elif rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    # Normalise to uint8 [0, 255]
    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        else:
            rgb = rgb.clip(0, 255).astype(np.uint8)

    # Gentle contrast boost so the city map pops
    rgb_bright = cv2.convertScaleAbs(rgb, alpha=1.35, beta=18)

    # ------------------------------------------------------------------
    # 2. Compute adaptive depth range from actual model output
    # ------------------------------------------------------------------
    THRESHOLD = 0.03          # metres — below this = dry land, fully transparent
    flat = predicted_depths[predicted_depths > THRESHOLD]

    if flat.size > 0:
        vmax = float(np.percentile(flat, 95))
        vmax = max(vmax, 0.10)   # at least 10 cm headroom
        vmax = min(vmax, 5.0)    # cap at 5 m (model's physical max)
    else:
        vmax = 0.5               # fallback if model outputs nothing useful

    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.cm.Blues

    # ------------------------------------------------------------------
    # 3. Build figure with a fixed colorbar (drawn once, not per frame)
    # ------------------------------------------------------------------
    H_out, W_out = rgb_bright.shape[:2]
    fig = plt.figure(figsize=(7, 6.5), dpi=150, facecolor='#0a0a0a')
    ax  = fig.add_axes([0.03, 0.03, 0.82, 0.90])   # main map axes
    cax = fig.add_axes([0.88, 0.12, 0.025, 0.55])  # colorbar axes

    ax.axis('off')
    ax.set_facecolor('#0a0a0a')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Water depth (m)', color='#cccccc', fontsize=7, labelpad=6)
    cb.ax.yaxis.set_tick_params(color='#cccccc', labelsize=6)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#cccccc')
    cax.set_facecolor('#0a0a0a')

    # ------------------------------------------------------------------
    # 4. Frame loop
    # ------------------------------------------------------------------
    frames = []
    bg_handle   = None   # imshow handle for the city map
    over_handle = None   # imshow handle for the flood overlay

    for t in range(predicted_depths.shape[0]):
        depth_128 = predicted_depths[t]  # shape (128, 128)

        # --- Upscale AI output to match the city map resolution ---
        depth_hi = cv2.resize(
            depth_128, (W_out, H_out), interpolation=cv2.INTER_CUBIC
        )

        # --- Adaptive blur: scale kernel to actual ground resolution ---
        # At 55 m/px (7 km bbox) even k=3 spreads 165 m — already too much.
        # At 16 m/px (2 km bbox) k=5 spreads only 80 m — natural-looking.
        # We target a physical spread of ~40 m regardless of zoom level.
        k = max(3, int(round(40.0 / gsd_m)) * 2 + 1)   # must be odd
        smoothed = cv2.GaussianBlur(depth_hi, (k, k), 0)

        # --- Build RGBA overlay ----------------------------------------
        # cmap() maps [0,1] → RGBA; we normalise depth then apply alpha.
        depth_norm = norm(np.clip(smoothed, 0, vmax))       # [0,1]
        flood_rgba = cmap(depth_norm).astype(np.float32)    # (H, W, 4)

        water_mask = smoothed >= THRESHOLD

        # Alpha: dry → 0 (transparent); wet → scales 0.25→0.75 with depth
        alpha_layer = np.where(water_mask, 0.25 + 0.50 * depth_norm, 0.0)
        flood_rgba[..., 3] = alpha_layer.astype(np.float32)

        # --- Draw or update imshow layers (reuse handles = much faster) ---
        if bg_handle is None:
            bg_handle   = ax.imshow(rgb_bright,  aspect='auto', interpolation='lanczos')
            over_handle = ax.imshow(flood_rgba,  aspect='auto', interpolation='bilinear')
        else:
            over_handle.set_data(flood_rgba)

        # --- Timestamp ---
        # Remove previous text artists to avoid stacking
        for txt in ax.texts:
            txt.remove()

        ax.text(
            0.02, 0.97,
            f"T + {t * 10:3d} min",
            transform=ax.transAxes,
            color='white', fontsize=10, fontweight='bold', va='top',
            bbox=dict(facecolor='#0a0a0a', alpha=0.65, edgecolor='none', pad=3)
        )

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(frame)

    plt.close(fig)

    gif_path = "flood_animation.gif"
    imageio.mimsave(gif_path, frames, fps=4, loop=0)
    return gif_path


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------

st.sidebar.header("Simulation Settings")

bbox_input  = st.sidebar.text_input(
    "BBox (Lon Min, Lat Min, Lon Max, Lat Max)",
    # ~4 km × 3.3 km around central London — ~31 m/px → good block detail without pixelation
    "-0.14, 51.488, -0.08, 51.518"
)
gee_project = st.sidebar.text_input("GEE Project ID")
intensity   = st.sidebar.slider("Rainfall (mm/hr)", 10.0, 150.0, 50.0)
time_steps  = st.sidebar.slider("Duration (steps)", 6, 24, 12)

# Live resolution calculator
try:
    _bbox = [float(x.strip()) for x in bbox_input.split(',')]
    _gx, _gy, _sx, _sy = compute_gsd(_bbox)
    _gsd_avg = (_gx + _gy) / 2
    if _gsd_avg <= 20:
        _res_label = "🟢 Street-block detail (~{:.0f} m/px)".format(_gsd_avg)
    elif _gsd_avg <= 40:
        _res_label = "🟡 District detail (~{:.0f} m/px)".format(_gsd_avg)
    else:
        _res_label = "🔴 City-scale blobs (~{:.0f} m/px) — zoom in for detail".format(_gsd_avg)
    st.sidebar.caption(
        f"**Model resolution:** {_gsd_avg:.0f} m/px  \n"
        f"Area: {_sx/1000:.1f} × {_sy/1000:.1f} km  \n{_res_label}"
    )
except Exception:
    _gsd_avg = 55.0

if st.sidebar.button("Run AI Simulation"):
    bbox = [float(x.strip()) for x in bbox_input.split(',')]
    gsd_x, gsd_y, span_x, span_y = compute_gsd(bbox)
    gsd_avg = (gsd_x + gsd_y) / 2

    with st.spinner("🌍 Fetching 10 m Sentinel-2 data from GEE…"):
        dem, lc, rgb = fetch_regional_data(bbox, project=gee_project.strip())

    with st.spinner("⚙️ Aligning tensors…"):
        dem_p, lc_p, rain_p, rgb_p = prepare_tensors_for_inference(
            dem, lc, rgb, time_steps, intensity
        )

    with st.spinner("🧠 Running U-RNN inference…"):
        predicted_depths = run_pipeline(dem_p, lc_p, rain_p)

    # ── Post-processing: suppress RNN hidden-state instability artifacts ──
    # ConvGRU gates saturate outside training distribution, causing:
    #   EXPLOSION  — update gate over-opens → small input change floods everything
    #   COLLAPSE   — gate fully closes → output decays to zero despite high rainfall
    # Suppressed via three conservative steps that don't touch model or pipeline:

    stabilised = predicted_depths.copy()

    # Step A: per-frame spatial outlier clamp (explosion artifacts sit at 10-100σ)
    for t in range(stabilised.shape[0]):
        frame = stabilised[t]
        if frame.max() > 0:
            ceiling = frame.mean() + 3.0 * frame.std()
            stabilised[t] = np.clip(frame, 0.0, ceiling)

    # Step B: temporal rolling mean over 3 steps — kills single-frame chaotic spikes
    from scipy.ndimage import uniform_filter1d
    stabilised = uniform_filter1d(stabilised, size=3, axis=0, mode='reflect')

    # Step C: physical floor / ceiling
    predicted_depths = np.clip(stabilised, 0.0, 5.0)

    # ── Depth diagnostics (useful for debugging model output) ─────────────
    min_d  = predicted_depths.min()
    max_d  = predicted_depths.max()
    mean_d = predicted_depths.mean()
    wet_px = (predicted_depths > 0.03).sum()
    st.info(
        f"**Model output** — min: `{min_d:.3f} m`  max: `{max_d:.3f} m`  "
        f"mean: `{mean_d:.3f} m`  wet pixels: `{wet_px}`"
    )

    with st.spinner("🎨 Rendering animation…"):
        gif_path = generate_smooth_flood_gif(rgb_p, predicted_depths, gsd_m=gsd_avg)

    with open(gif_path, "rb") as f:
        gif_bytes = f.read()

    # ── FIX: st.image() does NOT animate GIFs — use HTML instead ──────────
    gif_b64 = base64.b64encode(gif_bytes).decode("utf-8")
    st.markdown(
        f"""
        <div style="text-align:center;">
          <img
            src="data:image/gif;base64,{gif_b64}"
            style="width:100%; max-width:900px; border-radius:6px;"
            alt="Flood simulation animation"
          />
          <p style="color:#888; font-size:0.85em; margin-top:6px;">
            Urban flood simulation — {time_steps * 10} min forecast
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.download_button(
        "⬇️ Download animation (.gif)",
        data=gif_bytes,
        file_name="urnn_flood_sim.gif",
        mime="image/gif",
    )