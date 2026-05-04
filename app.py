import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import os, base64
from gee_pipeline import fetch_regional_data
from dip_engine import prepare_tensors_for_inference
from inference_hacker import run_pipeline

st.set_page_config(page_title="U-RNN Flood Sim", layout="wide")
st.title("🌊 U-RNN Urban Flood Nowcasting")

# --- SMOOTH VISUALIZATION LOGIC ---
def generate_smooth_flood_gif(rgb_base, predicted_depths):
    # Ensure RGB is 0-255 uint8
    if rgb_base.max() <= 1.0:
        rgb_base = (rgb_base * 255).astype(np.uint8)
        
    # Gentle brightening (so the city looks natural but visible)
    rgb_bright = cv2.convertScaleAbs(rgb_base, alpha=1.2, beta=15)

    cmap = plt.cm.Blues.copy()
    cmap.set_under('black', alpha=0) 

    gif_path = "flood_animation.gif"
    frames = []
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150) # Higher DPI for sharper text
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    
    for t in range(predicted_depths.shape[0]):
        ax.clear()
        ax.axis('off')
        
        # 1. Draw the High-Res City Map
        ax.imshow(rgb_bright)
        
        # 2. Get the 128x128 AI output
        depth_128 = predicted_depths[t]
        
        # 3. Upscale the AI water mask to match the 512x512 High-Res Map
        depth_highres = cv2.resize(depth_128, (rgb_bright.shape[1], rgb_bright.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # 4. Blur for fluid realism
        smoothed_depth = cv2.GaussianBlur(depth_highres, (7, 7), 0)
        
        # Mask out dry land (< 5cm water)
        masked_depth = np.ma.masked_where(smoothed_depth < 0.05, smoothed_depth)
        
        # Overlay water with alpha=0.6
        im = ax.imshow(masked_depth, cmap=cmap, vmin=0.05, vmax=1.0, interpolation='bilinear', alpha=0.6)
        
        # Add timestamp
        ax.text(10, 30, f"Simulation: +{t*10} mins", color='white', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=4))
        
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3] 
        frames.append(image)

    plt.close(fig)
    imageio.mimsave(gif_path, frames, fps=4, loop=0)
    return gif_path

# --- UI CONTROLS ---
st.sidebar.header("Simulation Settings")

# Zoomed out BBox to cover Central London (More recognizable city blocks & river)
bbox_input = st.sidebar.text_input("BBox (Lon Min, Lat Min, Lon Max, Lat Max)", "-0.20, 51.46, -0.10, 51.52")
gee_project = st.sidebar.text_input("GEE Project ID")
intensity = st.sidebar.slider("Rainfall (mm/hr)", 10.0, 150.0, 50.0)
time_steps = st.sidebar.slider("Duration (steps)", 6, 24, 12)

if st.sidebar.button("Run AI Simulation"):
    bbox = [float(x.strip()) for x in bbox_input.split(',')]
    
    with st.spinner("🌍 Fetching 10m Sentinel-2 Data from GEE..."):
        dem, lc, rgb = fetch_regional_data(bbox, project=gee_project.strip())
        
    with st.spinner("⚙️ Aligning Tensors..."):
        dem_p, lc_p, rain_p, rgb_p = prepare_tensors_for_inference(dem, lc, rgb, time_steps, intensity)
        
    with st.spinner("🧠 Running REAL U-RNN Inference..."):
        predicted_depths = run_pipeline(dem_p, lc_p, rain_p)
        
    with st.spinner("🎨 Rendering Smooth Heatmap..."):
        gif_path = generate_smooth_flood_gif(rgb_p, predicted_depths)
        
    # Read the GIF bytes for native Streamlit playback
    with open(gif_path, "rb") as f:
        gif_bytes = f.read()
        
    # Play animated GIF in UI
    st.image(gif_bytes, caption=f"Urban Flood Simulation ({time_steps*10} mins)", use_container_width=True)
    
    # Download button
    st.download_button("Download High-Res Animation", data=gif_bytes, file_name="urnn_flood_sim.gif", mime="image/gif")