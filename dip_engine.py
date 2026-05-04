import cv2
import numpy as np

def prepare_tensors_for_inference(dem, lc, rgb, time_steps, intensity):
    # The AI STRICTLY needs 128x128 for its tensors
    H, W = 128, 128
    dem_p = cv2.resize(dem.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    lc_p = cv2.resize(lc.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Keep the visualization background in HIGH RESOLUTION (512x512)
    rgb_p = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    # Rainfall scaled to 10-minute increments for the RNN
    rain_p = np.full((time_steps, H, W), intensity / 6.0, dtype=np.float32) 
    return dem_p, lc_p, rain_p, rgb_p