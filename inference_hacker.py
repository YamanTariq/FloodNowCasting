"""
inference_hacker.py – Faithful U‑RNN inference using the original preprocessing
and step‑by‑step forward pass.

This version exactly replicates the Inference function in test.py, so:
  • The correct 9‑channel input is built every timestep (historical rain,
    cumulative rain, DEM, imperviousness, manhole).
  • Hidden states are carried forward timestep by timestep.
  • Output is post‑processed identically (mm → metres).
"""

import torch
import glob
import os
import sys
import numpy as np


def run_pipeline(dem_p, lc_p, rain_p):
    """
    Run U‑RNN flood model step‑by‑step exactly as the original Inference() does.

    Parameters
    ----------
    dem_p : np.ndarray, shape (128, 128)
        Digital elevation model (float32, in metres).
    lc_p  : np.ndarray, shape (128, 128)
        Land cover (ESA WorldCover classes). Used to derive impervious fraction.
    rain_p: np.ndarray, shape (T, 128, 128)
        Rainfall intensity per 10‑minute step (mm/step).

    Returns
    -------
    depths_meters : np.ndarray, shape (T, 128, 128)
        Flood water depth in metres.
    """
    # ------------------------------------------------------------------
    # 0. Make the U‑RNN code importable
    # ------------------------------------------------------------------
    current_dir = os.getcwd()
    repo_code = os.path.join(current_dir, "U-RNN", "code")
    if repo_code not in sys.path:
        sys.path.insert(0, repo_code)

    from src.lib.model.networks.model import ED
    from src.lib.model.networks.net_params import get_network_params
    from src.lib.dataset.Dynamic2DFlood import (
        preprocess_inputs, MinMaxScaler, r_MinMaxScaler
    )
    from src.lib.utils.general import initialize_states

    S, H, W = rain_p.shape       # T steps, 128×128

    # ------------------------------------------------------------------
    # 1. Build the model (identical to original lite checkpoint)
    # ------------------------------------------------------------------
    use_checkpoint = False       # inference only
    input_channels = 9           # historical_nums=3 → 3*2+3

    enc_params, dec_params = get_network_params(
        use_checkpoint,
        input_height=H,
        input_width=W,
        input_channels=input_channels,
        net_cfg=None             # built‑in defaults match network.yaml
    )

    model = ED(
        clstm_flag=False,
        encoder_params=enc_params,
        decoder_params=dec_params,
        cls_thred=0.5,
        use_checkpoint=use_checkpoint,
        input_height=H,
        input_width=W,
    )

    # Load weights
    ckpt_list = glob.glob("/content/exp/**/save_model/*.pth.tar", recursive=True)
    if not ckpt_list:
        raise FileNotFoundError("No checkpoint found under /content/exp/")
    checkpoint_path = ckpt_list[0]
    print(f"[inference] Loading {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # The checkpoint may have 'module.' prefix if saved with DataParallel
    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

        # ------------------------------------------------------------------
    # 2. Convert our data to the format expected by preprocess_inputs
    # ------------------------------------------------------------------

    # --- Derive imperviousness from land cover ---
    # WorldCover classes:
    #   50 = Built-up → fully impervious (1.0)
    #   40 = Bare/sparse vegetation → partially impervious (0.3)
    #   All others → 0 (pervious)
    impervious = np.zeros_like(lc_p, dtype=np.float32)
    impervious[lc_p == 50] = 1.0
    impervious[lc_p == 40] = 0.3

    # Smooth with a mean filter to avoid sharp artificial boundaries.
    # Use scipy.ndimage.uniform_filter if available; otherwise a simple
    # OpenCV box filter (cv2.blur) works just as well.
    try:
        from scipy.ndimage import uniform_filter
        impervious = uniform_filter(impervious.astype(np.float32), size=3)
    except ImportError:
        import cv2
        impervious = cv2.blur(impervious, (3, 3))

        # --- Convert DEM to mm (original pipeline works in mm) ---
    dem_mm = dem_p * 1000.0                     # metres → mm

    # --- Rainfall (already mm/10min, shape (T, H, W)) ---
    # Build exactly as DataLoader would: (B=1, T, 1, H, W)
    rainfall_torch = torch.from_numpy(rain_p).float()          # (T, H, W)
    rainfall_torch = rainfall_torch.unsqueeze(1)               # (T, 1, H, W)
    rainfall_torch = rainfall_torch.unsqueeze(0)               # (1, T, 1, H, W)

    # Cumulative rainfall from event start
    cumsum = torch.cumsum(rainfall_torch, dim=1)              # (1, T, 1, H, W)

    # --- Static maps: add batch, channel, and a *singleton channel* dim ---
    # The concatenation in preprocess_inputs expects every tensor to be 5D:
    #   (B, 1, channels, H, W)
    # We add a size‑1 channel dimension for the three static maps.
    abs_dem = torch.from_numpy(dem_mm).float().unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1,1,1,H,W)
    imp = torch.from_numpy(impervious).float().unsqueeze(0).unsqueeze(0).unsqueeze(2)  # (1,1,1,H,W)
    manhole = torch.zeros(1, 1, 1, H, W)                                              # (1,1,1,H,W)

    max_dem = abs_dem.max().item()
    min_dem = abs_dem.min().item()

    # Build the inputs dict exactly matching the original dataset + DataLoader
    inputs = {
        "absolute_DEM":   abs_dem,                   # (1, 1, 1, H, W)
        "max_DEM":        torch.tensor([max_dem]),   # (1,)
        "min_DEM":        torch.tensor([min_dem]),   # (1,)
        "impervious":     imp,                       # (1, 1, 1, H, W)
        "manhole":        manhole,                   # (1, 1, 1, H, W)
        "rainfall":       rainfall_torch,            # (1, T, 1, H, W)
        "cumsum_rainfall": cumsum,                   # (1, T, 1, H, W)
    }
    
    # Normalisation constants from lite.yaml
    rain_max = 60.0            # mm/10min
    cumsum_rain_max = 250.0    # mm
    historical_nums = 3

    # ------------------------------------------------------------------
    # 3. Inference loop (exact copy of Inference function in test.py)
    # ------------------------------------------------------------------
    with torch.no_grad():
        Frames = S
        # Initialize hidden states
        e1, e2, e3, d1, d2, d3 = initialize_states(
            device,
            input_height=H,
            input_width=W,
            net_cfg=None          # use defaults: 64,96,96 etc.
        )

        output_data = []
        for t in range(Frames):
            # preprocess_inputs builds the 9‑channel tensor for this timestep
            input_t = preprocess_inputs(
                t, inputs, device,
                nums=historical_nums,
                rain_max=rain_max,
                cumsum_rain_max=cumsum_rain_max,
            )   # shape: (B, 1, C=9, H, W)

            # Forward one step
            output, e1, e2, e3, d1, d2, d3 = model(
                input_t, e1, e2, e3, d1, d2, d3
            )

            output_data.append(output)

    # ------------------------------------------------------------------
    # 4. Post‑process output → metres
    # ------------------------------------------------------------------
    # output is (1, 1, H, W) after head; stacked → (T, 1, H, W)
    # r_MinMaxScaler converts normalized depth to mm using flood_max=5000
    FLOOD_MAX = 5000.0  # mm
    pred_mm = []
    for out in output_data:
        # out shape: (1, 1, H, W)
        np_out = out.cpu().numpy()[0, 0]   # (H, W)
        pred_mm.append(r_MinMaxScaler(np_out, max=FLOOD_MAX, min=0))
    pred_mm = np.array(pred_mm)            # (T, H, W)

    # Convert to metres
    depths_meters = pred_mm / 1000.0

    print(f"[inference] final depth  min={depths_meters.min():.3f} m  "
          f"max={depths_meters.max():.3f} m")

    return depths_meters