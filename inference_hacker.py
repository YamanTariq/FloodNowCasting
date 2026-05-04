import torch
from torch import nn
import glob
import os
import sys
import numpy as np
from collections import OrderedDict


def run_pipeline(dem_p, lc_p, rain_p):
    current_dir = os.getcwd()
    repo_code_path = os.path.join(current_dir, 'U-RNN', 'code')
    if repo_code_path not in sys.path:
        sys.path.insert(0, repo_code_path)

    from src.lib.model.networks.model import ED
    from src.lib.model.networks.ConvRNN import CGRU_cell

    # --- 1. ARCHITECTURE ---
    encoder_subnets = [
        OrderedDict({'conv1': [9,  16, 1, 1, 0]}),
        OrderedDict({'conv2': [64, 64, 1, 2, 0]}),
        OrderedDict({'conv3': [96, 96, 1, 2, 0]})
    ]
    decoder_subnets = [
        OrderedDict({'deconv1': [96, 96, 2, 2, 0]}),
        OrderedDict({'deconv2': [96, 96, 2, 2, 0]}),
        OrderedDict({'deconv3': [64, 16, 1, 1, 0]})
    ]
    e_rnns = nn.ModuleList([
        CGRU_cell(False, (128, 128), 16, 1, 64, "encoder"),
        CGRU_cell(False, (64,  64),  64, 1, 96, "encoder"),
        CGRU_cell(False, (32,  32),  96, 1, 96, "encoder")
    ])
    d_rnns = nn.ModuleList([
        CGRU_cell(False, (32,  32),  96, 1, 96, "decoder"),
        CGRU_cell(False, (64,  64),  96, 1, 96, "decoder"),
        CGRU_cell(False, (128, 128), 96, 1, 64, "decoder")
    ])

    model = ED(
        clstm_flag=False,
        encoder_params=[encoder_subnets, e_rnns],
        decoder_params=[decoder_subnets, d_rnns],
        input_height=128,
        input_width=128,
        use_checkpoint=False
    )

    # --- 2. WEIGHT LOADING ---
    ckpt_list = glob.glob('/content/exp/**/save_model/*.pth.tar', recursive=True)
    checkpoint = torch.load(ckpt_list[0], map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('_wrapper.', '.')
        name = name.replace('_leaky_1', '').replace('_relu_1', '')
        name = name.replace('conv1_module_wrapper.0', 'conv1.0')
        name = name.replace('conv1_module_wrapper.1', 'conv1.1')
        name = name.replace('conv2_module_wrapper.0', 'conv2.0')
        name = name.replace('conv2_module_wrapper.1', 'conv2.1')
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # --- 3. EXECUTION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    S, H, W = rain_p.shape

    rain_norm = rain_p / 60.0
    dem_min, dem_max = dem_p.min(), dem_p.max()
    dem_norm = (dem_p - dem_min) / (dem_max - dem_min + 1e-6)

    input_t = torch.zeros((1, S, 9, H, W)).to(device)
    input_t[0, :, 0, :, :] = torch.FloatTensor(rain_norm).to(device)
    input_t[0, :, 1, :, :] = torch.FloatTensor(dem_norm).to(device)

    def get_h(dim, s):
        return torch.zeros((1, dim, H // s, W // s)).to(device)

    e1, e2, e3 = get_h(64, 1), get_h(96, 2), get_h(96, 4)
    d1, d2, d3 = get_h(96, 4), get_h(96, 2), get_h(64, 1)

    with torch.no_grad():
        results = model(input_t, e1, e2, e3, d1, d2, d3)

    # --- 4. ROBUST OUTPUT EXTRACTION ---
    raw = results[0] if isinstance(results, (list, tuple)) else results
    raw = raw.cpu().float()

    if raw.ndim == 5:
        raw = raw[0, :, 0, :, :]       # (1, S, 1, H, W) → (S, H, W)
    elif raw.ndim == 4:
        if raw.shape[0] == 1:
            raw = raw[0]               # (1, S, H, W) → (S, H, W)
        else:
            raw = raw[:, 0, :, :]      # (S, 1, H, W) → (S, H, W)
    # raw: (S, H, W)

    depths_np = raw.numpy()

    # Print raw stats — visible in Colab logs / Streamlit terminal
    print(f"[inference] raw output  min={depths_np.min():.4f}  max={depths_np.max():.4f}  "
          f"mean={depths_np.mean():.4f}  shape={depths_np.shape}")

    # --- 5. ADAPTIVE NORMALISATION ---
    # Do NOT apply sigmoid here.  The model may output:
    #   (a) already-normalised [0,1] values
    #   (b) raw logits in an arbitrary range  (large negative → sigmoid → ~0 → invisible)
    #   (c) direct depth predictions in mm or m
    #
    # Safe approach: shift floor to 0, then scale by 99th-percentile.
    # The spatial flood pattern is preserved regardless of the original range.
    d_min = depths_np.min()
    depths_shifted = depths_np - d_min      # floor = 0, all values ≥ 0

    p99 = float(np.percentile(depths_shifted, 99))

    if p99 < 1e-6:
        print("[inference] WARNING: model output is flat/zero. "
              "Weights may not be loading correctly.")
        return depths_shifted             # return zeros; app will show the warning

    depths_norm = np.clip(depths_shifted / p99, 0.0, 1.0)

    # Scale to metres (flood_max ≈ 5 m)
    depths_meters = depths_norm * 5.0

    print(f"[inference] scaled depth  min={depths_meters.min():.3f} m  "
          f"max={depths_meters.max():.3f} m")

    return depths_meters