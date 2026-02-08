from io import BytesIO
import torch
import numpy as np
import os
from omegaconf import OmegaConf
import argparse
import json
from recon.refiner import Refiner, Config

def part1by2_vec(x: torch.Tensor) -> torch.Tensor:
    """Interleave bits of x with 0s

    Args:
        x (torch.Tensor): Input tensor. Shape (N,)

    Returns:
        torch.Tensor: Output tensor. Shape (N,)
    """

    x = x & 0x000003FF
    x = (x ^ (x << 16)) & 0xFF0000FF
    x = (x ^ (x << 8)) & 0x0300F00F
    x = (x ^ (x << 4)) & 0x030C30C3
    x = (x ^ (x << 2)) & 0x09249249
    return x

def encode_morton3_vec(
    x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """Compute Morton codes for 3D coordinates

    Args:
        x (torch.Tensor): X coordinates. Shape (N,)
        y (torch.Tensor): Y coordinates. Shape (N,)
        z (torch.Tensor): Z coordinates. Shape (N,)
    Returns:
        torch.Tensor: Morton codes. Shape (N,)
    """
    return (part1by2_vec(z) << 2) + (part1by2_vec(y) << 1) + part1by2_vec(x)

def sort_centers(centers: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Sort centers based on Morton codes

    Args:
        centers (torch.Tensor): Centers. Shape (N, 3)
        indices (torch.Tensor): Indices. Shape (N,)
    Returns:
        torch.Tensor: Sorted indices. Shape (N,)
    """
    # Compute min and max values in a single operation
    min_vals, _ = torch.min(centers, dim=0)
    max_vals, _ = torch.max(centers, dim=0)

    # Compute the scaling factors
    lengths = max_vals - min_vals
    lengths[lengths == 0] = 1  # Prevent division by zero

    # Normalize and scale to 10-bit integer range (0-1024)
    scaled_centers = ((centers - min_vals) / lengths * 1024).floor().to(torch.int32)

    # Extract x, y, z coordinates
    x, y, z = scaled_centers[:, 0], scaled_centers[:, 1], scaled_centers[:, 2]

    # Compute Morton codes using vectorized operations
    morton = encode_morton3_vec(x, y, z)

    # Sort indices based on Morton codes
    sorted_indices = indices[torch.argsort(morton).to(indices.device)]

    return sorted_indices

def sh2rgb(sh: torch.Tensor) -> torch.Tensor:
    """Convert Sphere Harmonics to RGB

    Args:
        sh (torch.Tensor): SH tensor

    Returns:
        torch.Tensor: RGB tensor
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def splat2splat_bytes(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
) -> bytes:
    """Return the binary Splat file. Supported by antimatter15 viewer.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 3)

    Returns:
        bytes: Binary Splat file representing the model.
    """

    # Preprocess
    scales = torch.exp(scales)
    sh0_color = sh2rgb(sh0)
    colors = torch.cat([sh0_color, torch.sigmoid(opacities).unsqueeze(-1)], dim=1)
    colors = (colors * 255).clamp(0, 255).to(torch.uint8)

    rots = (quats / torch.linalg.norm(quats, dim=1, keepdim=True)) * 128 + 128
    rots = rots.clamp(0, 255).to(torch.uint8)

    # Sort splats
    num_splats = means.shape[0]
    indices = sort_centers(means, torch.arange(num_splats))

    # Reorder everything
    means = means[indices]
    scales = scales[indices]
    colors = colors[indices]
    rots = rots[indices]

    float_dtype = np.dtype(np.float32).newbyteorder("<")
    means_np = means.detach().cpu().numpy().astype(float_dtype)
    scales_np = scales.detach().cpu().numpy().astype(float_dtype)
    colors_np = colors.detach().cpu().numpy().astype(np.uint8)
    rots_np = rots.detach().cpu().numpy().astype(np.uint8)

    buffer = BytesIO()
    for i in range(num_splats):
        buffer.write(means_np[i].tobytes())
        buffer.write(scales_np[i].tobytes())
        buffer.write(colors_np[i].tobytes())
        buffer.write(rots_np[i].tobytes())

    return buffer.getvalue()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_cfg', type=str, required=True, help='exp cfg path')
    parser.add_argument('--base_cfg', type=str, default="exp_cfg/base.yaml", help='base cfg path')
    args = parser.parse_args()
    base_cfg = OmegaConf.load(args.base_cfg)
    exp_cfg = OmegaConf.load(args.exp_cfg)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    with open(os.path.join(cfg.base_dir, cfg.gs_cfg_file), "r") as f:
        config = Config(**json.load(f))
    refiner = Refiner(
        config, 
        load_step=cfg.load_step,
        test_split=cfg.test_split, 
        test_trans=cfg.test_trans, 
        test_rots=cfg.test_rots, 
        c_exp_index=cfg.c_exp_index,
        hessian_attr=cfg.hessian_attr,
        test_len = cfg.refine_end_idx - cfg.refine_start_idx,
        data_type=cfg.data_type,
    )

    data = splat2splat_bytes(refiner.splats['means'], 
                             refiner.splats['scales'], 
                             refiner.splats['quats'], 
                             refiner.splats['opacities'], 
                             refiner.splats['sh0'][:, 0, :])
    with open("dbg/garden.splat", "wb") as binary_file:
            binary_file.write(data)
