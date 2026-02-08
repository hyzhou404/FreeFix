import numpy as np
from typing import List
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import os
import math
from imageio.v2 import imread

m_psnr = PeakSignalNoiseRatio(data_range=1.0).to('cuda')
m_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
m_lpips = LearnedPerceptualImagePatchSimilarity().to('cuda')


def rgb_to_sh(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def read_images(image_folder):
    image_paths = sorted(os.listdir(image_folder))
    images = []
    for image_path in image_paths:
        image = imread(os.path.join(image_folder, image_path))
        images.append(image / 255.)
    return np.stack(images)

def batch_c2w_distances(query_c2w, c2ws, w_trans=1.0, w_rot=2.0):
    """
    query_c2w: 4x4
    c2ws: Nx4x4
    """

    t_q = query_c2w[:3, 3]
    t_k_batch = c2ws[:, :3, 3]

    # transmit distance
    d_trans_batch = np.linalg.norm(t_q - t_k_batch, axis=1)

    # rotate distance
    R_q = query_c2w[:3, :3]
    R_k_batch = c2ws[:, :3, :3]
    R_k_batch_transposed = R_k_batch.transpose(0, 2, 1)
    R_rel_batch = np.matmul(R_q, R_k_batch_transposed)
    trace_R_batch = np.trace(R_rel_batch, axis1=1, axis2=2)
    cos_theta_batch = np.clip((trace_R_batch - 1) / 2, -1.0, 1.0)
    angle_batch = np.arccos(cos_theta_batch)

    # weighted add
    total_distances = w_trans * d_trans_batch + w_rot * angle_batch

    return total_distances


def project_warp(prev_image, prev_c2w, prev_depth, img, c2w, ixt, depth=None, occ_thres=0.1):
    H, W = prev_depth.shape
    fx, fy = ixt[0, 0], ixt[1, 1]
    cx, cy = ixt[0, 2], ixt[1, 2]

    warped_image = img[0].clone()
    projection_mask = torch.zeros((H, W), dtype=bool)

    # unproject
    v_coords, u_coords = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=img.device),
                                        torch.arange(W, dtype=torch.float32, device=img.device),
                                        indexing='ij') # 'ij' ensures (row, col) indexing.indices((H, W))
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()
    depth_flat = prev_depth.flatten()

    valid_depth_mask = (depth_flat > 1e-6) & torch.isfinite(depth_flat)
    u_valid = u_flat[valid_depth_mask]
    v_valid = v_flat[valid_depth_mask]
    depth_valid = depth_flat[valid_depth_mask]

    x_c_valid = (u_valid - cx) * depth_valid / fx
    y_c_valid = (v_valid - cy) * depth_valid / fy
    z_c_valid = depth_valid
    P_c_valid = torch.stack([x_c_valid, y_c_valid, z_c_valid, torch.ones_like(z_c_valid)], axis=0)
    P_w = prev_c2w @ P_c_valid

    # project to warp view coord
    w2c = torch.linalg.inv(c2w)
    P_c_new = w2c @ P_w
    z_new = P_c_new[2, :]
    valid_proj_mask = (z_new > 1e-6) & torch.isfinite(z_new)

    # project to pixel
    u_proj_valid_filtered = torch.full_like(u_valid, -1.0, dtype=torch.float32)
    v_proj_valid_filtered = torch.full_like(v_valid, -1.0, dtype=torch.float32)
    if torch.any(valid_proj_mask):
        u_proj_valid_filtered[valid_proj_mask] = (P_c_new[0, valid_proj_mask] * fx / z_new[valid_proj_mask]) + cx
        v_proj_valid_filtered[valid_proj_mask] = (P_c_new[1, valid_proj_mask] * fy / z_new[valid_proj_mask]) + cy
    valid_bounds_mask = (u_proj_valid_filtered >= 0) & (u_proj_valid_filtered < W) & \
                        (v_proj_valid_filtered >= 0) & (v_proj_valid_filtered < H)
    final_target_u_coords_float = u_proj_valid_filtered[valid_proj_mask][valid_bounds_mask]
    final_target_v_coords_float = v_proj_valid_filtered[valid_proj_mask][valid_bounds_mask]
    final_target_u_int = torch.round(final_target_u_coords_float).int()
    final_target_v_int = torch.round(final_target_v_coords_float).int()
    final_target_u_int = torch.clip(final_target_u_int, 0, W - 1)
    final_target_v_int = torch.clip(final_target_v_int, 0, H - 1)

    # paint
    prev_image_flat = prev_image.reshape(prev_image.shape[1], -1)
    original_flat_indices = torch.where(valid_depth_mask)[0]
    final_source_indices = original_flat_indices[valid_proj_mask][valid_bounds_mask]
    warped_image[:, final_target_v_int, final_target_u_int] = prev_image_flat[:, final_source_indices]
    projection_mask[final_target_v_int, final_target_u_int] = True

    return warped_image, projection_mask


def query_warp(prev_img, prev_c2w, prev_depth, img, c2w, ixt, depth, occ_thres=0.1):
    device = img.device
    H, W = depth.shape
    prev_w2c = torch.linalg.inv(prev_c2w)
    w2c = torch.linalg.inv(c2w)
    ixt_inv = torch.linalg.inv(ixt)

    # unproject
    v_coords, u_coords = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                                        torch.arange(W, dtype=torch.float32, device=device),
                                        indexing='ij') # 'ij' ensures (row, col) indexing
    pixel_coords_hom = torch.stack([u_coords, v_coords, torch.ones_like(u_coords)], dim=-1) # HxWx3
    pixel_coords_hom = pixel_coords_hom.view(-1, 3).T # 3x(H*W)
    depth_flat = depth.view(1, -1) # 1x(H*W)
    P_curr_cam = ixt_inv @ pixel_coords_hom * depth_flat # 3x(H*W)
    P_curr_cam_hom = F.pad(P_curr_cam, (0, 0, 0, 1), mode='constant', value=1.0) # 4x(H*W)
    P_world_hom = c2w @ P_curr_cam_hom

    # to prev cam coord
    P_prev_cam_hom = prev_w2c @ P_world_hom
    P_prev_cam = P_prev_cam_hom[:3, :] # 3x(HxW)
    P_prev_proj_hom = ixt @ P_prev_cam
    valid_depth_projection_mask = P_prev_proj_hom[2, :] > 0 
    u_proj = P_prev_proj_hom[0, :] / P_prev_proj_hom[2, :]
    v_proj = P_prev_proj_hom[1, :] / P_prev_proj_hom[2, :]

    # valid indices
    valid_image_bounds_mask = (u_proj >= 0) & (u_proj < W) & \
                              (v_proj >= 0) & (v_proj < H)
    valid_mask_flat = valid_depth_projection_mask & valid_image_bounds_mask
    u_proj_valid = torch.full((H * W,), float('nan'), device=device)
    v_proj_valid = torch.full((H * W,), float('nan'), device=device)
    if valid_mask_flat.any():
        valid_indices = valid_mask_flat.nonzero(as_tuple=True)[0]
        u_proj_valid[valid_indices] = u_proj[valid_mask_flat]
        v_proj_valid[valid_indices] = v_proj[valid_mask_flat]
    grid_u_norm = (u_proj_valid / (W - 1)) * 2 - 1
    grid_v_norm = (v_proj_valid / (H - 1)) * 2 - 1

    # occlusion check
    grid_for_depth_sample = torch.stack([grid_u_norm, grid_v_norm], dim=-1).view(1, H, W, 2) # 1xHxWx2
    sampled_prev_depth_tensor = F.grid_sample(prev_depth.unsqueeze(0).unsqueeze(0), 
                                              grid_for_depth_sample,
                                              mode='bilinear', padding_mode='border', align_corners=True)
    sampled_prev_depth = sampled_prev_depth_tensor.squeeze(0).squeeze(0) # HxW
    depth_in_prev_cam = P_prev_cam[2, :].view(H, W)
    occlusion_mask = (torch.abs(depth_in_prev_cam - sampled_prev_depth) > occ_thres)
    final_valid_mask_flat = valid_mask_flat.view(H, W) & (~occlusion_mask)
    final_valid_mask_flat = final_valid_mask_flat.view(-1)

    # interpolate ibr
    u_prev_src_flat = torch.full((H * W,), float('nan'), device=device)
    v_prev_src_flat = torch.full((H * W,), float('nan'), device=device)
    if final_valid_mask_flat.any():
        final_valid_indices = final_valid_mask_flat.nonzero(as_tuple=True)[0]
        u_prev_src_flat[final_valid_indices] = u_proj[final_valid_mask_flat]
        v_prev_src_flat[final_valid_indices] = v_proj[final_valid_mask_flat]
    u_prev_src = u_prev_src_flat.view(H, W)
    v_prev_src = v_prev_src_flat.view(H, W)
    grid_u_norm = (u_prev_src / (W - 1)) * 2 - 1
    grid_v_norm = (v_prev_src / (H - 1)) * 2 - 1
    grid = torch.stack([grid_u_norm, grid_v_norm], dim=-1) # HxWx2
    # invalid_curr_depth_mask = (depth <= 0)
    # grid[invalid_curr_depth_mask] = float('nan')
    ibr_interpolated = F.grid_sample(prev_img, grid.unsqueeze(0),
                                      mode='nearest', padding_mode='border', align_corners=True) # (1, C, H, W)
    ibr_img = img.clone()
    final_valid_mask_2d = final_valid_mask_flat.view(H, W)
    ibr_img[0, :, final_valid_mask_2d] = ibr_interpolated[0, :, final_valid_mask_2d]
    
    return ibr_img, final_valid_mask_2d


def eval(images: np.ndarray, gt_images: np.ndarray):
    images = torch.from_numpy(images).float().permute(0, 3, 1, 2).cuda()
    gt_images = torch.from_numpy(gt_images).float().permute(0, 3, 1, 2).cuda()
    psnr = m_psnr(images, gt_images)
    ssim = m_ssim(images, gt_images)
    lpips = m_lpips(images, gt_images)
    return psnr, ssim, lpips


def neighbor_L1_loss(rendered_img, gt_img):
    """Compute minimum loss between rendered image and shifted ground truth.
    Args:
        rendered_img: (B, H, W, 3)
        gt_img: (B, H, W, 3)
        valid_mask: (B, H, W)
        weight_confidence: (B, 1)
    """
    shifts = torch.tensor([[-1,-1], [-1,0], [-1,1],
                         [0,-1],   [0,0],  [0,1],
                         [1,-1],   [1,0],  [1,1]], device=rendered_img.device)  # (9, 2)
    
    B, H, W = rendered_img.shape[:3]
    rendered_img_flat = rendered_img.reshape(B, -1, 3)      # (B, H*W, 3)
    
    # Stack shifted versions of gt_img and valid_mask
    gt_shifted = torch.stack([
        torch.roll(torch.roll(gt_img, dy.item(), dims=1), dx.item(), dims=2)
        for dy, dx in shifts
    ])  # (9, B, H, W, 3)
    
    # Reshape for parallel computation
    gt_shifted_flat = gt_shifted.reshape(9, B, -1, 3)       # (9, B, H*W, 3)
    
    # Compute L1 loss for all shifts at once
    rgb_loss = F.l1_loss(
        rendered_img_flat.unsqueeze(0).expand(9, -1, -1, -1),  # (9, B, H*W, 3)
        gt_shifted_flat,
        reduction='none'
    ).mean(dim=-1)  # (9, B, H*W)
    
    # Apply mask and weight
    rgb_loss = rgb_loss.mean(dim=-1)  # (9, B)
    
    # Get minimum loss across all shifts
    min_rgb_loss = rgb_loss.min(dim=0)[0].mean()  # scalar
    
    return min_rgb_loss

def align_depth(target_depth, pred_depth, mask):
    target_masked = target_depth[mask]
    pred_masked = pred_depth[mask]

    n = torch.sum(mask)
    sum_pred = torch.sum(pred_masked)
    sum_target = torch.sum(target_masked)
    sum_pred_sq = torch.sum(pred_masked * pred_masked)
    sum_pred_target = torch.sum(pred_masked * target_masked)

    denominator = n * sum_pred_sq - sum_pred * sum_pred
    s = (n * sum_pred_target - sum_pred * sum_target) / denominator
    t = (sum_pred_sq * sum_target - sum_pred * sum_pred_target) / denominator
    aligned_depth = s * pred_depth + t
    return aligned_depth

def splat_pixels(rgb, refined_image, ixt, c2w, depth, threshold=0.5, sh_degree=3):
    diff = torch.mean(rgb - refined_image, dim=-1)
    y_coords, x_coords = torch.where(diff > threshold)
    if y_coords.numel() == 0:
        return None
    
    depths = depth[y_coords, x_coords]
    pixel_coords_homo = torch.stack([x_coords.float(), y_coords.float(), torch.ones_like(x_coords).float()], dim=1)
    ixt_inv = torch.inverse(ixt)
    p_c = (ixt_inv @ pixel_coords_homo.T).T * depths.unsqueeze(1)

    positions = (c2w[:3, :3] @ p_c.T).T + c2w[:3, 3]
    colors = torch.zeros((y_coords.numel(), (sh_degree + 1) ** 2, 3), device=positions.device)  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(refined_image[y_coords, x_coords])
    x_scales = depths / (math.sqrt(2) * ixt[0,0])
    y_scales = depths / (math.sqrt(2) * ixt[1,1])
    scales = torch.log(torch.stack([x_scales, y_scales, (x_scales + y_scales)/2], dim=1))
    rotations = torch.zeros((positions.shape[0], 4), device=positions.device)
    rotations[:, 0] = 1.0 # (w, x, y, z)
    opacities = torch.logit(torch.full((positions.shape[0],), 0.9, device=positions.device))
    return {
        'means': positions,
        'scales': scales,
        'quats': rotations,
        'sh0': colors[:, :1, :],
        'shN': colors[:, 1:, :],
        'opacities': opacities,
    }