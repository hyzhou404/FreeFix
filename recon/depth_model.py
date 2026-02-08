import torch
import cv2
import numpy as np
from unidepth.models import UniDepthV2
from PIL import Image


def unproject_depth(im: torch.Tensor, depth: torch.Tensor, intrinsic: list, c2w: torch.Tensor, sample_per_frame: int = -1) -> torch.Tensor:
    fx, fy, cx, cy = intrinsic
    x = torch.arange(0, depth.shape[1])  # generate pixel coordinates
    y = torch.arange(0, depth.shape[0])
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    pixels = torch.vstack((xx.ravel(), yy.ravel())).T.reshape(-1, 2)
    pixels = pixels.to(depth.device)
    # unproject depth to pointcloud
    x = (pixels[..., 0] - cx) * depth.reshape(-1) / fx
    y = (pixels[..., 1] - cy) * depth.reshape(-1) / fy
    z = depth.reshape(-1)
    local_points = torch.stack([x, y, z], axis=1)
    local_colors = im.reshape(-1, im.shape[-1])
    if sample_per_frame > 0:
        if local_points.shape[0] < sample_per_frame:
            return None, None
        num_points = local_points.shape[0]
        sample_idx = torch.randperm(num_points)[:sample_per_frame]
        # sample_idx = torch.random.choice(
        #     torch.arange(local_points.shape[0]), sample_per_frame
        # )
        local_points = local_points[sample_idx]
        local_colors = local_colors[sample_idx]
    local_points_w = (c2w[:3, :3] @ local_points.T).T + c2w[:3, 3]
    return local_points_w, local_colors

def project2img(xyzs, rgbs, c2w, ixt, imsize, canvas=None):
    h, w = imsize
    w2c = torch.linalg.inv(c2w)
    # world to camera space
    xyzs_local = (w2c[:3, :3] @ xyzs.T).T + w2c[:3, 3]
    # camera to image
    uvz = (ixt[:3, :3] @ xyzs_local.T).T
    z = uvz[:, 2:3]
    uv = (uvz[:, :2] / z).long()
    invalid_mask = (z[:, 0] <= 0) | (uv[:, 0] < 0) | (uv[:, 0] >= w) | (uv[:, 1] < 0) | (uv[:, 1] >= h)
    if canvas is None:
        canvas = torch.zeros((h, w, 3)).to(rgbs.device)
    uv = uv[~invalid_mask]
    canvas[uv[:, 1], uv[:, 0], :] = rgbs[~invalid_mask]
    return canvas


class Metric3dModel:
    def __init__(self):
        # self.model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True).cuda().eval()
        self.model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", force_download=False)
        self.model = self.model.to("cuda")
        self.model.eval()
        self.mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        self.input_size = (616, 1064)

    def forward(self, rgb_path: str, intrinsic: torch.Tensor) -> torch.Tensor:
        # im = cv2.imread(rgb_path)[:, :, ::-1].copy()
        # rgb, intrinsic, pad_info = self.parse_for_metric3d(im, intrinsic)
        # depth, _, _ = self.model.inference({'input': rgb, 'cam_in': intrinsic})
        image = Image.open(rgb_path)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1)
        prediction = self.model.infer(image, intrinsic)
        depth = prediction["depth"][0][0].detach().cpu()
        # # un pad
        # depth = depth.squeeze()
        # depth = depth[pad_info[0] : depth.shape[0] - pad_info[1], pad_info[2] : depth.shape[1] - pad_info[3]]
        # # upsample to original size
        # depth = torch.nn.functional.interpolate(depth[None, None, :, :], im.shape[:2], mode='bilinear').squeeze()
        # return torch.from_numpy(im).float() / 255., depth.detach().cpu()
        return image.permute(1, 2, 0) / 255., depth.detach().cpu()

    def parse_for_metric3d(self, im, intrinsic): # [H, W, 3(RGB)]
        h, w = im.shape[:2]
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        rgb = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        # remember to scale intrinsic, hold depth
        intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = self.input_size[0] - h
        pad_w = self.input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - self.mean), self.std)
        rgb = rgb[None, :, :, :].cuda()
        return rgb, intrinsic, pad_info